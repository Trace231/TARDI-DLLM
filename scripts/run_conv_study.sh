#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for SEED in 7 31; do
  for tag in uniform cont; do
    out=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/conv_study/${tag}_s$SEED
    [ -f $out/final_adapter/adapter_model.safetensors ] && continue
    extra="--noise-ratios 0.15,0.35,0.65,0.85 --adaptive-noise none"; [ "$tag" = cont ] && extra="--noise-ratios 0.1,0.25,0.4,0.55,0.7,0.8,0.9 --noise-jitter 0.07 --adaptive-noise reducible_loss --adaptive-where sampling --adaptive-signal choice_ppl --adaptive-eps 0.03 --adaptive-temp 0.2 --adaptive-ema 0.95 --adaptive-fast-ema 0.7"
    echo "[$(date)] TRAIN $tag seed$SEED"
    python3 scripts/train_llada_choice_noise_lora.py --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct --train-jsonl results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl --peft-variant lora       --mode choice_noise --batch-size 1 --grad-accum 8 --lr 1e-4 --lora-r 16 --lora-alpha 32 --mask-prompt --denoise-weight 0.15 --consistency-weight 0.05 --lr-scheduler cosine --warmup-ratio 0.1 --max-steps 600 --val-fraction 0.15 --val-every 40 --weight-decay 0.05 $extra --seed $SEED --out $out > results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/conv_study/logs/${tag}_s$SEED.log 2>&1 || echo "FAIL $tag $SEED"
  done
done
echo "[$(date)] === CONVERGENCE (mean val_acc by step, 3 seeds incl existing) ==="
python3 - <<'PY'
import json,glob,re,statistics as st
OD="results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained"; CS="results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/conv_study"
def curve(log):
    txt=open(log).read(); return {int(m[0]):float(m[1]) for m in re.findall(r'"step": (\d+), "val_acc": ([0-9.]+)',txt)}
runs={'uniform':[OD+'/proper_ab/logs/train_best_uniform_v2.log', CS+'/uniform_s7.log', CS+'/uniform_s31.log'],
      'cont':[OD+'/best_v3/logs/train.log', CS+'/cont_s7.log', CS+'/cont_s31.log']}
import os
for name,logs in runs.items():
    cs=[curve(l) for l in logs if os.path.exists(l)]
    steps=sorted(set().union(*[set(c) for c in cs])) if cs else []
    mean={s:st.mean([c[s] for c in cs if s in c]) for s in steps}
    # step to first reach >=0.785
    reach=next((s for s in steps if mean.get(s,0)>=0.785),None)
    print(f"  {name:8} (n={len(cs)} seeds)  first val>=0.785 at step: {reach}")
    print('           ', ' '.join('%d:%.3f'%(s,mean[s]) for s in steps if s%80==0 or s==steps[-1]))
PY
