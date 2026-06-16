#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Confirm the BEST method (best_uniform_v2 = choice_noise recipe + proper training, uniform noise) is robustly >=0.771
# at multiple seeds on the LARGE test. seed 23 already = 0.7763.
for SEED in 17 41; do
  out=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/confirm_best/seed$SEED
  if [ ! -f $out/final_adapter/adapter_model.safetensors ]; then
    python3 scripts/train_llada_choice_noise_lora.py --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct --train-jsonl results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl --peft-variant lora       --mode choice_noise --batch-size 1 --grad-accum 8 --lr 1e-4 --lora-r 16 --lora-alpha 32 --mask-prompt       --denoise-weight 0.15 --consistency-weight 0.05 --noise-ratios 0.15,0.35,0.65,0.85 --adaptive-noise none       --lr-scheduler cosine --warmup-ratio 0.1 --max-steps 600 --val-fraction 0.15 --val-every 40 --weight-decay 0.05       --seed $SEED --out $out > results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/confirm_best/logs/train_s$SEED.log 2>&1
  fi
  python3 scripts/eval_domain_shift.py --backend llada --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct --adapter $out/final_adapter     --tasks mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq --limit 200 --seed 23 --steps 32 --gen-length 32 --block-length 32     --prompt-style final_label_typed --out results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/confirm_best/raw_s$SEED.json > results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/confirm_best/logs/eval_s$SEED.log 2>&1
  m=$(python3 -c "import json,statistics as s;d=json.load(open('results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/confirm_best/raw_s$SEED.json'));print('%.4f'%s.mean([v['accuracy'] for v in d['summary'].values()]))")
  echo "[$(date)] best_uniform_v2 seed$SEED BIG = $m"
done
python3 -c "
import json,statistics as st
vals=[0.7763]  # seed 23 (existing)
import glob
for f in sorted(glob.glob('results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/confirm_best/raw_s*.json')):
    d=json.load(open(f)); vals.append(st.mean([v['accuracy'] for v in d['summary'].values()]))
print('[$(date)] === BEST METHOD multi-seed BIG: ', [round(v,4) for v in vals], ' mean=%.4f >= 0.771? %s'%(st.mean(vals), st.mean(vals)>=0.771))"
