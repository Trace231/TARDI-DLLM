#!/usr/bin/env bash
# Shot at beating the polished 0.7711: apply the WINNING denoise-PPL sampling to the choice_noise lineage
# (the config that produced 0.7711). Exact old config (batch1/200steps/consistency0.05) + denoise sampling.
# GATED: waits until >=28GB GPU is free (so it doesn't OOM against the running highgrid/eval).
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/choice_noise_denoise_v1
mkdir -p "$OD"/logs

wait_for_gpu () {  # wait until >= $1 MiB free
  local need="$1"
  while true; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$need" ] && break
    echo "[$(date)] waiting for GPU: ${free}MiB free < ${need}MiB"; sleep 60
  done
}

# exact 0.7711 lineage config
COMMON="--model $MODEL --train-jsonl $TRAIN --peft-variant lora --mode choice_noise \
  --max-steps 200 --grad-accum 8 --batch-size 1 --lr 1e-4 --seed 23 --lora-r 16 --lora-alpha 32 \
  --mask-prompt --denoise-weight 0.15 --consistency-weight 0.05"
DEN="--adaptive-noise reducible_loss --adaptive-where sampling --adaptive-signal denoise_ppl \
  --adaptive-eps 0.1 --adaptive-temp 0.5 --adaptive-ema 0.97 --adaptive-fast-ema 0.8"

train_eval () {
  local name="$1"; shift; local extra="$*"; local out="$OD/$name"
  wait_for_gpu 28000
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name :: $extra"
    # shellcheck disable=SC2086
    python3 scripts/train_llada_choice_noise_lora.py $COMMON $extra --out "$out" > "$OD"/logs/train_"$name".log 2>&1 \
      || { echo "[$(date)] TRAIN $name FAILED"; tail -25 "$OD"/logs/train_"$name".log; return 1; }
  fi
  if [ ! -f "$OD/raw_$name.json" ]; then
    echo "[$(date)] EVAL $name"
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$out/final_adapter" \
      --tasks "$TASKS" --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
      --prompt-style final_label_typed --out "$OD"/raw_"$name".json > "$OD"/logs/eval_"$name".log 2>&1 \
      || { echo "[$(date)] EVAL $name FAILED"; tail -25 "$OD"/logs/eval_"$name".log; return 1; }
  fi
  echo "[$(date)] DONE $name :: $(python3 -c "import json,statistics as s;d=json.load(open('$OD/raw_$name.json'));print('macro=%.4f'%s.mean([v['accuracy'] for v in d['summary'].values()]))" 2>/dev/null || echo pending)"
}

train_eval cn_denoise          "--noise-ratios 0.15,0.35,0.65,0.85 $DEN"
train_eval cn_denoise_highgrid "--noise-ratios 0.35,0.55,0.75,0.9 $DEN"
train_eval cn_baseline_recheck "--noise-ratios 0.15,0.35,0.65,0.85 --adaptive-noise none"
echo "[$(date)] CHOICE_NOISE+DENOISE DONE"
