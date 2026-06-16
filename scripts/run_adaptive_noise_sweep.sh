#!/usr/bin/env bash
# Tuning sweep for option C. Same base config as the A/B baseline (200 steps) so the
# only variable is the adaptive-noise mechanism/aggressiveness. Compare vs baseline 0.771
# and the un-tuned loss_aware 0.767.
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/adaptive_noise_ab
mkdir -p "$OD"/logs

COMMON="--model $MODEL --train-jsonl $TRAIN --peft-variant lora --mode choice_noise \
  --max-steps 200 --grad-accum 8 --batch-size 1 --lr 1e-4 --seed 23 --lora-r 16 --lora-alpha 32 \
  --mask-prompt --denoise-weight 0.15 --consistency-weight 0.05 --noise-ratios 0.15,0.35,0.65,0.85"

echo "[$(date)] SMOKE reducible_loss (2 steps)"
# shellcheck disable=SC2086
python3 scripts/train_llada_choice_noise_lora.py $COMMON --adaptive-noise reducible_loss --max-steps 2 \
  --out "$OD"/smoke_red > "$OD"/logs/smoke_red.log 2>&1 \
  || { echo "[$(date)] SMOKE reducible FAILED"; tail -25 "$OD"/logs/smoke_red.log; exit 1; }
echo "[$(date)] smoke reducible ok"

train_eval () {
  local name="$1"; local extra="$2"; local out="$OD/$name"
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name"
    # shellcheck disable=SC2086
    python3 scripts/train_llada_choice_noise_lora.py $COMMON $extra --out "$out" > "$OD"/logs/train_"$name".log 2>&1 \
      || { echo "[$(date)] TRAIN $name FAILED"; tail -20 "$OD"/logs/train_"$name".log; return 1; }
  fi
  if [ ! -f "$OD/raw_$name.json" ]; then
    echo "[$(date)] EVAL $name"
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$out/final_adapter" \
      --tasks "$TASKS" --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
      --prompt-style final_label_typed --out "$OD"/raw_"$name".json > "$OD"/logs/eval_"$name".log 2>&1 \
      || { echo "[$(date)] EVAL $name FAILED"; tail -20 "$OD"/logs/eval_"$name".log; return 1; }
  fi
  echo "[$(date)] DONE $name"
}

train_eval reducible_t05    "--adaptive-noise reducible_loss --adaptive-temp 0.5 --adaptive-fast-ema 0.6"
train_eval reducible_t10    "--adaptive-noise reducible_loss --adaptive-temp 1.0 --adaptive-fast-ema 0.6"
train_eval lossaware_soft   "--adaptive-noise loss_aware --adaptive-temp 1.0 --adaptive-eps 0.3"
echo "[$(date)] SWEEP DONE"
