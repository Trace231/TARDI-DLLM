#!/usr/bin/env bash
# v2 adaptive-noise: tests the MECHANISM FIX (per-example bucket attribution) + variance reduction.
# Fair A/B: matched baseline vs improved adaptive at the SAME batch/steps, so any gain is attributable
# to the fixed curriculum mechanism, not just larger batch. 9-task balanced train, 9-task eval-450.
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/adaptive_noise_v2
mkdir -p "$OD"/logs

# Variance reduction: batch_size 2 (smoother per-step + clean per-example attribution), 300 steps.
# eff batch = 2*4 = 8. mode=label (single forward, no consistency double-forward) -> fits alongside the
# running collection AND is a cleaner isolation of the per-example curriculum fix (no consistency confound).
COMMON="--model $MODEL --train-jsonl $TRAIN --peft-variant lora --mode label \
  --max-steps 300 --grad-accum 4 --batch-size 2 --lr 1e-4 --seed 23 --lora-r 16 --lora-alpha 32 \
  --mask-prompt --denoise-weight 0.15 --noise-ratios 0.15,0.35,0.65,0.85"

echo "[$(date)] SMOKE (2 steps, validates per-example attribution path)"
# shellcheck disable=SC2086
python3 scripts/train_llada_choice_noise_lora.py $COMMON --adaptive-noise reducible_loss --max-steps 2 \
  --out "$OD"/smoke > "$OD"/logs/smoke.log 2>&1
if [ $? -ne 0 ]; then echo "[$(date)] SMOKE FAILED"; tail -30 "$OD"/logs/smoke.log; exit 1; fi
echo "[$(date)] smoke ok"

train_eval () {
  local name="$1"; local extra="$2"; local out="$OD/$name"
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name"
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
  echo "[$(date)] DONE $name"
}

# matched baseline (no curriculum) and improved adaptive (reducible-loss + slow stable EMA + exploration floor)
train_eval baseline_b2s300 "--adaptive-noise none"
train_eval reducible_fixed "--adaptive-noise reducible_loss --adaptive-eps 0.1 --adaptive-temp 0.5 --adaptive-ema 0.97 --adaptive-fast-ema 0.8"
echo "[$(date)] V2 A/B DONE"
