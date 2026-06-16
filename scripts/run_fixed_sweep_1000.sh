#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
RAW=results/domain_shift/task_aware/solid_v2/raw
LOG=results/domain_shift/task_aware/solid_v2/logs
mkdir -p "$RAW" "$LOG"
for S in 2 4 8 16 24; do
  OUT=$RAW/llada8b_fixed${S}step_wino_cqa_limit1000_seed23.json
  if [ -f "$OUT" ]; then echo "[skip] $OUT exists"; continue; fi
  echo "[start] steps=$S $(date)"
  python scripts/eval_domain_shift.py \
    --backend llada \
    --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct \
    --tasks winogrande,commonsenseqa \
    --limit 1000 --seed 23 \
    --steps $S --gen-length 32 --block-length 32 \
    --prompt-style final_label_typed \
    --out "$OUT" > $LOG/fixed${S}_limit1000_seed23.log 2>&1
  echo "[done] steps=$S rc=$? $(date)"
done
echo "[ALL DONE] $(date)"
