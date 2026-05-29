#!/usr/bin/env bash
set -euo pipefail

cd /data/llada_eval
source scripts/env_llada.sh

ROOT="results/domain_shift/task_aware/solid_v2/coverage_addendum"
RAW="$ROOT/raw"
LOGS="$ROOT/logs"
TABLES="$ROOT/tables"
REPORTS="$ROOT/reports"
mkdir -p "$RAW" "$LOGS" "$TABLES" "$REPORTS"

LLADA_MODEL="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct"
QWEN_MODEL="/data/hf/models/Qwen/Qwen2.5-7B-Instruct"
SEED=23

run_if_missing() {
  local out="$1"
  shift
  if [[ -s "$out" ]]; then
    echo "[coverage] skip existing $out"
  else
    "$@"
  fi
}

CLOSED_TASKS="arc_challenge,hellaswag,boolq"
CLOSED_LIMIT=300
GSM_LIMIT=100

run_if_missing "$RAW/llada8b_8step_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_domain_shift.py \
    --backend llada \
    --model $LLADA_MODEL \
    --tasks $CLOSED_TASKS \
    --limit $CLOSED_LIMIT \
    --seed $SEED \
    --steps 8 \
    --gen-length 32 \
    --block-length 32 \
    --prompt-style final_label_typed \
    --out $RAW/llada8b_8step_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/llada8b_8step_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.log"

run_if_missing "$RAW/llada8b_32step_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_domain_shift.py \
    --backend llada \
    --model $LLADA_MODEL \
    --tasks $CLOSED_TASKS \
    --limit $CLOSED_LIMIT \
    --seed $SEED \
    --steps 32 \
    --gen-length 32 \
    --block-length 32 \
    --prompt-style final_label_typed \
    --out $RAW/llada8b_32step_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/llada8b_32step_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.log"

run_if_missing "$RAW/llada8b_calibrated_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_llada_budget_controller.py \
    --model $LLADA_MODEL \
    --tasks $CLOSED_TASKS \
    --limit $CLOSED_LIMIT \
    --seed $SEED \
    --binary-direct-fallback-threshold 0.70 \
    --binary-medium-threshold 0.70 \
    --multi-disagreement-policy ignore \
    --out $RAW/llada8b_calibrated_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/llada8b_calibrated_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.log"

run_if_missing "$RAW/qwen25_7b_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_domain_shift.py \
    --backend ar \
    --model $QWEN_MODEL \
    --tasks $CLOSED_TASKS \
    --limit $CLOSED_LIMIT \
    --seed $SEED \
    --max-new-tokens 32 \
    --prompt-style final_label_typed \
    --out $RAW/qwen25_7b_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/qwen25_7b_coverage_closed_limit${CLOSED_LIMIT}_seed${SEED}.log"

run_if_missing "$RAW/llada8b_8step_gsm8k_limit${GSM_LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_domain_shift.py \
    --backend llada \
    --model $LLADA_MODEL \
    --tasks gsm8k \
    --limit $GSM_LIMIT \
    --seed $SEED \
    --steps 8 \
    --gen-length 128 \
    --block-length 32 \
    --prompt-style final_label_typed \
    --out $RAW/llada8b_8step_gsm8k_limit${GSM_LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/llada8b_8step_gsm8k_limit${GSM_LIMIT}_seed${SEED}.log"

run_if_missing "$RAW/llada8b_32step_gsm8k_limit${GSM_LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_domain_shift.py \
    --backend llada \
    --model $LLADA_MODEL \
    --tasks gsm8k \
    --limit $GSM_LIMIT \
    --seed $SEED \
    --steps 32 \
    --gen-length 128 \
    --block-length 32 \
    --prompt-style final_label_typed \
    --out $RAW/llada8b_32step_gsm8k_limit${GSM_LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/llada8b_32step_gsm8k_limit${GSM_LIMIT}_seed${SEED}.log"

run_if_missing "$RAW/qwen25_7b_gsm8k_limit${GSM_LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_domain_shift.py \
    --backend ar \
    --model $QWEN_MODEL \
    --tasks gsm8k \
    --limit $GSM_LIMIT \
    --seed $SEED \
    --max-new-tokens 128 \
    --prompt-style final_label_typed \
    --out $RAW/qwen25_7b_gsm8k_limit${GSM_LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/qwen25_7b_gsm8k_limit${GSM_LIMIT}_seed${SEED}.log"

python scripts/analyze_coverage_addendum.py \
  --root "$ROOT" \
  2>&1 | tee "$LOGS/analyze_coverage_addendum.log"

echo "[coverage] completed"
