#!/usr/bin/env bash
set -euo pipefail

cd /data/llada_eval
source scripts/env_llada.sh

ROOT="results/domain_shift/task_aware/solid_v2"
RAW="$ROOT/raw"
LOGS="$ROOT/logs"
mkdir -p "$RAW" "$LOGS" "$ROOT/tables" "$ROOT/figures" "$ROOT/reports"

MODEL="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct"

wait_for_baseline() {
  while pgrep -af "eval_domain_shift.py .*llada8b_32step_wino_cqa_limit1000_seed23.json" >/dev/null; do
    echo "[solid_v2] waiting for 32-step baseline..."
    sleep 60
  done
  if [[ ! -s "$RAW/llada8b_32step_wino_cqa_limit1000_seed23.json" ]]; then
    echo "[solid_v2] missing 32-step baseline JSON; aborting remaining queue" >&2
    exit 2
  fi
}

run_if_missing() {
  local out="$1"
  local log="$2"
  shift 2
  if [[ -s "$out" ]]; then
    echo "[solid_v2] skip existing $out"
    return
  fi
  echo "[solid_v2] running $out"
  "$@" > "$log" 2>&1
}

wait_for_baseline

run_if_missing \
  "$RAW/llada8b_forward_aware_wino_cqa_limit1000_seed23.json" \
  "$LOGS/forward_aware_limit1000_seed23.log" \
  python scripts/eval_llada_forward_aware_router.py \
    --model "$MODEL" \
    --tasks winogrande,commonsenseqa \
    --limit 1000 \
    --seed 23 \
    --binary-confidence-threshold 0.70 \
    --out "$RAW/llada8b_forward_aware_wino_cqa_limit1000_seed23.json"

for t in 0.60 0.65 0.70 0.75 0.80; do
  tag="$(echo "$t" | tr -d '.')"
  run_if_missing \
    "$RAW/llada8b_calibrated_sweep_t${tag}_wino_cqa_limit500_seed23.json" \
    "$LOGS/sweep_t${tag}_limit500_seed23.log" \
    python scripts/eval_llada_budget_controller.py \
      --model "$MODEL" \
      --tasks winogrande,commonsenseqa \
      --limit 500 \
      --seed 23 \
      --binary-direct-fallback-threshold "$t" \
      --binary-medium-threshold "$t" \
      --multi-disagreement-policy ignore \
      --out "$RAW/llada8b_calibrated_sweep_t${tag}_wino_cqa_limit500_seed23.json"
done

run_if_missing \
  "$RAW/llada8b_8step_boundary_pubmed_ceval_limit300_seed23.json" \
  "$LOGS/boundary_8step_pubmed_ceval_limit300_seed23.log" \
  python scripts/eval_domain_shift.py \
    --backend llada \
    --model "$MODEL" \
    --tasks pubmedqa,ceval_computer_network \
    --limit 300 \
    --seed 23 \
    --steps 8 \
    --gen-length 32 \
    --block-length 32 \
    --prompt-style final_label_typed \
    --out "$RAW/llada8b_8step_boundary_pubmed_ceval_limit300_seed23.json"

run_if_missing \
  "$RAW/llada8b_32step_boundary_pubmed_ceval_limit300_seed23.json" \
  "$LOGS/boundary_32step_pubmed_ceval_limit300_seed23.log" \
  python scripts/eval_domain_shift.py \
    --backend llada \
    --model "$MODEL" \
    --tasks pubmedqa,ceval_computer_network \
    --limit 300 \
    --seed 23 \
    --steps 32 \
    --gen-length 32 \
    --block-length 32 \
    --prompt-style final_label_typed \
    --out "$RAW/llada8b_32step_boundary_pubmed_ceval_limit300_seed23.json"

run_if_missing \
  "$RAW/llada8b_calibrated_boundary_pubmed_ceval_limit300_seed23.json" \
  "$LOGS/boundary_calibrated_pubmed_ceval_limit300_seed23.log" \
  python scripts/eval_llada_budget_controller.py \
    --model "$MODEL" \
    --tasks pubmedqa,ceval_computer_network \
    --limit 300 \
    --seed 23 \
    --binary-direct-fallback-threshold 0.70 \
    --binary-medium-threshold 0.70 \
    --multi-disagreement-policy ignore \
    --out "$RAW/llada8b_calibrated_boundary_pubmed_ceval_limit300_seed23.json"

python scripts/solid_v2_analyze.py --root "$ROOT" --legacy-domain-root results/domain_shift \
  > "$LOGS/analyze_solid_v2.log" 2>&1

echo "[solid_v2] all queued experiments completed"
