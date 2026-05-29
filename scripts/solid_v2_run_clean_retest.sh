#!/usr/bin/env bash
set -euo pipefail

cd /data/llada_eval
source scripts/env_llada.sh

ROOT="results/domain_shift/task_aware/solid_v2"
OUT="$ROOT/clean_retest"
MODEL="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct"
TASKS="winogrande,commonsenseqa,arc_challenge,hellaswag,boolq"
LIMIT="${LIMIT:-50}"
SEED="${SEED:-23}"

mkdir -p "$OUT/raw" "$OUT/logs" "$OUT/tables" "$OUT/reports"

run_sampler() {
  local name="$1"
  local steps="$2"
  local schedule="$3"
  python3 scripts/eval_llada_sampler_variants.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --steps "$steps" \
    --gen-length 32 \
    --block-length 32 \
    --schedule "$schedule" \
    --prompt-style final_label_typed \
    --out "$OUT/raw/${name}_limit${LIMIT}_seed${SEED}.json" \
    2>&1 | tee "$OUT/logs/${name}_limit${LIMIT}_seed${SEED}.log"
}

run_sampler "llada8b_fixed8_back_loaded" 8 back_loaded
run_sampler "llada8b_fixed16_back_loaded" 16 back_loaded
run_sampler "llada8b_jys_like_middle16" 16 middle_heavy
run_sampler "llada8b_fixed32_uniform" 32 uniform

python3 scripts/eval_llada_prophet_early_commit.py \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --max-steps 32 \
  --min-steps 8 \
  --check-interval 4 \
  --patience 2 \
  --gen-length 32 \
  --block-length 32 \
  --schedule uniform \
  --out "$OUT/raw/llada8b_prophet_early_commit_limit${LIMIT}_seed${SEED}.json" \
  2>&1 | tee "$OUT/logs/prophet_early_commit_limit${LIMIT}_seed${SEED}.log"

python3 scripts/eval_llada_refinement_controller.py \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --budgets 8,16,24,32 \
  --risk-t16 0.24 \
  --risk-t24 0.38 \
  --risk-t32 0.56 \
  --multi-disagreement-policy ignore \
  --out "$OUT/raw/llada8b_refinement_controller_limit${LIMIT}_seed${SEED}.json" \
  2>&1 | tee "$OUT/logs/refinement_controller_limit${LIMIT}_seed${SEED}.log"

python3 scripts/audit_eval_outputs.py "$OUT/raw" \
  --out "$OUT/tables/output_audit_limit${LIMIT}_seed${SEED}.csv" \
  --max-invalid-rate 0.02 \
  2>&1 | tee "$OUT/logs/output_audit_limit${LIMIT}_seed${SEED}.log" || true

python3 scripts/analyze_clean_retest.py \
  --root "$OUT" \
  2>&1 | tee "$OUT/logs/analyze_clean_retest_limit${LIMIT}_seed${SEED}.log"
