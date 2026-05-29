#!/usr/bin/env bash
set -euo pipefail

cd /data/llada_eval
source scripts/env_llada.sh

ROOT="results/domain_shift/task_aware/solid_v2"
DYN="$ROOT/dynamic_controller"
MODEL="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct"
TASKS="winogrande,commonsenseqa,arc_challenge,hellaswag,boolq"
LIMIT=300
SEED=23

mkdir -p "$DYN/raw" "$DYN/logs" "$DYN/tables" "$DYN/reports"

while pgrep -f 'solid_v2_run_external_sampler_baselines.sh|eval_llada_sampler_variants.py|eval_llada_prophet_early_commit.py' >/dev/null; do
  echo "[dynamic-controller] waiting for external sampler baselines to finish..."
  sleep 60
done

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
  --out "$DYN/raw/llada8b_refinement_controller_wino_cqa_arc_hella_boolq_limit300_seed23.json" \
  2>&1 | tee "$DYN/logs/refinement_controller_limit300_seed23.log"

python3 scripts/eval_llada_risk_controller.py \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --budgets 8,16,24,32 \
  --risk-t16 0.24 \
  --risk-t24 0.38 \
  --risk-t32 0.56 \
  --multi-disagreement-policy ignore \
  --out "$DYN/raw/llada8b_risk_controller_wino_cqa_arc_hella_boolq_limit300_seed23.json" \
  2>&1 | tee "$DYN/logs/risk_controller_limit300_seed23.log"

python3 scripts/eval_llada_multibudget_controller.py \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --budgets 8,16,24,32 \
  --multi-disagreement-policy ignore \
  --out "$DYN/raw/llada8b_multibudget_controller_wino_cqa_arc_hella_boolq_limit300_seed23.json" \
  2>&1 | tee "$DYN/logs/multibudget_controller_limit300_seed23.log"

python3 scripts/analyze_dynamic_controller.py \
  --solid-root "$ROOT" \
  --dynamic-root "$DYN" \
  2>&1 | tee "$DYN/logs/analyze_dynamic_controller.log"
