#!/usr/bin/env bash
set -euo pipefail

cd /data/llada_eval
source scripts/env_llada.sh

ROOT="results/domain_shift/task_aware/solid_v2/external_sampler_baselines"
RAW="$ROOT/raw"
LOGS="$ROOT/logs"
TABLES="$ROOT/tables"
mkdir -p "$RAW" "$LOGS" "$TABLES"

MODEL="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct"
TASKS="winogrande,commonsenseqa,arc_challenge,hellaswag,boolq"
LIMIT=300
SEED=23

run_if_missing() {
  local out="$1"
  shift
  if [[ -s "$out" ]]; then
    echo "[external] skip existing $out"
  else
    "$@"
  fi
}

run_if_missing "$RAW/llada8b_jys_like_middle16_wino_cqa_arc_hella_boolq_limit${LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_llada_sampler_variants.py \
    --model $MODEL \
    --tasks $TASKS \
    --limit $LIMIT \
    --seed $SEED \
    --steps 16 \
    --gen-length 32 \
    --block-length 32 \
    --schedule middle_heavy \
    --prompt-style final_label_typed \
    --out $RAW/llada8b_jys_like_middle16_wino_cqa_arc_hella_boolq_limit${LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/jys_like_middle16_limit${LIMIT}_seed${SEED}.log"

run_if_missing "$RAW/llada8b_jys_like_back16_wino_cqa_arc_hella_boolq_limit${LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_llada_sampler_variants.py \
    --model $MODEL \
    --tasks $TASKS \
    --limit $LIMIT \
    --seed $SEED \
    --steps 16 \
    --gen-length 32 \
    --block-length 32 \
    --schedule back_loaded \
    --prompt-style final_label_typed \
    --out $RAW/llada8b_jys_like_back16_wino_cqa_arc_hella_boolq_limit${LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/jys_like_back16_limit${LIMIT}_seed${SEED}.log"

run_if_missing "$RAW/llada8b_prophet_early_commit_wino_cqa_arc_hella_boolq_limit${LIMIT}_seed${SEED}.json" \
  bash -lc "python scripts/eval_llada_prophet_early_commit.py \
    --model $MODEL \
    --tasks $TASKS \
    --limit $LIMIT \
    --seed $SEED \
    --max-steps 32 \
    --min-steps 8 \
    --check-interval 4 \
    --patience 2 \
    --gen-length 32 \
    --block-length 32 \
    --schedule uniform \
    --out $RAW/llada8b_prophet_early_commit_wino_cqa_arc_hella_boolq_limit${LIMIT}_seed${SEED}.json \
    2>&1 | tee $LOGS/prophet_early_commit_limit${LIMIT}_seed${SEED}.log"

python scripts/analyze_external_sampler_baselines.py \
  --root "$ROOT" \
  2>&1 | tee "$LOGS/analyze_external_sampler_baselines.log"

echo "[external] completed"
