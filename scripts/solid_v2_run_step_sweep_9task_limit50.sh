#!/usr/bin/env bash
set -euo pipefail

cd "${LLADA_EVAL_ROOT:-/data/llada_eval}"
if [ -f scripts/env_llada.sh ]; then
  source scripts/env_llada.sh
fi

ROOT="${ROOT:-results/domain_shift/task_aware/solid_v2/step_sweep_limit50_4to32}"
MODEL="${MODEL:-/data/hf/models/GSAI-ML/LLaDA-8B-Instruct}"
TASKS="${TASKS:-mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq}"
LIMIT="${LIMIT:-50}"
SEED="${SEED:-23}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS_LIST="${STEPS_LIST:-4 8 12 16 20 24 28 32}"

mkdir -p "$ROOT/raw" "$ROOT/logs" "$ROOT/tables"

for steps in $STEPS_LIST; do
  out="$ROOT/raw/llada8b_fixed_steps${steps}_9task_limit${LIMIT}_seed${SEED}.json"
  log="$ROOT/logs/llada8b_fixed_steps${steps}_9task_limit${LIMIT}_seed${SEED}.log"
  if [ -f "$out" ]; then
    echo "[step-sweep] skip steps=$steps; $out exists"
    continue
  fi
  echo "[step-sweep] run steps=$steps tasks=$TASKS limit=$LIMIT batch=$BATCH_SIZE"
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    python scripts/eval_domain_shift.py \
      --backend llada \
      --model "$MODEL" \
      --tasks "$TASKS" \
      --limit "$LIMIT" \
      --batch-size "$BATCH_SIZE" \
      --seed "$SEED" \
      --steps "$steps" \
      --gen-length 32 \
      --block-length 32 \
      --prompt-style final_label_typed \
      --out "$out" \
      2>&1 | tee "$log"
done

python scripts/analyze_step_sweep_results.py \
  --root "$ROOT" \
  --out "$ROOT/tables/step_sweep_9task_limit${LIMIT}_seed${SEED}.csv"

echo "[step-sweep] completed"
