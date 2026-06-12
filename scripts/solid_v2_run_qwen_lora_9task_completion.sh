#!/usr/bin/env bash
set -euo pipefail

cd "${LLADA_EVAL_ROOT:-/data/llada_eval}"
if [ -f scripts/env_llada.sh ]; then
  source scripts/env_llada.sh
fi

ROOT="${ROOT:-results/domain_shift/task_aware/solid_v2}"
MODEL_Q="${MODEL_Q:-/data/hf/models/Qwen/Qwen2.5-7B-Instruct}"
TRAIN_JSON="${TRAIN_JSON:-results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl}"
QWEN_OUT="${QWEN_OUT:-results/adaptation/qwen25_9task_final_typed_lora_r8_steps200}"
TASKS="${TASKS:-mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq}"
LIMIT="${LIMIT:-50}"
SEED="${SEED:-23}"
STEPS="${STEPS:-200}"

mkdir -p "$ROOT/raw" "$ROOT/logs" "$ROOT/tables"

BASE_OUT="$ROOT/raw/qwen25_7b_9task_base_final_label_typed_limit${LIMIT}_seed${SEED}.json"
LORA_OUT="$ROOT/raw/qwen25_7b_9task_lora_final_label_typed_limit${LIMIT}_seed${SEED}.json"

echo "[qwen-9task] root=$ROOT"
echo "[qwen-9task] tasks=$TASKS limit=$LIMIT seed=$SEED"

if [ ! -f "$QWEN_OUT/final_adapter/adapter_model.safetensors" ]; then
  echo "[qwen-9task] train Qwen LoRA from $TRAIN_JSON"
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    python scripts/train_qwen_json_lora.py \
      --model "$MODEL_Q" \
      --train-jsonl "$TRAIN_JSON" \
      --out "$QWEN_OUT" \
      --seed "$SEED" \
      --max-steps "$STEPS" \
      --max-length 1024 \
      --lr 1e-4 \
      2>&1 | tee "$ROOT/logs/qwen_9task_lora_train_steps${STEPS}.log"
else
  echo "[qwen-9task] skip train; adapter exists at $QWEN_OUT/final_adapter"
fi

if [ ! -f "$BASE_OUT" ]; then
  echo "[qwen-9task] eval Qwen base"
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    python scripts/eval_domain_shift.py \
      --backend ar \
      --model "$MODEL_Q" \
      --tasks "$TASKS" \
      --limit "$LIMIT" \
      --seed "$SEED" \
      --prompt-style final_label_typed \
      --max-new-tokens 32 \
      --out "$BASE_OUT" \
      2>&1 | tee "$ROOT/logs/qwen_9task_base_limit${LIMIT}_seed${SEED}.log"
else
  echo "[qwen-9task] skip base eval; $BASE_OUT exists"
fi

if [ ! -f "$LORA_OUT" ]; then
  echo "[qwen-9task] eval Qwen LoRA"
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    python scripts/eval_domain_shift.py \
      --backend ar \
      --model "$MODEL_Q" \
      --adapter "$QWEN_OUT/final_adapter" \
      --tasks "$TASKS" \
      --limit "$LIMIT" \
      --seed "$SEED" \
      --prompt-style final_label_typed \
      --max-new-tokens 32 \
      --out "$LORA_OUT" \
      2>&1 | tee "$ROOT/logs/qwen_9task_lora_limit${LIMIT}_seed${SEED}.log"
else
  echo "[qwen-9task] skip lora eval; $LORA_OUT exists"
fi

python scripts/analyze_qwen_lora_9task.py \
  --base "$BASE_OUT" \
  --lora "$LORA_OUT" \
  --out "$ROOT/tables/qwen_lora_9task_limit${LIMIT}_seed${SEED}.csv"

echo "[qwen-9task] completed"
