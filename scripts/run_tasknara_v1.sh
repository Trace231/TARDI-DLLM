#!/usr/bin/env bash
set -euo pipefail

cd "${LLADA_EVAL_ROOT:-/data/llada_eval}"

ROOT="${ROOT:-results/domain_shift/task_aware/lora_tasknara_v1}"
TRAIN="${TRAIN:-results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl}"
MODEL="${MODEL:-/data/hf/models/GSAI-ML/LLaDA-8B-Instruct}"
TASKS="${TASKS:-mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq}"
TASK_LIST="${TASK_LIST:-$TASKS}"
LIMIT="${LIMIT:-50}"
SEED="${SEED:-23}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

mkdir -p "$ROOT"/{adapters,raw,logs,tables,reports}

run_one() {
  local name="$1"
  shift
  local extra="$*"
  local out="$ROOT/adapters/${name}_steps100"
  local raw="$ROOT/raw/llada_${name}_fixed32_limit${LIMIT}_seed${SEED}.json"

  if [ ! -f "$out/final_adapter/nara_adapter.pt" ]; then
    echo "[$(date)] train $name"
    # shellcheck disable=SC2086
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/train_llada_choice_noise_lora.py \
      --model "$MODEL" \
      --train-jsonl "$TRAIN" \
      --out "$out" \
      --peft-variant tasknara \
      --mode vanilla \
      --max-steps 100 \
      --grad-accum 8 \
      --batch-size 1 \
      --lr 1e-4 \
      --seed "$SEED" \
      --target-modules "$TARGET_MODULES" \
      --task-list "$TASK_LIST" \
      --task-embedding-dim 32 \
      --nara-c-scale 0.1 \
      $extra \
      2>&1 | tee "$ROOT/logs/train_${name}_steps100.log"
  else
    echo "[$(date)] skip train $name"
  fi

  if [ ! -f "$raw" ]; then
    echo "[$(date)] eval $name"
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/eval_domain_shift.py \
      --backend llada \
      --model "$MODEL" \
      --adapter "$out/final_adapter" \
      --tasks "$TASKS" \
      --limit "$LIMIT" \
      --seed "$SEED" \
      --steps 32 \
      --gen-length 32 \
      --block-length 32 \
      --prompt-style final_label_typed \
      --out "$raw" \
      2>&1 | tee "$ROOT/logs/eval_${name}_fixed32_limit${LIMIT}.log"
  else
    echo "[$(date)] skip eval $name"
  fi
}

run_one tasknara_r8_highnoise_s100 --lora-r 8 --lora-alpha 16 --noise-ratios 0.65,0.85,1.0
run_one tasknara_r16_s100 --lora-r 16 --lora-alpha 32 --noise-ratios 0.15,0.35,0.65,0.85

echo "[$(date)] tasknara queue done"
