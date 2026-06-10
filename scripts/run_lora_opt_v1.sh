#!/usr/bin/env bash
set -euo pipefail

cd "${LLADA_EVAL_ROOT:-/data/llada_eval}"

MODEL="${MODEL:-/data/hf/models/GSAI-ML/LLaDA-8B-Instruct}"
ROOT="${ROOT:-results/domain_shift/task_aware/lora_opt_v1}"
TRAIN="${TRAIN:-$ROOT/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl}"
TASKS="${TASKS:-mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq}"
LIMIT="${LIMIT:-50}"
SEED="${SEED:-23}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

mkdir -p "$ROOT"/{adapters,raw,logs,tables,reports}

run_train_eval() {
  local name="$1"
  local variant="$2"
  local mode="$3"
  local steps="$4"
  local lr="$5"
  local extra="${6:-}"
  local out="$ROOT/adapters/${name}_steps${steps}"
  local adapter="$out/final_adapter"
  local raw="$ROOT/raw/llada_${name}_fixed32_limit${LIMIT}_seed${SEED}.json"

  if [ ! -f "$adapter/adapter_model.safetensors" ] && [ ! -f "$adapter/nara_adapter.pt" ]; then
    echo "[$(date)] train $name variant=$variant mode=$mode steps=$steps lr=$lr"
    # shellcheck disable=SC2086
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/train_llada_choice_noise_lora.py \
      --model "$MODEL" \
      --train-jsonl "$TRAIN" \
      --out "$out" \
      --peft-variant "$variant" \
      --mode "$mode" \
      --max-steps "$steps" \
      --grad-accum 8 \
      --batch-size 1 \
      --lr "$lr" \
      --seed "$SEED" \
      --target-modules "$TARGET_MODULES" \
      $extra \
      2>&1 | tee "$ROOT/logs/train_${name}_steps${steps}.log"
  else
    echo "[$(date)] skip train $name; adapter exists"
  fi

  if [ ! -f "$raw" ]; then
    echo "[$(date)] eval $name"
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/eval_domain_shift.py \
      --backend llada \
      --model "$MODEL" \
      --adapter "$adapter" \
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
    echo "[$(date)] skip eval $name; raw exists"
  fi
}

run_train_eval "balanced_vanilla_r8_s100" "lora" "vanilla" 100 "1e-4"
run_train_eval "balanced_vanilla_r8_highnoise_s100" "lora" "vanilla" 100 "1e-4" "--noise-ratios 0.65,0.85,1.0"
run_train_eval "balanced_vanilla_r16_s100" "lora" "vanilla" 100 "1e-4" "--lora-r 16 --lora-alpha 32"
run_train_eval "balanced_loraplus_r16_s100" "loraplus" "vanilla" 100 "1e-4" "--lora-r 16 --lora-alpha 32 --loraplus-lr-ratio 8"
run_train_eval "balanced_vanilla_r8_s150_lr5e5" "lora" "vanilla" 150 "5e-5"

echo "[$(date)] lora opt v1 queue done"
