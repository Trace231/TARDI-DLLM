#!/usr/bin/env bash
set -euo pipefail

cd "${LLADA_EVAL_ROOT:-/data/llada_eval}"

MODEL="${MODEL:-/data/hf/models/GSAI-ML/LLaDA-8B-Instruct}"
TRAIN="${TRAIN:-results/domain_shift/task_aware/solid_v2/lora_control_v2/train/domain_mix_final_typed_control_seed23.jsonl}"
ROOT="${ROOT:-results/domain_shift/task_aware/lora_external_v1}"
TASKS="${TASKS:-mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq}"
LIMIT="${LIMIT:-50}"
SEED="${SEED:-23}"

mkdir -p "$ROOT"/{adapters,raw,logs,tables,reports}

run_train_eval() {
  local name="$1"
  local variant="$2"
  local mode="$3"
  local steps="$4"
  local extra="${5:-}"
  local out="$ROOT/adapters/${name}_steps${steps}"
  local adapter="$out/final_adapter"
  local raw="$ROOT/raw/llada_${name}_fixed32_limit${LIMIT}_seed${SEED}.json"

  if [ ! -f "$adapter/adapter_model.safetensors" ] && [ ! -f "$adapter/nara_adapter.pt" ]; then
    echo "[$(date)] train $name variant=$variant mode=$mode steps=$steps"
    # shellcheck disable=SC2086
    python3 scripts/train_llada_choice_noise_lora.py \
      --model "$MODEL" \
      --train-jsonl "$TRAIN" \
      --out "$out" \
      --peft-variant "$variant" \
      --mode "$mode" \
      --max-steps "$steps" \
      --grad-accum 8 \
      --batch-size 1 \
      --lr 1e-4 \
      --seed "$SEED" \
      $extra \
      2>&1 | tee "$ROOT/logs/train_${name}_steps${steps}.log"
  else
    echo "[$(date)] skip train $name; adapter exists"
  fi

  if [ ! -f "$raw" ]; then
    echo "[$(date)] eval $name"
    python3 scripts/eval_domain_shift.py \
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

# External / improved LoRA baselines under the same denoising objective.
run_train_eval "rslora_vanilla" "rslora" "vanilla" 100
run_train_eval "dora_vanilla" "dora" "vanilla" 100
run_train_eval "nara_vanilla" "nara" "vanilla" 100 "--nara-buckets 4"

# Our NaRA-style extension: dynamic noise-aware adapter plus fixed-label objective.
run_train_eval "nara_choice_noise" "nara" "choice_noise" 150 "--nara-buckets 4 --denoise-weight 0.15 --consistency-weight 0.05"

python3 scripts/analyze_external_lora_baselines.py \
  --external-root "$ROOT" \
  --choice-root results/domain_shift/task_aware/choice_noise_v1 \
  --out-root "$ROOT"

echo "[$(date)] external LoRA baseline queue done"
