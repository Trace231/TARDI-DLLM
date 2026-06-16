#!/usr/bin/env bash
# A/B test for option C (online loss-aware adaptive noise) vs baseline TARDI-style LoRA.
# Same config for both; only --adaptive-noise differs. 9-task balanced train, 9-task eval.
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/adaptive_noise_ab
mkdir -p "$OD"/logs

COMMON="--model $MODEL --train-jsonl $TRAIN --peft-variant lora --mode choice_noise \
  --max-steps 200 --grad-accum 8 --batch-size 1 --lr 1e-4 --seed 23 --lora-r 16 --lora-alpha 32 \
  --mask-prompt --denoise-weight 0.15 --consistency-weight 0.05 --noise-ratios 0.15,0.35,0.65,0.85"

echo "[$(date)] SMOKE adaptive (2 steps) to validate code path"
# shellcheck disable=SC2086
python3 scripts/train_llada_choice_noise_lora.py $COMMON --adaptive-noise loss_aware --max-steps 2 \
  --out "$OD"/smoke > "$OD"/logs/smoke.log 2>&1
if [ $? -ne 0 ]; then echo "[$(date)] SMOKE FAILED — aborting A/B"; tail -25 "$OD"/logs/smoke.log; exit 1; fi
echo "[$(date)] smoke ok"

train_eval () {
  local name="$1"; local extra="$2"
  local out="$OD/$name"
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name"
    # shellcheck disable=SC2086
    python3 scripts/train_llada_choice_noise_lora.py $COMMON $extra --out "$out" > "$OD"/logs/train_"$name".log 2>&1 \
      || { echo "[$(date)] TRAIN $name FAILED"; tail -20 "$OD"/logs/train_"$name".log; return 1; }
  fi
  if [ ! -f "$OD/raw_$name.json" ]; then
    echo "[$(date)] EVAL $name"
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$out/final_adapter" \
      --tasks "$TASKS" --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
      --prompt-style final_label_typed --out "$OD"/raw_"$name".json > "$OD"/logs/eval_"$name".log 2>&1 \
      || { echo "[$(date)] EVAL $name FAILED"; tail -20 "$OD"/logs/eval_"$name".log; return 1; }
  fi
}

train_eval baseline "--adaptive-noise none"
train_eval adaptive "--adaptive-noise loss_aware --adaptive-eps 0.2 --adaptive-temp 0.5 --adaptive-ema 0.9"
echo "[$(date)] A/B DONE"
