#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
AD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/proper_ab/best_ppl_v2/final_adapter
RAW=results/domain_shift/task_aware/solid_v2/raw
LOG=results/domain_shift/task_aware/solid_v2/logs
mkdir -p "$RAW" "$LOG"
C=$RAW/llada8b_lora_32step_wino_cqa_limit1000_seed23.json
D=$RAW/llada8b_lora_calibrated_wino_cqa_limit1000_seed23.json
if [ ! -f "$C" ]; then
  echo "[$(date)] START C: LoRA @ 32 step"
  python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$AD" \
    --tasks winogrande,commonsenseqa --limit 1000 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
    --prompt-style final_label_typed --out "$C" > "$LOG"/combo_lora_32step.log 2>&1 && echo "[$(date)] DONE C" || echo "[$(date)] FAIL C"
fi
if [ ! -f "$D" ]; then
  echo "[$(date)] START D: LoRA + calibrated controller"
  python3 scripts/eval_llada_budget_controller.py --model "$MODEL" --adapter "$AD" \
    --tasks winogrande,commonsenseqa --limit 1000 --seed 23 --gen-length 32 --block-length 32 \
    --temperature 0.0 --cfg 0.0 --remasking low_confidence \
    --binary-direct-fallback-threshold 0.7 --binary-medium-threshold 0.7 --multi-disagreement-policy ignore \
    --out "$D" > "$LOG"/combo_lora_calibrated.log 2>&1 && echo "[$(date)] DONE D" || echo "[$(date)] FAIL D"
fi
echo "[$(date)] === COMBO DONE ==="
