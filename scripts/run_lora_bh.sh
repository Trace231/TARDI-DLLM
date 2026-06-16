#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
AD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/proper_ab/best_ppl_v2/final_adapter
R=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/raw
LOG=results/domain_shift/task_aware/solid_v2/logs
OUT=$R/mdpr_lora_boolq_hellaswag_seed23.json
echo "[$(date)] collect LoRA boolq+hellaswag (limit150)"
python3 scripts/collect_llada_counterfactual_actions.py --model "$M" --adapter "$AD" --tasks boolq,hellaswag --limit 150 --seed 23 --checkpoints 2,4,8,16,32 --targets 16,32 --gen-length 32 --block-length 32 --out "$OUT" > "$LOG"/lora_bh.log 2>&1 && echo "[$(date)] BH DONE" || { echo "[$(date)] BH FAIL"; tail -6 "$LOG"/lora_bh.log; }
