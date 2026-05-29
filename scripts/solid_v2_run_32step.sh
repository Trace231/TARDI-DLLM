#!/usr/bin/env bash
set -euo pipefail
cd /data/llada_eval
source scripts/env_llada.sh
mkdir -p results/domain_shift/task_aware/solid_v2/raw results/domain_shift/task_aware/solid_v2/logs
python scripts/eval_domain_shift.py \
  --backend llada \
  --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct \
  --tasks winogrande,commonsenseqa \
  --limit 1000 \
  --seed 23 \
  --steps 32 \
  --gen-length 32 \
  --block-length 32 \
  --prompt-style final_label_typed \
  --out results/domain_shift/task_aware/solid_v2/raw/llada8b_32step_wino_cqa_limit1000_seed23.json
