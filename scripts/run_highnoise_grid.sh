#!/usr/bin/env bash
# Tests the user's FIRST-ORDER lever: shift the noise buckets UP (drop the lowest 0.15, concentrate high).
# Uniform sampling, no adaptive -- a clean test of "are the low buckets just wasteful for this task".
# Compare to uniform_baseline (grid 0.15..0.85) macro 0.7400. Runs in PARALLEL using the idle ~32GB.
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/adaptive_noise_v3
mkdir -p "$OD"/logs

run () {
  local name="$1"; local grid="$2"; local out="$OD/$name"
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name grid=$grid"
    python3 scripts/train_llada_choice_noise_lora.py --model "$MODEL" --train-jsonl "$TRAIN" \
      --peft-variant lora --mode label --max-steps 300 --grad-accum 4 --batch-size 2 --lr 1e-4 --seed 23 \
      --lora-r 16 --lora-alpha 32 --mask-prompt --denoise-weight 0.15 --adaptive-noise none \
      --noise-ratios "$grid" --out "$out" > "$OD"/logs/train_"$name".log 2>&1 \
      || { echo "[$(date)] TRAIN $name FAILED"; tail -20 "$OD"/logs/train_"$name".log; return 1; }
  fi
  if [ ! -f "$OD/raw_$name.json" ]; then
    echo "[$(date)] EVAL $name"
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$out/final_adapter" \
      --tasks "$TASKS" --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
      --prompt-style final_label_typed --out "$OD"/raw_"$name".json > "$OD"/logs/eval_"$name".log 2>&1 \
      || { echo "[$(date)] EVAL $name FAILED"; tail -20 "$OD"/logs/eval_"$name".log; return 1; }
  fi
  echo "[$(date)] DONE $name :: $(python3 -c "import json,statistics as s;d=json.load(open('$OD/raw_$name.json'));print('macro=%.4f'%s.mean([v['accuracy'] for v in d['summary'].values()]))" 2>/dev/null || echo pending)"
}

run highnoise_grid "0.35,0.55,0.75,0.9"
echo "[$(date)] HIGHNOISE GRID DONE"
