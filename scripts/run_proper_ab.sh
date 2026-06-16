#!/usr/bin/env bash
# PROPER training mechanism (replaces the toy constant-LR/fixed-step and the bad train-loss early-stop):
#   fixed budget + warmup->cosine decay  +  held-out VALIDATION -> keep best-val checkpoint  +  weight decay.
# Clean A/B on the strongest recipe (choice_noise), full GPU. Answers: does PPL help when trained PROPERLY?
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/proper_ab
mkdir -p "$OD"/logs
BASE="--model $MODEL --train-jsonl $TRAIN --peft-variant lora --mode choice_noise --batch-size 1 --grad-accum 8 \
  --lr 1e-4 --lora-r 16 --lora-alpha 32 --mask-prompt --denoise-weight 0.15 --consistency-weight 0.05 \
  --noise-ratios 0.15,0.35,0.65,0.85 --lr-scheduler cosine --warmup-ratio 0.1 --max-steps 600 \
  --val-fraction 0.15 --val-every 40 --weight-decay 0.05"
PPL="--adaptive-noise reducible_loss --adaptive-where sampling --adaptive-signal denoise_ppl \
  --adaptive-eps 0.1 --adaptive-temp 0.5 --adaptive-ema 0.97 --adaptive-fast-ema 0.8"

run () {
  local name="$1"; shift; local extra="$*"; local out="$OD/$name"
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name (proper: cosine + val-selection + wd) :: $extra"
    # shellcheck disable=SC2086
    python3 scripts/train_llada_choice_noise_lora.py $BASE $extra --out "$out" > "$OD"/logs/train_"$name".log 2>&1 \
      || { echo "[$(date)] TRAIN $name FAILED"; tail -20 "$OD"/logs/train_"$name".log; return 1; }
  fi
  grep -oE '"selected_by_val": true, "best_val": [0-9.]+, "best_val_step": [0-9]+' "$OD"/logs/train_"$name".log | tail -1
  if [ ! -f "$OD/raw_$name.json" ]; then
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$out/final_adapter" \
      --tasks "$TASKS" --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
      --prompt-style final_label_typed --out "$OD"/raw_"$name".json > "$OD"/logs/eval_"$name".log 2>&1 \
      || { echo "[$(date)] EVAL $name FAILED"; return 1; }
  fi
  m=$(python3 -c "import json,statistics as s;d=json.load(open('$OD/raw_$name.json'));print('%.4f'%s.mean([v['accuracy'] for v in d['summary'].values()]))" 2>/dev/null)
  echo "[$(date)] DONE $name :: eval-450 macro=$m"
}

run best_ppl_v2     "$PPL"
run best_uniform_v2 "--adaptive-noise none"
echo "[$(date)] === PROPER A/B DONE (eval-450) ==="