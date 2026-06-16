#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/matched_grid_v1
mkdir -p "$OD"/logs
BASE="--model $MODEL --train-jsonl $TRAIN --mode choice_noise --batch-size 1 --grad-accum 8 --lr 1e-4 --lora-r 16 --lora-alpha 32 --mask-prompt --denoise-weight 0.15 --consistency-weight 0.05 --noise-ratios 0.15,0.35,0.65,0.85 --lr-scheduler cosine --warmup-ratio 0.1 --val-fraction 0.15 --val-every 40 --weight-decay 0.05 --max-steps 600 --adaptive-noise none"
runb () {
  local name="$1"; local variant="$2"; local out="$OD/$name"
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name (peft=$variant, 600 steps)"
    python3 scripts/train_llada_choice_noise_lora.py $BASE --peft-variant "$variant" --out "$out" > "$OD"/logs/train_"$name".log 2>&1 || { echo "[$(date)] TRAIN $name FAILED"; tail -15 "$OD"/logs/train_"$name".log; return 1; }
  fi
  if [ ! -f "$OD/raw_$name.json" ]; then
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$out/final_adapter" --tasks "$TASKS" --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32 --prompt-style final_label_typed --out "$OD"/raw_"$name".json > "$OD"/logs/eval_"$name".log 2>&1 || { echo "[$(date)] EVAL $name FAILED"; return 1; }
  fi
  m=$(python3 -c "import json,statistics as s;d=json.load(open(\"$OD/raw_$name.json\"));print(\"%.4f\"%s.mean([v[\"accuracy\"] for v in d[\"summary\"].values()]))" 2>/dev/null)
  echo "[$(date)] DONE $name :: macro=$m"
}
runb rslora_600   rslora
runb nara_600     nara
runb dora_600     dora
runb loraplus_600 loraplus
echo "[$(date)] === GRID600 DONE ==="
