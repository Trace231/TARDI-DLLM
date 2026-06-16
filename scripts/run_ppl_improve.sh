#!/usr/bin/env bash
# Improve/confirm the PPL win on the line where it WORKS (label mode, matched A/B).
# Best so far: denoise-PPL sampling 0.7644 vs uniform 0.7400 (+2.44). Two goals:
#  (A) multi-seed CONFIRM the +2.44 is not single-seed noise (uniform vs ppl at seeds 7, 41)
#  (B) TUNE the PPL sampler to beat 0.7644 (sharper temp / faster ema / less exploration)
# GATED on GPU. label mode batch2 300 steps, grid 0.15..0.85.
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/ppl_improve_v1
mkdir -p "$OD"/logs
BASE="--model $MODEL --train-jsonl $TRAIN --peft-variant lora --mode label --max-steps 300 --grad-accum 4 \
  --batch-size 2 --lr 1e-4 --lora-r 16 --lora-alpha 32 --mask-prompt --denoise-weight 0.15 --noise-ratios 0.15,0.35,0.65,0.85"
PPL="--adaptive-noise reducible_loss --adaptive-where sampling --adaptive-signal denoise_ppl \
  --adaptive-eps 0.1 --adaptive-temp 0.5 --adaptive-ema 0.97 --adaptive-fast-ema 0.8"

wait_for_gpu () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "$f" -ge 20000 ] && break; echo "[$(date)] wait GPU ${f}"; sleep 120; done; }

run () {
  local name="$1"; shift; local extra="$*"; local out="$OD/$name"
  wait_for_gpu
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name :: $extra"
    # shellcheck disable=SC2086
    python3 scripts/train_llada_choice_noise_lora.py $BASE $extra --out "$out" > "$OD"/logs/train_"$name".log 2>&1 \
      || { echo "[$(date)] TRAIN $name FAILED"; tail -20 "$OD"/logs/train_"$name".log; return 1; }
  fi
  if [ ! -f "$OD/raw_$name.json" ]; then
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$out/final_adapter" \
      --tasks "$TASKS" --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
      --prompt-style final_label_typed --out "$OD"/raw_"$name".json > "$OD"/logs/eval_"$name".log 2>&1 \
      || { echo "[$(date)] EVAL $name FAILED"; return 1; }
  fi
  local m; m=$(python3 -c "import json,statistics as s;d=json.load(open('$OD/raw_$name.json'));print('%.4f'%s.mean([v['accuracy'] for v in d['summary'].values()]))" 2>/dev/null)
  echo "[$(date)] DONE $name :: macro=$m"
}

# (A) multi-seed confirm +2.44 (uniform vs PPL at seeds 7, 41)
run uniform_s7  "--seed 7  --adaptive-noise none"
run ppl_s7      "--seed 7  $PPL"
run uniform_s41 "--seed 41 --adaptive-noise none"
run ppl_s41     "--seed 41 $PPL"
# (B) tune to beat 0.7644 (seed 23)
run ppl_temp03  "--seed 23 $PPL --adaptive-temp 0.3"
run ppl_ema9    "--seed 23 --adaptive-noise reducible_loss --adaptive-where sampling --adaptive-signal denoise_ppl --adaptive-eps 0.1 --adaptive-temp 0.5 --adaptive-ema 0.9 --adaptive-fast-ema 0.6"
echo "[$(date)] === PPL IMPROVE DONE ==="