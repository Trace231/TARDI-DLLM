#!/usr/bin/env bash
# v5: runs my BEST BET first (combo: PPL on both ends), then the clean attribution A/B.
# Same OD as v3/v4 -> reuses uniform_baseline adapter. All mode=label, batch2, 300 steps, matched.
#   combo_denoise    : dynamic-noise SAMPLING + LOSS focal, both by denoise-PPL   <-- best bet, runs FIRST
#   uniform_baseline : no perplexity (reused)
#   denoise_ppl      : sampling only (attribution)
#   loss_denoise     : loss-weight only (attribution)
#   loss_choice      : loss-weight, choice signal (attribution)
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/adaptive_noise_v3
mkdir -p "$OD"/logs

COMMON="--model $MODEL --train-jsonl $TRAIN --peft-variant lora --mode label \
  --max-steps 300 --grad-accum 4 --batch-size 2 --lr 1e-4 --seed 23 --lora-r 16 --lora-alpha 32 \
  --mask-prompt --denoise-weight 0.15 --noise-ratios 0.15,0.35,0.65,0.85"
# combo sampling: FASTER EMA (adapts within 300 steps), some exploration; loss: moderate gamma
COMBO="--adaptive-noise reducible_loss --adaptive-where both --adaptive-signal denoise_ppl \
  --adaptive-eps 0.15 --adaptive-temp 0.4 --adaptive-ema 0.9 --adaptive-fast-ema 0.6 --adaptive-gamma 0.7"
EMA="--adaptive-eps 0.1 --adaptive-temp 0.5 --adaptive-ema 0.97 --adaptive-fast-ema 0.8"

train_eval () {
  local name="$1"; shift; local extra="$*"; local out="$OD/$name"
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name :: $extra"
    # shellcheck disable=SC2086
    python3 scripts/train_llada_choice_noise_lora.py $COMMON $extra --out "$out" > "$OD"/logs/train_"$name".log 2>&1 \
      || { echo "[$(date)] TRAIN $name FAILED"; tail -25 "$OD"/logs/train_"$name".log; return 1; }
  else echo "[$(date)] REUSE $name adapter"; fi
  if [ ! -f "$OD/raw_$name.json" ]; then
    echo "[$(date)] EVAL $name"
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$out/final_adapter" \
      --tasks "$TASKS" --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
      --prompt-style final_label_typed --out "$OD"/raw_"$name".json > "$OD"/logs/eval_"$name".log 2>&1 \
      || { echo "[$(date)] EVAL $name FAILED"; tail -25 "$OD"/logs/eval_"$name".log; return 1; }
  fi
  echo "[$(date)] DONE $name :: $(python3 -c "import json,statistics as s;d=json.load(open('$OD/raw_$name.json'));print('macro=%.4f'%s.mean([v['accuracy'] for v in d['summary'].values()]))" 2>/dev/null || echo '(macro pending)')"
}

train_eval uniform_baseline "--adaptive-noise none"
train_eval combo_denoise "$COMBO"
train_eval denoise_ppl  "--adaptive-noise reducible_loss --adaptive-where sampling    --adaptive-signal denoise_ppl $EMA"
train_eval loss_denoise "--adaptive-noise reducible_loss --adaptive-where loss_weight --adaptive-signal denoise_ppl --adaptive-gamma 1.0"
train_eval loss_choice  "--adaptive-noise reducible_loss --adaptive-where loss_weight --adaptive-signal choice_ppl  --adaptive-gamma 1.0"
echo "[$(date)] V5 DONE"
