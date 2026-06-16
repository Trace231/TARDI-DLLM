#!/usr/bin/env bash
# v3 adaptive-noise: difficulty-adaptive BANDIT sampler over noise buckets.
#   - per-(task,bucket) slow/fast EMA of a PER-EXAMPLE perplexity (fixed attribution, no batch-mean contamination)
#   - priority = reducible difficulty (slow-fast) -> upsample buckets still LEARNABLE
#   - softmax categorical over 4 buckets + eps exploration floor (bandit, NOT MCMC/MH)
# Fair A/B at matched batch/steps/mode (label). Answers two axes:
#   (1) does adaptive noise beat UNIFORM at all?   (baseline vs the rest)
#   (2) which perplexity is the right signal?      (denoise_ppl vs choice_ppl vs mix)
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/adaptive_noise_v3
mkdir -p "$OD"/logs

# label mode = single forward (fits alongside collection, isolates the noise-sampling change cleanly).
COMMON="--model $MODEL --train-jsonl $TRAIN --peft-variant lora --mode label \
  --max-steps 300 --grad-accum 4 --batch-size 2 --lr 1e-4 --seed 23 --lora-r 16 --lora-alpha 32 \
  --mask-prompt --denoise-weight 0.15 --noise-ratios 0.15,0.35,0.65,0.85"
# reducible-loss bandit, slow stable EMA, exploration floor 0.1
ADAPT="--adaptive-noise reducible_loss --adaptive-eps 0.1 --adaptive-temp 0.5 --adaptive-ema 0.97 --adaptive-fast-ema 0.8"

echo "[$(date)] SMOKE (3 steps, denoise_ppl bandit path)"
# shellcheck disable=SC2086
python3 scripts/train_llada_choice_noise_lora.py $COMMON $ADAPT --adaptive-signal denoise_ppl --max-steps 3 \
  --out "$OD"/smoke > "$OD"/logs/smoke.log 2>&1
if [ $? -ne 0 ]; then echo "[$(date)] SMOKE FAILED"; tail -30 "$OD"/logs/smoke.log; exit 1; fi
echo "[$(date)] smoke ok"; rm -rf "$OD"/smoke

train_eval () {
  local name="$1"; shift; local extra="$*"; local out="$OD/$name"
  if [ ! -f "$out/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date)] TRAIN $name :: $extra"
    # shellcheck disable=SC2086
    python3 scripts/train_llada_choice_noise_lora.py $COMMON $extra --out "$out" > "$OD"/logs/train_"$name".log 2>&1 \
      || { echo "[$(date)] TRAIN $name FAILED"; tail -25 "$OD"/logs/train_"$name".log; return 1; }
  fi
  if [ ! -f "$OD/raw_$name.json" ]; then
    echo "[$(date)] EVAL $name"
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$out/final_adapter" \
      --tasks "$TASKS" --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
      --prompt-style final_label_typed --out "$OD"/raw_"$name".json > "$OD"/logs/eval_"$name".log 2>&1 \
      || { echo "[$(date)] EVAL $name FAILED"; tail -25 "$OD"/logs/eval_"$name".log; return 1; }
  fi
  echo "[$(date)] DONE $name"
}

train_eval uniform_baseline "--adaptive-noise none"
train_eval denoise_ppl  "$ADAPT --adaptive-signal denoise_ppl"
train_eval choice_ppl   "$ADAPT --adaptive-signal choice_ppl"
train_eval mix_ppl      "$ADAPT --adaptive-signal mix"
echo "[$(date)] V3 DONE"
