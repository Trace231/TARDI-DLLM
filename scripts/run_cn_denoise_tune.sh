#!/usr/bin/env bash
# Tune denoise-PPL adaptive sampling ON the polished choice_noise lineage to try to BEAT 0.7711.
# Prior: a single aggressive config gave 0.7622 (<0.7711) -> aggressive sampling perturbs the well-tuned
# uniform. So sweep the GENTLE region (high temp / high eps = near-uniform + mild tilt to learnable buckets),
# plus a couple of medium settings. Exact 0.7711 config (choice_noise/batch1/200/consistency0.05) + denoise sampling.
# GATED on GPU (collectors are running). Compare every run to 0.7711.
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/cn_denoise_tune
mkdir -p "$OD"/logs
BASE="--model $MODEL --train-jsonl $TRAIN --peft-variant lora --mode choice_noise --max-steps 200 \
  --grad-accum 8 --batch-size 1 --lr 1e-4 --seed 23 --lora-r 16 --lora-alpha 32 --mask-prompt \
  --denoise-weight 0.15 --consistency-weight 0.05 --noise-ratios 0.15,0.35,0.65,0.85 \
  --adaptive-noise reducible_loss --adaptive-where sampling --adaptive-signal denoise_ppl"

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
  local verdict; verdict=$(python3 -c "print('BEATS 0.7711' if $m>0.7711 else ('ties' if $m>=0.7700 else 'below'))" 2>/dev/null)
  echo "[$(date)] DONE $name :: macro=$m  ($verdict)"
}

# gentle region first (most likely to preserve 0.7711 + small gain), then medium
run gentle_t1_e3   "--adaptive-temp 1.0 --adaptive-eps 0.3 --adaptive-ema 0.95 --adaptive-fast-ema 0.8"
run vgentle_t2_e5  "--adaptive-temp 2.0 --adaptive-eps 0.5 --adaptive-ema 0.97 --adaptive-fast-ema 0.85"
run gentle_t1_e2   "--adaptive-temp 1.0 --adaptive-eps 0.2 --adaptive-ema 0.9 --adaptive-fast-ema 0.6"
run med_t07_e2     "--adaptive-temp 0.7 --adaptive-eps 0.2 --adaptive-ema 0.9 --adaptive-fast-ema 0.6"
run gentle_mix     "--adaptive-temp 1.0 --adaptive-eps 0.3 --adaptive-ema 0.95 --adaptive-fast-ema 0.8 --adaptive-signal mix"
echo "[$(date)] === CN DENOISE TUNE DONE ==="
grep -hE "DONE (gentle|vgentle|med)" "$OD"/logs/../logs/driver.log 2>/dev/null || true
