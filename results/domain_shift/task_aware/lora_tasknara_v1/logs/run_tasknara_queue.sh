#!/usr/bin/env bash
set -euo pipefail
cd /data/llada_eval
run_one() {
  local name="$1"; shift
  local extra="$*"
  local out="results/domain_shift/task_aware/lora_tasknara_v1/adapters/${name}_steps100"
  local raw="results/domain_shift/task_aware/lora_tasknara_v1/raw/llada_${name}_fixed32_limit50_seed23.json"
  if [ ! -f "$out/final_adapter/nara_adapter.pt" ]; then
    echo "[$(date)] train $name"
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/train_llada_choice_noise_lora.py \
      --model "/data/hf/models/GSAI-ML/LLaDA-8B-Instruct" \
      --train-jsonl "results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl" \
      --out "$out" \
      --peft-variant tasknara \
      --mode vanilla \
      --max-steps 100 \
      --grad-accum 8 \
      --batch-size 1 \
      --lr 1e-4 \
      --seed 23 \
      --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
      --task-list "mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq" \
      --task-embedding-dim 32 \
      --nara-c-scale 0.1 \
      $extra \
      2>&1 | tee "results/domain_shift/task_aware/lora_tasknara_v1/logs/train_${name}_steps100.log"
  else
    echo "[$(date)] skip train $name"
  fi
  if [ ! -f "$raw" ]; then
    echo "[$(date)] eval $name"
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/eval_domain_shift.py \
      --backend llada \
      --model "/data/hf/models/GSAI-ML/LLaDA-8B-Instruct" \
      --adapter "$out/final_adapter" \
      --tasks "mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq" \
      --limit 50 \
      --seed 23 \
      --steps 32 \
      --gen-length 32 \
      --block-length 32 \
      --prompt-style final_label_typed \
      --out "$raw" \
      2>&1 | tee "results/domain_shift/task_aware/lora_tasknara_v1/logs/eval_${name}_fixed32_limit50.log"
  else
    echo "[$(date)] skip eval $name"
  fi
}
run_one tasknara_r8_highnoise_s100 --lora-r 8 --lora-alpha 16 --noise-ratios 0.65,0.85,1.0
run_one tasknara_r16_s100 --lora-r 16 --lora-alpha 32 --noise-ratios 0.15,0.35,0.65,0.85
echo "[$(date)] tasknara queue done"
