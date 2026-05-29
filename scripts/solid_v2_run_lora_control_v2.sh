#!/usr/bin/env bash
set -euo pipefail
cd /data/llada_eval
source scripts/env_llada.sh

ROOT=results/domain_shift/task_aware/solid_v2
TRAIN_JSON=$ROOT/lora_control_v2/train/domain_mix_final_typed_control_seed23.jsonl
QWEN_OUT=results/adaptation/qwen25_domain_mix_final_typed_control_lora_r8_steps200
LLADA_OUT=results/adaptation/llada_domain_mix_final_typed_control_grpo_lora_r8_steps200
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa
MODEL_Q=/data/hf/models/Qwen/Qwen2.5-7B-Instruct
MODEL_L=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct

mkdir -p "$ROOT/raw" "$ROOT/logs" "$ROOT/tables" "$ROOT/reports"

echo "[control_v2] qwen base eval"
python scripts/eval_domain_shift.py \
  --backend ar \
  --model "$MODEL_Q" \
  --tasks "$TASKS" \
  --limit 100 \
  --seed 7 \
  --prompt-style final_label_typed \
  --max-new-tokens 32 \
  --out "$ROOT/raw/qwen25_7b_control_base_final_label_typed_domain_shift_limit100.json" \
  > "$ROOT/logs/control_v2_qwen_base_eval.log" 2>&1

echo "[control_v2] qwen lora train"
python scripts/train_qwen_json_lora.py \
  --model "$MODEL_Q" \
  --train-jsonl "$TRAIN_JSON" \
  --out "$QWEN_OUT" \
  --seed 23 \
  --max-steps 200 \
  --max-length 1024 \
  --lr 1e-4 \
  > "$ROOT/logs/control_v2_qwen_train.log" 2>&1

echo "[control_v2] qwen lora eval"
python scripts/eval_domain_shift.py \
  --backend ar \
  --model "$MODEL_Q" \
  --adapter "$QWEN_OUT/final_adapter" \
  --tasks "$TASKS" \
  --limit 100 \
  --seed 7 \
  --prompt-style final_label_typed \
  --max-new-tokens 32 \
  --out "$ROOT/raw/qwen25_7b_control_lora_final_label_typed_domain_shift_limit100.json" \
  > "$ROOT/logs/control_v2_qwen_lora_eval.log" 2>&1

echo "[control_v2] llada base eval"
python scripts/eval_domain_shift.py \
  --backend llada \
  --model "$MODEL_L" \
  --tasks "$TASKS" \
  --limit 100 \
  --seed 7 \
  --steps 32 \
  --gen-length 32 \
  --block-length 32 \
  --prompt-style final_label_typed \
  --out "$ROOT/raw/llada8b_control_base_final_label_typed_domain_shift_limit100.json" \
  > "$ROOT/logs/control_v2_llada_base_eval.log" 2>&1

echo "[control_v2] llada lora train"
cd /data/llada_eval/dllm
source /data/llada_eval/scripts/env_llada.sh
accelerate launch --num_processes 1 /data/llada_eval/scripts/train_llada_json_grpo.py \
  --model_name_or_path "$MODEL_L" \
  --trust_remote_code true \
  --dtype bfloat16 \
  --load_in_4bit true \
  --use_peft true \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --train_jsonl "/data/llada_eval/$TRAIN_JSON" \
  --max_steps 200 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_generations 2 \
  --max_completion_length 32 \
  --generation_batch_size 2 \
  --steps 32 \
  --block_size 32 \
  --temperature 0.7 \
  --cfg_scale 0.0 \
  --remasking low_confidence \
  --p_mask_prompt 0.15 \
  --beta 0.04 \
  --epsilon 0.5 \
  --learning_rate 3e-6 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 1 \
  --report_to none \
  --output_dir "/data/llada_eval/$LLADA_OUT" \
  > "/data/llada_eval/$ROOT/logs/control_v2_llada_train.log" 2>&1
cd /data/llada_eval

echo "[control_v2] llada lora eval"
python scripts/eval_domain_shift.py \
  --backend llada \
  --model "$MODEL_L" \
  --adapter "$LLADA_OUT/checkpoint-200" \
  --tasks "$TASKS" \
  --limit 100 \
  --seed 7 \
  --steps 32 \
  --gen-length 32 \
  --block-length 32 \
  --prompt-style final_label_typed \
  --out "$ROOT/raw/llada8b_control_lora_final_label_typed_domain_shift_limit100.json" \
  > "$ROOT/logs/control_v2_llada_lora_eval.log" 2>&1

echo "[control_v2] analyze"
python scripts/analyze_lora_control_v2.py --root "$ROOT" > "$ROOT/logs/control_v2_analyze.log" 2>&1

echo "[control_v2] completed"
