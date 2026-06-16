#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
AD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/proper_ab/best_ppl_v2/final_adapter
R=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/raw
LOG=results/domain_shift/task_aware/solid_v2/logs
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
TR=$R/mdpr_lora_trainpool_train150_seed23.json
EV=$R/mdpr_lora_eval_limit50_seed23.json
if [ ! -f "$TR" ]; then
  echo "[$(date)] collect LoRA mdpr TRAIN (limit150)"
  python3 scripts/collect_llada_counterfactual_actions.py --model "$M" --adapter "$AD" --tasks "$TASKS" --limit 150 --seed 23 --checkpoints 2,4,8,16,32 --targets 16,32 --gen-length 32 --block-length 32 --out "$TR" > "$LOG"/lora_mdp_train.log 2>&1 && echo "[$(date)] train done" || { echo "[$(date)] TRAIN FAIL"; tail -6 "$LOG"/lora_mdp_train.log; exit 1; }
fi
if [ ! -f "$EV" ]; then
  echo "[$(date)] collect LoRA mdpr EVAL (limit50)"
  python3 scripts/collect_llada_counterfactual_actions.py --model "$M" --adapter "$AD" --tasks "$TASKS" --limit 50 --seed 23 --checkpoints 2,4,8,16,32 --targets 16,32 --gen-length 32 --block-length 32 --out "$EV" > "$LOG"/lora_mdp_eval.log 2>&1 && echo "[$(date)] eval done" || { echo "[$(date)] EVAL FAIL"; tail -6 "$LOG"/lora_mdp_eval.log; exit 1; }
fi
echo "[$(date)] === remask MDP on LoRA (leak-free) ==="
python3 scripts/eval_sequential_remask_mdp.py --train-table "$TR" --eval-table "$EV" --lambdas 0.0,0.01,0.02,0.03 2>&1 | tee "$LOG"/lora_mdp_seq.log
echo "[$(date)] === LORA_MDP DONE ==="
