#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
AD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/proper_ab/best_ppl_v2/final_adapter
R=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/raw
LOG=results/domain_shift/task_aware/solid_v2/logs
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
CK=2,4,6,8,12,16,24,32
TR=$R/mdp_lora_trainpool_8ckpt_train60_seed23.json
EV=$R/mdp_lora_eval_8ckpt_limit50_seed23.json
if [ ! -f "$TR" ]; then
  echo "[$(date)] collect LoRA train pool"
  python3 scripts/collect_llada_counterfactual_actions.py --model "$MODEL" --adapter "$AD" --tasks "$TASKS" --limit 60 --seed 23 --checkpoints "$CK" --gen-length 32 --block-length 32 --out "$TR" > "$LOG"/mdp_lora_train.log 2>&1 && echo "[$(date)] done train" || { echo "[$(date)] FAIL train"; tail -8 "$LOG"/mdp_lora_train.log; exit 1; }
fi
if [ ! -f "$EV" ]; then
  echo "[$(date)] collect LoRA eval pool"
  python3 scripts/collect_llada_counterfactual_actions.py --model "$MODEL" --adapter "$AD" --tasks "$TASKS" --limit 50 --seed 23 --checkpoints "$CK" --gen-length 32 --block-length 32 --out "$EV" > "$LOG"/mdp_lora_eval.log 2>&1 && echo "[$(date)] done eval" || { echo "[$(date)] FAIL eval"; tail -8 "$LOG"/mdp_lora_eval.log; exit 1; }
fi
echo "[$(date)] === MDP on LoRA (leak-free) ==="
python3 scripts/eval_sequential_stopping_mdp.py --train-table "$TR" --eval-table "$EV" --lambdas 0.0,0.01,0.02,0.03 2>&1 | tee "$LOG"/mdp_lora_seq.log
echo "[$(date)] === MDP-LoRA DONE ==="
