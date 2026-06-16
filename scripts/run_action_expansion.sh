#!/usr/bin/env bash
# Efficiency-oriented action-set expansion (overnight, GATED on GPU after LoRA cn).
# Adds CHEAP recovery actions to push the anytime frontier LEFT (fewer steps at same acc):
#   targets 8,16,32  (the cheap '8' lets remask recover without jumping to 16/32)
#   families label_only,lowconf  (label_only = relabel only, very cheap)  ; NO restart (anti-efficiency)
# SMART EARLY-EXIT: collect EVAL first, recompute ORACLE; if the ceiling doesn't rise, skip the
# expensive TRAIN collection (expansion won't help) and just report.
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained
RAW=$OD/raw; mkdir -p "$OD"/logs
EVAL_X=$RAW/mdpr_eval_expanded_seed23.json
TRAIN_X=$RAW/mdpr_trainpool_expanded_seed23.json
TGT="8,12,16,24,32"; FAM="label_only,lowconf"; CKPT="2,4,8,16,32"

wait_for_gpu () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "$f" -ge "$1" ] && break; echo "[$(date)] wait GPU ${f}<$1"; sleep 120; done; }

oracle () {  # macro accuracy ceiling = best correct over ALL checkpoints x actions
  python3 - "$1" <<'PY'
import json,sys,numpy as np
d=json.load(open(sys.argv[1])); pt={}
for s in d["samples"]:
    best=0.0
    for st in s["states"]:
        if st.get("state_correct"): best=1.0
        for a in st.get("actions",[]):
            if a.get("correct"): best=1.0
    pt.setdefault(s["task"],[]).append(best)
print("%.4f"%np.mean([np.mean(v) for v in pt.values()]))
PY
}

echo "[$(date)] === ACTION EXPANSION start (gated) ==="
wait_for_gpu 18000

echo "[$(date)] collect EVAL table (expanded actions)"
python3 scripts/collect_llada_counterfactual_actions.py --model "$MODEL" --tasks "$TASKS" --limit 50 --seed 23 \
  --checkpoints "$CKPT" --targets "$TGT" --remask-families "$FAM" --no-include-restart \
  --out "$EVAL_X" > "$OD"/logs/expand_eval.log 2>&1 || { echo "EVAL COLLECT FAILED"; tail -15 "$OD"/logs/expand_eval.log; exit 1; }

OLD_ORACLE=$(oracle "$RAW/mdpr_eval_limit50_seed23.json")
NEW_ORACLE=$(oracle "$EVAL_X")
echo "[$(date)] ORACLE  old(2 actions)=$OLD_ORACLE  ->  new(expanded)=$NEW_ORACLE"

# early-exit if ceiling didn't rise meaningfully (+0.01)
RISE=$(python3 -c "print(1 if $NEW_ORACLE-$OLD_ORACLE>=0.01 else 0)")
if [ "$RISE" -ne 1 ]; then
  echo "[$(date)] oracle did NOT rise >= +0.01 ($OLD_ORACLE -> $NEW_ORACLE). Expansion won't lift the ceiling."
  echo "[$(date)] skipping expensive TRAIN collection. DONE (negative-but-informative result)."
  exit 0
fi
echo "[$(date)] oracle rose -> worth it. Collecting TRAIN pool (expanded)."

CHECKPOINTS="$CKPT" TARGETS="$TGT" REMASK_FAMILIES="$FAM" NO_RESTART=1 \
  TRAIN_LIMIT=100 POOL=500 ROUNDS=14 OUT="$TRAIN_X" \
  python3 scripts/collect_cf_trainpool.py > "$OD"/logs/expand_train.log 2>&1 || { echo "TRAIN COLLECT FAILED"; tail -15 "$OD"/logs/expand_train.log; exit 1; }

echo "[$(date)] full-action MDP on EXPANDED tables (CPU)"
OMP_NUM_THREADS=16 python3 scripts/eval_sequential_mdp_fullaction.py \
  --train-table "$TRAIN_X" --eval-table "$EVAL_X" --lambdas 0.0,0.005,0.01,0.02,0.03,0.05 \
  > "$OD"/logs/expand_mdp.log 2>&1
echo "[$(date)] === EXPANSION DONE. results: ==="; cat "$OD"/logs/expand_mdp.log
