#!/usr/bin/env bash
# CORRECTED goal: the earlier oracle early-exit checked "ceiling rise" — wrong for the call-minimization
# objective. Cheaper remask targets (8,12) don't raise the ceiling but let the MDP hit the SAME accuracy
# at FEWER calls. So collect the expanded TRAIN table and run the full-action MDP on expanded train+eval,
# then compare avg_calls at iso-accuracy vs the 2-action MDP (0.718@6.40).
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained
RAW=$OD/raw
EVAL_X=$RAW/mdpr_eval_expanded_seed23.json
TRAIN_X=$RAW/mdpr_trainpool_expanded_seed23.json

wait_for_gpu () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "$f" -ge 18000 ] && break; echo "[$(date)] wait GPU ${f}"; sleep 120; done; }

if [ ! -f "$TRAIN_X" ]; then
  wait_for_gpu
  echo "[$(date)] collect EXPANDED TRAIN table (cheaper targets 8,12 + label_only)"
  CHECKPOINTS="2,4,8,16,32" TARGETS="8,12,16,24,32" REMASK_FAMILIES="label_only,lowconf" NO_RESTART=1 \
    TRAIN_LIMIT=80 POOL=400 ROUNDS=12 OUT="$TRAIN_X" \
    python3 scripts/collect_cf_trainpool.py > "$OD"/logs/expand_train.log 2>&1 \
    || { echo "TRAIN COLLECT FAILED"; tail -15 "$OD"/logs/expand_train.log; exit 1; }
fi

echo "[$(date)] === full-action MDP on EXPANDED tables (does it cut avg_calls?) ==="
python3 scripts/eval_sequential_mdp_fullaction.py --train-table "$TRAIN_X" --eval-table "$EVAL_X" \
  --lambdas 0.0,0.005,0.01,0.015,0.02,0.03,0.05 > "$OD"/logs/expand_mdp.log 2>&1
echo "[$(date)] EXPANDED MDP RESULT:"; cat "$OD"/logs/expand_mdp.log
echo "[$(date)] reference 2-action MDP: 0.722@7.08, 0.718@6.40, 0.713@5.71"
