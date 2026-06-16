#!/usr/bin/env bash
# ITERATIVE (depth-2) MDP collection: remask is no longer terminal -- after a remask we capture the
# post-remask state and its own actions (next_state), so the remask axis becomes multi-step.
# GATED after the widened depth-1 expansion (PID passed as $1). SMART EARLY-EXIT: compute the
# iterative oracle vs the depth-1 oracle; only worth a learned tree-solver if the ceiling rises.
# Modest sample (depth-2 collection is ~10x costlier per sample).
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained
RAW=$OD/raw; mkdir -p "$OD"/logs
ITER=$RAW/mdpr_eval_iter_depth2_seed23.json
GATE_PID="${1:-}"

[ -n "$GATE_PID" ] && { echo "[$(date)] gate on widened expansion PID $GATE_PID"; while kill -0 "$GATE_PID" 2>/dev/null; do sleep 120; done; echo "[$(date)] gate released"; }
while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "$f" -ge 18000 ] && break; echo "[$(date)] wait GPU ${f}"; sleep 120; done

echo "[$(date)] ITERATIVE depth-2 collection (modest sample)"
python3 scripts/collect_llada_counterfactual_actions.py --model "$MODEL" --tasks "$TASKS" --limit 20 --seed 23 \
  --checkpoints 2,4,8 --targets 8,16,32 --remask-families label_only,lowconf --no-include-restart \
  --remask-depth 2 --child-families lowconf --child-targets 16,32 \
  --out "$ITER" > "$OD"/logs/iter_collect.log 2>&1 || { echo "ITER COLLECT FAILED"; tail -15 "$OD"/logs/iter_collect.log; exit 1; }

echo "[$(date)] === ITERATIVE ORACLE: does multi-step remask raise the ceiling? ==="
python3 - "$ITER" <<'PY'
import json,sys,numpy as np
d=json.load(open(sys.argv[1]))
def d1(st):
    b=1.0 if st.get("state_correct") else 0.0
    for a in st.get("actions",[]):
        if a.get("correct"): b=1.0
    return b
def d2(st):
    b=1.0 if st.get("state_correct") else 0.0
    for a in st.get("actions",[]):
        if a.get("action","").startswith("accept"):
            if a.get("correct"): b=1.0
        elif "next_state" in a: b=max(b,d2(a["next_state"]))
        else:
            if a.get("correct"): b=1.0
    return b
pt1,pt2={},{}
for s in d["samples"]:
    pt1.setdefault(s["task"],[]).append(max(d1(st) for st in s["states"]))
    pt2.setdefault(s["task"],[]).append(max(d2(st) for st in s["states"]))
o1=np.mean([np.mean(v) for v in pt1.values()]); o2=np.mean([np.mean(v) for v in pt2.values()])
print(f"  depth-1 oracle (single remask)   = {o1:.4f}")
print(f"  depth-2 oracle (iterative remask)= {o2:.4f}")
print(f"  GAP (iterative recovers)         = {o2-o1:+.4f}")
print("  VERDICT:", "ITERATIVE RAISES CEILING -> build learned tree-solver" if o2-o1>=0.01 else "iterative does NOT raise ceiling meaningfully")
PY
echo "[$(date)] === ITERATIVE MDP probe DONE ==="
