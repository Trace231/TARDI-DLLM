#!/usr/bin/env bash
# Consistent-protocol BIG test: evaluate the CONVERGED models on a 4x larger eval (200/task = 1800)
# to cut variance (SE ~1pt vs ~2pt at 450) and get directly comparable numbers. Same protocol for all.
# Gated: waits for both converged adapters to exist (best_ppl trains+evals, then best_uniform).
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained
CB=$OD/converged_best
BT=$OD/bigtest_200; mkdir -p "$BT"/logs
LIMIT=200

# candidates: the two CONVERGED models (the clean A/B). adapter path -> output name
declare -A CAND=(
  ["$CB/best_ppl/final_adapter"]="best_ppl"
  ["$CB/best_uniform/final_adapter"]="best_uniform"
)

wait_for () { while [ ! -f "$1" ]; do echo "[$(date)] waiting for $1"; sleep 120; done; }
wait_for_gpu () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "$f" -ge 16000 ] && break; sleep 120; done; }

for ad in "${!CAND[@]}"; do
  name="${CAND[$ad]}"
  wait_for "$ad"
  if [ ! -f "$BT/raw_$name.json" ]; then
    wait_for_gpu
    echo "[$(date)] BIG-EVAL $name (limit $LIMIT = $((LIMIT*9)) samples)"
    python3 scripts/eval_domain_shift.py --backend llada --model "$MODEL" --adapter "$ad" \
      --tasks "$TASKS" --limit $LIMIT --seed 23 --steps 32 --gen-length 32 --block-length 32 \
      --prompt-style final_label_typed --out "$BT"/raw_"$name".json > "$BT"/logs/eval_"$name".log 2>&1 \
      || { echo "[$(date)] BIG-EVAL $name FAILED"; tail -15 "$BT"/logs/eval_"$name".log; continue; }
  fi
  m=$(python3 -c "import json,statistics as s;d=json.load(open('$BT/raw_$name.json'));print('%.4f'%s.mean([v['accuracy'] for v in d['summary'].values()]))" 2>/dev/null)
  echo "[$(date)] DONE $name :: macro=$m  (n=$((LIMIT*9)))"
done

echo "[$(date)] === BIG TEST SUMMARY (n=$((LIMIT*9)), SE~1pt) ==="
python3 - <<PY
import json,statistics as st,math
BT="$BT"
for n in ["best_uniform","best_ppl"]:
    try:
        d=json.load(open(f"{BT}/raw_{n}.json")); accs=[v['accuracy'] for v in d['summary'].values()]
        N=sum(v['n'] for v in d['summary'].values()); m=st.mean(accs); se=math.sqrt(m*(1-m)/N)
        print(f"  {n:12} macro={m:.4f}  n={N}  SE={se*100:.2f}pt")
    except Exception as e: print(f"  {n}: pending ({e})")
try:
    u=json.load(open(f"{BT}/raw_best_uniform.json")); p=json.load(open(f"{BT}/raw_best_ppl.json"))
    mu=st.mean([v['accuracy'] for v in u['summary'].values()]); mp=st.mean([v['accuracy'] for v in p['summary'].values()])
    N=sum(v['n'] for v in u['summary'].values()); sed=math.sqrt(2*0.76*0.24/N)
    diff=mp-mu
    print(f"  PPL - uniform = {diff*100:+.2f}pt   (diff SE ~{sed*100:.2f}pt -> {'REAL' if abs(diff)>2*sed else 'within noise'})")
except Exception: pass
PY
