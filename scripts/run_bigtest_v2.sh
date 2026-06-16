#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
OD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained
PA=$OD/proper_ab; BT=$OD/bigtest_v2; mkdir -p $BT/logs
for name in best_uniform_v2 best_ppl_v2; do
  [ -f $BT/raw_$name.json ] && continue
  echo "[$(date)] BIG-EVAL $name (200/task = 1800)"
  python3 scripts/eval_domain_shift.py --backend llada --model $MODEL --adapter $PA/$name/final_adapter \
    --tasks $TASKS --limit 200 --seed 23 --steps 32 --gen-length 32 --block-length 32 \
    --prompt-style final_label_typed --out $BT/raw_$name.json > $BT/logs/eval_$name.log 2>&1 || { echo "FAIL $name"; continue; }
  m=$(python3 -c "import json,statistics as s;d=json.load(open('$BT/raw_$name.json'));print('%.4f'%s.mean([v['accuracy'] for v in d['summary'].values()]))")
  echo "[$(date)] DONE $name :: macro=$m"
done
python3 - <<PY
import json,statistics as st,math
BT="$BT"
try:
  u=json.load(open(f"{BT}/raw_best_uniform_v2.json")); p=json.load(open(f"{BT}/raw_best_ppl_v2.json"))
  mu=st.mean([v['accuracy'] for v in u['summary'].values()]); mp=st.mean([v['accuracy'] for v in p['summary'].values()])
  N=sum(v['n'] for v in u['summary'].values()); sed=math.sqrt(2*0.76*0.24/N)
  print(f"[BIGTEST n={N}] uniform={mu:.4f} ppl={mp:.4f} diff={(mp-mu)*100:+.2f}pt sed={sed*100:.2f}pt -> {'REAL' if abs(mp-mu)>2*sed else 'within noise'}")
except Exception as e: print("pending",e)
PY
