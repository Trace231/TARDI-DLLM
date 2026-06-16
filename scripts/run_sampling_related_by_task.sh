#!/usr/bin/env bash
set -euo pipefail

cd "${LLADA_EVAL_ROOT:-/data/llada_eval}"
if [ -f scripts/env_llada.sh ]; then
  # shellcheck disable=SC1091
  source scripts/env_llada.sh
fi

MODEL="${MODEL:-/data/hf/models/GSAI-ML/LLaDA-8B-Instruct}"
ROOT="${ROOT:-results/domain_shift/task_aware/solid_v2/related_work_v1}"
TASKS="${TASKS:-mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq,gsm8k}"
LIMIT="${LIMIT:-50}"
SEED="${SEED:-23}"

mkdir -p "$ROOT"/{raw,logs,tables}

IFS=',' read -r -a TASK_ARR <<< "$TASKS"

run_eval_variant_task() {
  local method="$1"
  local task="$2"
  local steps="$3"
  local schedule="$4"
  local sampler="${5:-standard}"
  local extra="${6:-}"
  local out="$ROOT/raw/by_task__${method}__${task}_limit${LIMIT}_seed${SEED}.json"
  if [ -f "$out" ]; then
    echo "[$(date)] skip $method $task; exists: $out"
    return
  fi
  echo "[$(date)] run $method task=$task"
  # shellcheck disable=SC2086
  python3 scripts/eval_llada_sampler_variants.py \
    --model "$MODEL" \
    --tasks "$task" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --steps "$steps" \
    --gen-length 32 \
    --block-length 32 \
    --schedule "$schedule" \
    --sampler "$sampler" \
    --prompt-style final_label_typed \
    --out "$out" \
    $extra \
    2>&1 | tee "$ROOT/logs/by_task__${method}__${task}_limit${LIMIT}_seed${SEED}.log"
}

run_prophet_task() {
  local method="prophet_early_commit"
  local task="$1"
  local out="$ROOT/raw/by_task__${method}__${task}_limit${LIMIT}_seed${SEED}.json"
  if [ -f "$out" ]; then
    echo "[$(date)] skip $method $task; exists: $out"
    return
  fi
  echo "[$(date)] run $method task=$task"
  python3 scripts/eval_llada_prophet_early_commit.py \
    --model "$MODEL" \
    --tasks "$task" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --max-steps 32 \
    --min-steps 4 \
    --check-interval 2 \
    --patience 2 \
    --gen-length 32 \
    --block-length 32 \
    --schedule uniform \
    --out "$out" \
    2>&1 | tee "$ROOT/logs/by_task__${method}__${task}_limit${LIMIT}_seed${SEED}.log"
}

run_refinement_task() {
  local method="$1"
  local task="$2"
  local extra="$3"
  local out="$ROOT/raw/by_task__${method}__${task}_limit${LIMIT}_seed${SEED}.json"
  if [ -f "$out" ]; then
    echo "[$(date)] skip $method $task; exists: $out"
    return
  fi
  echo "[$(date)] run $method task=$task"
  # shellcheck disable=SC2086
  python3 scripts/eval_llada_refinement_controller.py \
    --model "$MODEL" \
    --tasks "$task" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --budgets 8,16,24,32 \
    --risk-t16 0.24 \
    --risk-t24 0.38 \
    --risk-t32 0.56 \
    --multi-disagreement-policy ignore \
    --out "$out" \
    $extra \
    2>&1 | tee "$ROOT/logs/by_task__${method}__${task}_limit${LIMIT}_seed${SEED}.log"
}

for task in "${TASK_ARR[@]}"; do
  run_eval_variant_task "fixed32_uniform" "$task" 32 uniform
  run_eval_variant_task "jys_like_middle16" "$task" 16 middle_heavy
  run_eval_variant_task "fixed16_uniform" "$task" 16 uniform
  run_eval_variant_task "front_loaded16" "$task" 16 front_loaded
  run_eval_variant_task "back_loaded16" "$task" 16 back_loaded
  run_eval_variant_task "pc_middle16" "$task" 16 middle_heavy predictor_corrector "--corrector-weight 0.5"
  run_prophet_task "$task"
  run_refinement_task "ours_refinement_lowconf" "$task" ""
  run_refinement_task "remdm_like_structured_remask" "$task" "--structured-remask"
  run_refinement_task "saber_like_answer_consistency" "$task" "--answer-consistency-remask"
done

python3 - <<'PY'
import csv, json, statistics
from pathlib import Path

import os

root = Path("results/domain_shift/task_aware/solid_v2/related_work_v1")
limit = os.environ.get("LIMIT", "50")
seed = os.environ.get("SEED", "23")
raw = root / "raw"
tables = root / "tables"
tables.mkdir(parents=True, exist_ok=True)

by_method = {}
task_rows = []
for p in sorted(raw.glob("by_task__*_limit*_seed*.json")):
    name = p.name[len("by_task__"):]
    method, rest = name.split("__", 1)
    task = rest.split("_limit", 1)[0]
    data = json.loads(p.read_text())
    summary = data.get("summary", {})
    s = summary.get(task) or next(iter(summary.values()), {})
    rows = data.get("rows", [])
    by_method.setdefault(method, {"summary": {}, "rows": []})
    by_method[method]["summary"][task] = s
    by_method[method]["rows"].extend(rows)
    task_rows.append({
        "method": method,
        "task": task,
        "n": s.get("n", len(rows)),
        "accuracy": s.get("accuracy", ""),
        "avg_forward_calls": s.get("avg_forward_calls", ""),
        "seconds": s.get("seconds", ""),
    })

for method, payload in by_method.items():
    out = raw / f"{method}_limit{limit}_seed{seed}.merged_by_task.json"
    out.write_text(json.dumps({
        "complete": True,
        "merged_by_task": True,
        "method": method,
        "summary": payload["summary"],
        "rows": payload["rows"],
    }, ensure_ascii=False, indent=2))

with (tables / "sampling_related_work_by_task.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "task", "n", "accuracy", "avg_forward_calls", "seconds"])
    w.writeheader()
    w.writerows(task_rows)

summary_rows = []
for method in sorted(by_method):
    xs = [r for r in task_rows if r["method"] == method]
    acc = [float(x["accuracy"]) for x in xs if x["accuracy"] != ""]
    calls = [float(x["avg_forward_calls"]) for x in xs if x["avg_forward_calls"] != ""]
    secs = [float(x["seconds"]) for x in xs if x["seconds"] != ""]
    summary_rows.append({
        "method": method,
        "tasks": len(xs),
        "macro_accuracy": statistics.mean(acc) if acc else "",
        "avg_forward_calls": statistics.mean(calls) if calls else "",
        "total_seconds": sum(secs) if secs else "",
    })

with (tables / "sampling_related_work_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "tasks", "macro_accuracy", "avg_forward_calls", "total_seconds"])
    w.writeheader()
    w.writerows(summary_rows)

print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
PY

echo "[$(date)] by-task sampling related-work run done: $ROOT"
