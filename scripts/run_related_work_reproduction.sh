#!/usr/bin/env bash
set -euo pipefail

# Reproduce related-work style baselines for the TARDI-DLLM report.
#
# Usage on the remote server:
#   cd /data/llada_eval
#   source scripts/env_llada.sh
#   bash scripts/run_related_work_reproduction.sh
#
# Knobs:
#   RUN_LORA=1/0       train/evaluate LoRA-side baselines
#   RUN_ACCEL=1/0      evaluate sampling/inference baselines
#   LIMIT=50           examples per task
#   TASKS=...          comma-separated task list

cd "${LLADA_EVAL_ROOT:-/data/llada_eval}"
if [ -f scripts/env_llada.sh ]; then
  # shellcheck disable=SC1091
  source scripts/env_llada.sh
fi

MODEL="${MODEL:-/data/hf/models/GSAI-ML/LLaDA-8B-Instruct}"
ROOT="${ROOT:-results/domain_shift/task_aware/solid_v2/related_work_v1}"
TASKS="${TASKS:-mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq,gsm8k}"
LORA_TASKS="${LORA_TASKS:-mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq}"
LIMIT="${LIMIT:-50}"
SEED="${SEED:-23}"
RUN_LORA="${RUN_LORA:-1}"
RUN_ACCEL="${RUN_ACCEL:-1}"

mkdir -p "$ROOT"/{raw,logs,tables,reports,adapters}

run_eval_variant() {
  local name="$1"
  local steps="$2"
  local schedule="$3"
  local sampler="${4:-standard}"
  local extra="${5:-}"
  local out="$ROOT/raw/${name}_limit${LIMIT}_seed${SEED}.json"
  if [ -f "$out" ]; then
    echo "[$(date)] skip $name; exists: $out"
    return
  fi
  echo "[$(date)] run sampler $name"
  # shellcheck disable=SC2086
  python3 scripts/eval_llada_sampler_variants.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
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
    2>&1 | tee "$ROOT/logs/${name}_limit${LIMIT}_seed${SEED}.log"
}

run_prophet() {
  local out="$ROOT/raw/prophet_early_commit_limit${LIMIT}_seed${SEED}.json"
  if [ -f "$out" ]; then
    echo "[$(date)] skip prophet; exists: $out"
    return
  fi
  echo "[$(date)] run Prophet-style early commit"
  python3 scripts/eval_llada_prophet_early_commit.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
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
    2>&1 | tee "$ROOT/logs/prophet_early_commit_limit${LIMIT}_seed${SEED}.log"
}

run_refinement() {
  local name="$1"
  local extra="$2"
  local out="$ROOT/raw/${name}_limit${LIMIT}_seed${SEED}.json"
  if [ -f "$out" ]; then
    echo "[$(date)] skip $name; exists: $out"
    return
  fi
  echo "[$(date)] run refinement controller $name"
  # shellcheck disable=SC2086
  python3 scripts/eval_llada_refinement_controller.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --budgets 8,16,24,32 \
    --risk-t16 0.24 \
    --risk-t24 0.38 \
    --risk-t32 0.56 \
    --multi-disagreement-policy ignore \
    --out "$out" \
    $extra \
    2>&1 | tee "$ROOT/logs/${name}_limit${LIMIT}_seed${SEED}.log"
}

if [ "$RUN_ACCEL" = "1" ]; then
  # Fixed-step and schedule baselines.
  run_eval_variant "fixed32_uniform" 32 uniform
  run_eval_variant "jys_like_middle16" 16 middle_heavy
  run_eval_variant "fixed16_uniform" 16 uniform
  run_eval_variant "front_loaded16" 16 front_loaded
  run_eval_variant "back_loaded16" 16 back_loaded

  # Predictor-corrector approximates solver-style correction with an extra model
  # call per step. It is not a DPM-Solver reproduction, but gives a stronger
  # local-correction baseline under the same evaluator.
  run_eval_variant "pc_middle16" 16 middle_heavy predictor_corrector "--corrector-weight 0.5"

  # Early-answer-convergence baseline.
  run_prophet

  # Remasking/backtracking family. These are ReMDM/Saber-like baselines under
  # the current LLaDA evaluator, not full reproductions of those papers.
  run_refinement "ours_refinement_lowconf" ""
  run_refinement "remdm_like_structured_remask" "--structured-remask"
  run_refinement "saber_like_answer_consistency" "--answer-consistency-remask"
fi

if [ "$RUN_LORA" = "1" ]; then
  echo "[$(date)] run LoRA related baselines"
  RUN_OFFICIAL_SCALE_NARA="${RUN_OFFICIAL_SCALE_NARA:-0}" \
  RUN_NARA_OFFICIAL_TARGETS="${RUN_NARA_OFFICIAL_TARGETS:-0}" \
  MODEL="$MODEL" \
  ROOT="$ROOT/lora_external" \
  TASKS="$LORA_TASKS" \
  LIMIT="$LIMIT" \
  SEED="$SEED" \
  bash scripts/run_external_lora_baselines.sh \
    2>&1 | tee "$ROOT/logs/run_external_lora_limit${LIMIT}_seed${SEED}.log"
fi

echo "[$(date)] analyze related-work reproduction"
python3 - <<'PY'
import csv, json, statistics
from pathlib import Path

root = Path("results/domain_shift/task_aware/solid_v2/related_work_v1")
raw = root / "raw"
tables = root / "tables"
tables.mkdir(parents=True, exist_ok=True)

rows = []
for p in sorted(raw.glob("*.json")):
    data = json.loads(p.read_text())
    summary = data.get("summary", {})
    for task, s in summary.items():
        rows.append({
            "method": p.stem,
            "task": task,
            "n": s.get("n", ""),
            "accuracy": s.get("accuracy", ""),
            "avg_forward_calls": s.get("avg_forward_calls", ""),
            "seconds": s.get("seconds", ""),
        })

with (tables / "sampling_related_work_by_task.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "task", "n", "accuracy", "avg_forward_calls", "seconds"])
    w.writeheader()
    w.writerows(rows)

by_method = {}
for r in rows:
    by_method.setdefault(r["method"], []).append(r)
summary_rows = []
for method, xs in sorted(by_method.items()):
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

echo "[$(date)] related-work reproduction done: $ROOT"
