#!/usr/bin/env python3
"""Collect a counterfactual action table on a TRAIN pool that is disjoint (by id)
from the canonical eval-450 set, mirroring build_balanced_lora_train's selection.

We reuse collect_llada_counterfactual_actions.main() verbatim, but monkeypatch
eval_domain_shift.build_samples so it yields the disjoint train pool instead of
the eval samples. This keeps the action/feature schema identical to the eval
table, so a gate calibrated on TRAIN can be applied to the eval-450 table.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as ed

_orig_build_samples = ed.build_samples

TASKS = [
    "mmlu_pro", "pubmedqa", "ceval_computer_network", "sciq", "winogrande",
    "commonsenseqa", "arc_challenge", "hellaswag", "boolq",
]
EVAL_SEED = int(os.environ.get("EVAL_SEED", "23"))
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "50"))
TRAIN_LIMIT = int(os.environ.get("TRAIN_LIMIT", "120"))
POOL = int(os.environ.get("POOL", "400"))
ROUNDS = int(os.environ.get("ROUNDS", "12"))
MODEL = os.environ.get("MODEL", "/data/hf/models/GSAI-ML/LLaDA-8B-Instruct")
OUT = os.environ["OUT"]

# Build a per-task train pool disjoint from the eval-450 ids.
pools = {}
for task in TASKS:
    eval_ids = {str(s["id"]) for s in _orig_build_samples(task, EVAL_LIMIT, EVAL_SEED)}
    rows = {}
    for r in range(ROUNDS):
        seed = EVAL_SEED + 1 + r * 9973
        for sample in _orig_build_samples(task, POOL, seed):
            sid = str(sample["id"])
            if sid in eval_ids or sid in rows:
                continue
            rows[sid] = sample
            if len(rows) >= TRAIN_LIMIT:
                break
        if len(rows) >= TRAIN_LIMIT:
            break
    pools[task] = list(rows.values())
    print(f"[trainpool] {task}: train={len(pools[task])} (excluded {len(eval_ids)} eval ids)", flush=True)

print(f"[trainpool] total train samples: {sum(len(v) for v in pools.values())}", flush=True)


def _patched_build_samples(task, limit, seed):
    return pools[task][:limit]


ed.build_samples = _patched_build_samples

# Reuse the collector's main() with the patched sampler.
argv = [
    "collect_llada_counterfactual_actions.py",
    "--model", MODEL,
    "--tasks", ",".join(TASKS),
    "--limit", str(TRAIN_LIMIT),
    "--seed", str(EVAL_SEED),
    "--out", OUT,
    "--checkpoints", os.environ.get("CHECKPOINTS", "2"),
    "--targets", os.environ.get("TARGETS", "4,8,16"),
    "--remask-families", os.environ.get("REMASK_FAMILIES", "label_only,lowconf"),
]
if os.environ.get("NO_RESTART", "0") == "1":
    argv.append("--no-include-restart")
sys.argv = argv
import collect_llada_counterfactual_actions as cf  # noqa: E402

cf.main()
