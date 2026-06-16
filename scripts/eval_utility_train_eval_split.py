#!/usr/bin/env python3
"""Fit the per-sample counterfactual-utility policy on a TRAIN action table and
evaluate it on a SEPARATE eval action table (no in-eval-set leakage).

Reuses flatten/build_schema/vectorize/fit_model/group_rows/summarize from
train_llada_counterfactual_utility, but fits on TRAIN rows and scores EVAL rows.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_llada_counterfactual_utility as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-table", required=True)
    ap.add_argument("--eval-table", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-task", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--decision-checkpoint", type=int, default=2)
    ap.add_argument("--lambda-call", type=float, default=1 / 32)
    ap.add_argument("--lambda-remask", type=float, default=0.0)
    args = ap.parse_args()

    tr_payload = json.loads(Path(args.train_table).read_text())
    ev_payload = json.loads(Path(args.eval_table).read_text())
    tr_rows, tr_base = U.flatten(tr_payload, args.lambda_call, args.lambda_remask)
    ev_rows, _ = U.flatten(ev_payload, args.lambda_call, args.lambda_remask)
    tr_rows = [r for r in tr_rows if int(r["checkpoint"]) == args.decision_checkpoint]
    ev_rows = [r for r in ev_rows if int(r["checkpoint"]) == args.decision_checkpoint]

    names = U.build_schema(tr_base, tr_rows)
    X = np.vstack([U.vectorize(r, names) for r in tr_rows])
    y = np.array([r["reward"] for r in tr_rows], dtype=np.float64)
    model, model_name = U.fit_model(X, y, args.seed)

    test_groups = U.group_rows(ev_rows)
    learned = []
    for group in test_groups.values():
        Xg = np.vstack([U.vectorize(r, names) for r in group])
        pred = model.predict(Xg)
        learned.append(group[int(np.argmax(pred))])
    oracle = [max(group, key=lambda r: (r["reward"], r["correct"], -r["total_calls"]))
              for group in test_groups.values()]

    policies = [U.summarize(learned, "learned_utility_TRAINfit"), U.summarize(oracle, "oracle")]
    for action in ["accept_current", "restart_to_8", "restart_to_16"]:
        c = U.choose_by_action(test_groups, action)
        if c:
            policies.append(U.summarize(c, action))
    for prefix in ["remask_label_only_to_", "remask_lowconf_to_"]:
        c = U.choose_by_prefix(test_groups, prefix)
        if c:
            policies.append(U.summarize(c, prefix + "max"))

    summary_rows = [{
        "policy": p["policy"], "n": p["n"], "macro_accuracy": p["macro_accuracy"],
        "avg_calls": p["avg_calls"], "avg_remasked": p["avg_remasked"], "avg_reward": p["avg_reward"],
    } for p in policies]
    task_rows = []
    for p in policies:
        task_rows.extend(p["task_rows"])
    U.write_csv(args.out_summary, summary_rows)
    U.write_csv(args.out_task, task_rows)
    print(json.dumps({"model": model_name, "n_train_states": len(set((r["task"], r["id"]) for r in tr_rows)),
                      "n_eval_states": len(test_groups), "summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
