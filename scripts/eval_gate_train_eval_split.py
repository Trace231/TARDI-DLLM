#!/usr/bin/env python3
"""Calibrate the selective-remask gate on a TRAIN action table and evaluate it on
a SEPARATE eval action table (held-out by construction, no in-eval-set split).

Both tables must come from the same collector config (same checkpoints / targets /
remask families), so the gate fitted on TRAIN transfers to the eval rows. Reuses
the tune/evaluate/summarize functions from eval_llada_counterfactual_gate_policy.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_llada_counterfactual_gate_policy as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-table", required=True)
    ap.add_argument("--eval-table", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-task", required=True)
    ap.add_argument("--checkpoint", type=int, default=2)
    ap.add_argument("--lambda-call", type=float, default=1 / 32)
    ap.add_argument("--objective", choices=["reward", "accuracy"], default="reward")
    ap.add_argument("--actions", default="")
    ap.add_argument("--scores", default="risk,probe_entropy,margin_deficit")
    args = ap.parse_args()

    train_payload = json.loads(Path(args.train_table).read_text())
    eval_payload = json.loads(Path(args.eval_table).read_text())
    cal = G.rows_from_payload(train_payload, args.checkpoint)
    test = G.rows_from_payload(eval_payload, args.checkpoint)

    if args.actions:
        actions = [x.strip() for x in args.actions.split(",") if x.strip()]
    else:
        actions = sorted({a for row in cal for a in row["actions"] if a != "accept_current"})
    scores = [x.strip() for x in args.scores.split(",") if x.strip()]

    best = G.tune(cal, actions, scores, args.objective, args.lambda_call)
    test_stats = G.evaluate(test, best["action"], best["score"], best["threshold"], args.lambda_call)
    accept_stats = G.evaluate(test, "accept_current", "risk", 2.0, args.lambda_call)

    fixed_rows = []
    for action in actions:
        fixed_rows.append({
            "policy": f"fixed_{action}", "action": action, "score": "-", "threshold": "-",
            **G.evaluate(test, action, "risk", -1.0, args.lambda_call),
        })

    summary = [
        {"policy": "calibrated_gate_TRAINfit", "action": best["action"], "score": best["score"],
         "threshold": best["threshold"], **test_stats},
        {"policy": "accept_current", "action": "accept_current", "score": "-", "threshold": "-",
         **accept_stats},
        *fixed_rows,
    ]
    G.write_csv(args.out_summary, summary)
    task_rows = G.summarize_by_task(test, best["action"], best["score"], best["threshold"],
                                    args.lambda_call, "calibrated_gate_TRAINfit")
    G.write_csv(args.out_task, task_rows)
    print(json.dumps({
        "best": best,
        "n_cal_states": len(cal),
        "n_test_states": len(test),
        "test": test_stats,
        "accept": accept_stats,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
