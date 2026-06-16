#!/usr/bin/env python3
"""Evaluate calibrated gate policies on LLaDA counterfactual action tables."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_name(sample, seed, frac):
    return "calibration" if stable_value(seed, sample["task"], sample["id"]) < frac else "test"


def rows_from_payload(payload, checkpoint):
    rows = []
    for sample in payload["samples"]:
        for state in sample["states"]:
            if int(state["checkpoint"]) != checkpoint:
                continue
            acts = {a["action"]: a for a in state["actions"]}
            feats = state.get("features") or {}
            rows.append(
                {
                    "task": sample["task"],
                    "id": sample["id"],
                    "features": feats,
                    "risk": float(feats.get("risk_score", 0.0) or 0.0),
                    "probe_entropy": float(feats.get("probe_entropy", 0.0) or 0.0),
                    "margin_deficit": float(feats.get("margin_deficit", 0.0) or 0.0),
                    "actions": acts,
                }
            )
    return rows


def choose(row, action_name, score_name, threshold):
    if row[score_name] >= threshold and action_name in row["actions"]:
        return row["actions"][action_name]
    return row["actions"]["accept_current"]


def evaluate(rows, action_name, score_name, threshold, lambda_call):
    chosen = [choose(row, action_name, score_name, threshold) for row in rows]
    n = len(chosen)
    return {
        "n": n,
        "accuracy": sum(a["correct"] for a in chosen) / n,
        "avg_calls": sum(float(a["total_calls"]) for a in chosen) / n,
        "avg_remasked": sum(float(a["remasked_tokens"]) for a in chosen) / n,
        "avg_reward": sum(float(a["correct"]) - lambda_call * float(a["total_calls"]) for a in chosen) / n,
    }


def tune(cal_rows, actions, scores, objective, lambda_call):
    best = None
    for action in actions:
        for score in scores:
            vals = sorted({row[score] for row in cal_rows})
            for threshold in [-1.0, *vals, 2.0]:
                stats = evaluate(cal_rows, action, score, threshold, lambda_call)
                value = stats["accuracy"] if objective == "accuracy" else stats["avg_reward"]
                if best is None or value > best["objective_value"]:
                    best = {
                        "action": action,
                        "score": score,
                        "threshold": threshold,
                        "objective_value": value,
                        "calibration": stats,
                    }
    return best


def summarize_by_task(rows, action_name, score_name, threshold, lambda_call, policy):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    out = []
    for task, group in sorted(by_task.items()):
        stats = evaluate(group, action_name, score_name, threshold, lambda_call)
        out.append({"policy": policy, "task": task, **stats})
    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-table", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-task", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--calibration-fraction", type=float, default=0.5)
    ap.add_argument("--checkpoint", type=int, default=4)
    ap.add_argument("--lambda-call", type=float, default=1 / 32)
    ap.add_argument("--objective", choices=["reward", "accuracy"], default="reward")
    ap.add_argument("--actions", default="")
    ap.add_argument("--scores", default="risk,probe_entropy,margin_deficit")
    args = ap.parse_args()

    payload = json.loads(Path(args.action_table).read_text())
    rows = rows_from_payload(payload, args.checkpoint)
    cal = [r for r in rows if split_name(r, args.seed, args.calibration_fraction) == "calibration"]
    test = [r for r in rows if split_name(r, args.seed, args.calibration_fraction) == "test"]
    if args.actions:
        actions = [x.strip() for x in args.actions.split(",") if x.strip()]
    else:
        actions = sorted({a for row in rows for a in row["actions"] if a != "accept_current"})
    scores = [x.strip() for x in args.scores.split(",") if x.strip()]
    best = tune(cal, actions, scores, args.objective, args.lambda_call)
    test_stats = evaluate(test, best["action"], best["score"], best["threshold"], args.lambda_call)
    accept_stats = evaluate(test, "accept_current", "risk", 2.0, args.lambda_call)
    fixed_rows = []
    for action in actions:
        fixed_rows.append({"policy": f"fixed_{action}", "action": action, "score": "-", "threshold": "-", **evaluate(test, action, "risk", -1.0, args.lambda_call)})
    summary = [
        {
            "policy": "calibrated_gate",
            "action": best["action"],
            "score": best["score"],
            "threshold": best["threshold"],
            **test_stats,
        },
        {
            "policy": "accept_current",
            "action": "accept_current",
            "score": "-",
            "threshold": "-",
            **accept_stats,
        },
        *fixed_rows,
    ]
    write_csv(args.out_summary, summary)
    task_rows = summarize_by_task(test, best["action"], best["score"], best["threshold"], args.lambda_call, "calibrated_gate")
    write_csv(args.out_task, task_rows)
    print(json.dumps({"best": best, "test": test_stats, "accept": accept_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
