#!/usr/bin/env python3
"""Calibrate and evaluate anytime stopping policies from LLaDA traces."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_name(row, seed, frac):
    return "calibration" if stable_value(seed, row["task"], row["id"]) < frac else "test"


def state_at(row, step):
    states = row.get("states") or []
    exact = [s for s in states if int(s["step"]) == int(step)]
    if exact:
        return exact[0]
    before = [s for s in states if int(s["step"]) <= int(step)]
    return before[-1] if before else states[0]


def choose_fixed(row, step):
    s = state_at(row, step)
    return {
        "task": row["task"],
        "id": row["id"],
        "policy_step": int(s["step"]),
        "pred": s["pred"],
        "correct": bool(s["correct"]),
        "calls": float(s.get("forward_calls", s["step"])),
        "stop_reason": f"fixed_{step}",
    }


def choose_oracle(row, lambda_call, max_calls):
    best = None
    for s in row.get("states", []):
        reward = float(s["correct"]) - lambda_call * float(s.get("forward_calls", s["step"])) / max_calls
        key = (reward, float(s["correct"]), -float(s.get("forward_calls", s["step"])))
        if best is None or key > best[0]:
            best = (key, s)
    s = best[1]
    return {
        "task": row["task"],
        "id": row["id"],
        "policy_step": int(s["step"]),
        "pred": s["pred"],
        "correct": bool(s["correct"]),
        "calls": float(s.get("forward_calls", s["step"])),
        "stop_reason": "oracle",
    }


def stop_allowed(state, rule):
    if int(state["step"]) < rule["min_step"]:
        return False
    if rule["require_valid"] and not state.get("valid"):
        return False
    if float(state.get("filled_ratio", 0.0)) < rule["min_filled"]:
        return False
    if int(state.get("stable_count", 0)) < rule["stable_count"]:
        return False
    if float(state.get("mean_fill_confidence", 0.0)) < rule["min_mean_conf"]:
        return False
    if float(state.get("min_fill_confidence", 0.0)) < rule["min_token_conf"]:
        return False
    if rule["max_flips"] is not None and int(state.get("flip_count", 0)) > rule["max_flips"]:
        return False
    return True


def choose_anytime(row, rule):
    states = row.get("states") or []
    for state in states:
        if stop_allowed(state, rule):
            return {
                "task": row["task"],
                "id": row["id"],
                "policy_step": int(state["step"]),
                "pred": state["pred"],
                "correct": bool(state["correct"]),
                "calls": float(state.get("forward_calls", state["step"])),
                "stop_reason": "anytime_stop",
            }
    state = states[-1]
    return {
        "task": row["task"],
        "id": row["id"],
        "policy_step": int(state["step"]),
        "pred": state["pred"],
        "correct": bool(state["correct"]),
        "calls": float(state.get("forward_calls", state["step"])),
        "stop_reason": "max_budget",
    }


def summarize(chosen, policy):
    by_task = defaultdict(list)
    for row in chosen:
        by_task[row["task"]].append(row)
    task_rows = []
    for task, rows in sorted(by_task.items()):
        n = len(rows)
        task_rows.append(
            {
                "policy": policy,
                "task": task,
                "n": n,
                "accuracy": sum(r["correct"] for r in rows) / n,
                "avg_calls": sum(r["calls"] for r in rows) / n,
                "avg_stop_step": sum(r["policy_step"] for r in rows) / n,
            }
        )
    return {
        "policy": policy,
        "n": len(chosen),
        "macro_accuracy": sum(r["accuracy"] for r in task_rows) / len(task_rows),
        "micro_accuracy": sum(r["correct"] for r in chosen) / len(chosen),
        "avg_calls": sum(r["calls"] for r in chosen) / len(chosen),
        "avg_stop_step": sum(r["policy_step"] for r in chosen) / len(chosen),
        "task_rows": task_rows,
    }


def reward(summary, lambda_call, max_calls, min_acc_floor=None):
    if min_acc_floor is not None and summary["macro_accuracy"] < min_acc_floor:
        return -1e9 + summary["macro_accuracy"]
    return summary["macro_accuracy"] - lambda_call * summary["avg_calls"] / max_calls


def grid_rules(args):
    for min_step in [int(x) for x in args.min_steps.split(",") if x.strip()]:
        for stable_count in [int(x) for x in args.stable_counts.split(",") if x.strip()]:
            for min_filled in [float(x) for x in args.min_filled_values.split(",") if x.strip()]:
                for min_mean_conf in [float(x) for x in args.mean_conf_values.split(",") if x.strip()]:
                    for min_token_conf in [float(x) for x in args.token_conf_values.split(",") if x.strip()]:
                        for max_flips_raw in [x.strip() for x in args.max_flips_values.split(",") if x.strip()]:
                            max_flips = None if max_flips_raw == "none" else int(max_flips_raw)
                            yield {
                                "min_step": min_step,
                                "stable_count": stable_count,
                                "min_filled": min_filled,
                                "min_mean_conf": min_mean_conf,
                                "min_token_conf": min_token_conf,
                                "max_flips": max_flips,
                                "require_valid": args.require_valid,
                            }


def tune(cal_rows, args, max_calls):
    fixed32 = summarize([choose_fixed(r, args.max_step) for r in cal_rows], f"fixed_{args.max_step}")
    floor = None
    if args.accuracy_floor_from_full:
        floor = fixed32["macro_accuracy"] - args.floor_tolerance
    best = None
    for rule in grid_rules(args):
        chosen = [choose_anytime(r, rule) for r in cal_rows]
        stats = summarize(chosen, "anytime")
        val = reward(stats, args.lambda_call, max_calls, floor)
        if best is None or val > best["value"]:
            best = {"rule": rule, "summary": stats, "value": val, "floor": floor}
    return best


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
    ap.add_argument("--trace-json", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-task", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-policy", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--calibration-fraction", type=float, default=0.5)
    ap.add_argument("--fixed-steps", default="2,4,6,8,12,16,24,32")
    ap.add_argument("--max-step", type=int, default=32)
    ap.add_argument("--lambda-call", type=float, default=0.03)
    ap.add_argument("--accuracy-floor-from-full", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--floor-tolerance", type=float, default=0.01)
    ap.add_argument("--require-valid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-steps", default="2,4,6,8")
    ap.add_argument("--stable-counts", default="1,2,3")
    ap.add_argument("--min-filled-values", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--mean-conf-values", default="0.0,0.2,0.4,0.6,0.8")
    ap.add_argument("--token-conf-values", default="0.0,0.1,0.2,0.3")
    ap.add_argument("--max-flips-values", default="none,0,1,2")
    args = ap.parse_args()

    payload = json.loads(Path(args.trace_json).read_text())
    rows = payload["rows"]
    cal = [r for r in rows if split_name(r, args.seed, args.calibration_fraction) == "calibration"]
    test = [r for r in rows if split_name(r, args.seed, args.calibration_fraction) == "test"]
    max_calls = float(args.max_step)
    best = tune(cal, args, max_calls)
    anytime = [choose_anytime(r, best["rule"]) for r in test]
    policies = [summarize(anytime, "anytime_calibrated")]
    for step in [int(x) for x in args.fixed_steps.split(",") if x.strip()]:
        policies.append(summarize([choose_fixed(r, step) for r in test], f"fixed_{step}"))
    policies.append(summarize([choose_oracle(r, args.lambda_call, max_calls) for r in test], "oracle_per_sample"))

    summary_rows = [
        {
            "policy": p["policy"],
            "n": p["n"],
            "macro_accuracy": p["macro_accuracy"],
            "micro_accuracy": p["micro_accuracy"],
            "avg_calls": p["avg_calls"],
            "avg_stop_step": p["avg_stop_step"],
        }
        for p in policies
    ]
    task_rows = []
    for p in policies:
        task_rows.extend(p["task_rows"])
    chosen_rows = [
        {
            "task": r["task"],
            "id": r["id"],
            "policy_step": r["policy_step"],
            "pred": r["pred"],
            "correct": r["correct"],
            "calls": r["calls"],
            "stop_reason": r["stop_reason"],
        }
        for r in anytime
    ]
    write_csv(args.out_summary, summary_rows)
    write_csv(args.out_task, task_rows)
    write_csv(args.out_rows, chosen_rows)
    Path(args.out_policy).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_policy).write_text(
        json.dumps(
            {
                "args": vars(args),
                "trace_json": args.trace_json,
                "n_calibration": len(cal),
                "n_test": len(test),
                "selected_rule": best["rule"],
                "calibration_summary": best["summary"],
                "calibration_value": best["value"],
                "accuracy_floor": best["floor"],
                "test_summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"selected_rule": best["rule"], "summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
