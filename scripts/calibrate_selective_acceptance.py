#!/usr/bin/env python3
"""Split-calibrate direct-accept risk from online decision tables."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_name(sample, seed, frac):
    return "calibration" if stable_value(seed, sample["task"], sample["id"]) < frac else "test"


def state_risk(state):
    f = state["router_features"]
    bayes = float(f.get("probe_bayes_risk", 1.0))
    ent = float(f.get("probe_entropy", 1.0))
    margin = float(f.get("probe_margin", 0.0))
    disagree = float(f.get("probe_scout_disagree", 0.0))
    invalid = 0.0 if state.get("llda_valid") else 1.0
    return 0.45 * bayes + 0.25 * ent + 0.20 * max(0.0, 1.0 - margin) + 0.05 * disagree + 0.05 * invalid


def first_valid_state(sample, max_checkpoint):
    for state in sample["states"]:
        if state["checkpoint"] <= max_checkpoint and state["llda_valid"]:
            return state
    return None


def choose_threshold(calibration, epsilon, max_checkpoint):
    candidates = []
    for sample in calibration:
        state = first_valid_state(sample, max_checkpoint)
        if state is None:
            continue
        candidates.append((state_risk(state), int(not state["llda_correct"])))
    candidates.sort(key=lambda x: x[0])
    best = None
    errors = 0
    for i, (risk, err) in enumerate(candidates, start=1):
        errors += err
        if errors / i <= epsilon:
            best = risk
    return best if best is not None else -1.0


def evaluate(samples, threshold, max_checkpoint):
    rows = []
    for sample in samples:
        state = first_valid_state(sample, max_checkpoint)
        if state is None:
            accept = False
            risk = 1.0
            correct = False
        else:
            risk = state_risk(state)
            accept = risk <= threshold
            correct = bool(state["llda_correct"])
        rows.append({"task": sample["task"], "id": sample["id"], "accept": accept, "risk_score": risk, "correct": correct})
    accepted = [r for r in rows if r["accept"]]
    return {
        "n": len(rows),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "accepted_error_rate": sum(not r["correct"] for r in accepted) / len(accepted) if accepted else 0.0,
        "accepted_accuracy": sum(r["correct"] for r in accepted) / len(accepted) if accepted else 0.0,
        "rows": rows,
    }


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-table", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--calibration-fraction", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=0.12)
    ap.add_argument("--max-checkpoint", type=int, default=8)
    args = ap.parse_args()

    payload = json.loads(Path(args.decision_table).read_text())
    samples = payload["samples"]
    calibration = [s for s in samples if split_name(s, args.seed, args.calibration_fraction) == "calibration"]
    test = [s for s in samples if split_name(s, args.seed, args.calibration_fraction) == "test"]
    threshold = choose_threshold(calibration, args.epsilon, args.max_checkpoint)
    cal = evaluate(calibration, threshold, args.max_checkpoint)
    tst = evaluate(test, threshold, args.max_checkpoint)
    summary = [
        {
            "split": "calibration",
            "epsilon": args.epsilon,
            "threshold": threshold,
            "n": cal["n"],
            "coverage": cal["coverage"],
            "accepted_error_rate": cal["accepted_error_rate"],
            "accepted_accuracy": cal["accepted_accuracy"],
        },
        {
            "split": "test",
            "epsilon": args.epsilon,
            "threshold": threshold,
            "n": tst["n"],
            "coverage": tst["coverage"],
            "accepted_error_rate": tst["accepted_error_rate"],
            "accepted_accuracy": tst["accepted_accuracy"],
        },
    ]
    write_csv(args.out_summary, summary)
    write_csv(args.out_rows, tst["rows"])
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
