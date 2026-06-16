#!/usr/bin/env python3
"""Stream-evaluate an online cost-sensitive LinUCB router."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


ACTIONS = ["D", "R", "A"]


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_name(sample, seed, calibration_fraction):
    return "calibration" if stable_value(seed, sample["task"], sample["id"]) < calibration_fraction else "test"


def feature_schema(samples):
    names = []
    seen = set()
    for sample in samples:
        for state in sample["states"]:
            for key in state["router_features"]:
                if key not in seen:
                    names.append(key)
                    seen.add(key)
    return names


def vectorize(features, names):
    x = np.array([float(features.get(name, 0.0)) for name in names], dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm > 8.0:
        x = x * (8.0 / norm)
    return x


def state_risk(state):
    """Online-visible risk score for conformal direct-accept gating."""
    f = state["router_features"]
    bayes = float(f.get("probe_bayes_risk", 1.0))
    entropy = float(f.get("probe_entropy", 1.0))
    margin = float(f.get("probe_margin", 0.0))
    disagree = float(f.get("probe_scout_disagree", 0.0))
    invalid = 0.0 if state.get("llda_valid") else 1.0
    late = float(f.get("checkpoint_norm", 0.0))
    return (
        0.40 * bayes
        + 0.25 * entropy
        + 0.18 * max(0.0, 1.0 - margin)
        + 0.07 * disagree
        + 0.06 * invalid
        + 0.04 * late
    )


def calibrate_direct_threshold(samples, epsilon, max_checkpoint):
    candidates = []
    for sample in samples:
        for state in sample["states"]:
            if state["checkpoint"] > max_checkpoint or not state["llda_valid"]:
                continue
            candidates.append((state_risk(state), int(not state["llda_correct"])))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    threshold = None
    errors = 0
    for idx, (risk, err) in enumerate(candidates, start=1):
        errors += err
        if errors / idx <= epsilon:
            threshold = risk
    return threshold


class LinUCBRouter:
    def __init__(self, dim, alpha=0.75, ridge=1.0):
        self.dim = dim
        self.alpha = alpha
        self.A = {a: np.eye(dim, dtype=np.float64) * ridge for a in ACTIONS}
        self.b = {a: np.zeros(dim, dtype=np.float64) for a in ACTIONS}

    def score(self, action, x):
        inv = np.linalg.inv(self.A[action])
        theta = inv @ self.b[action]
        mean = float(theta @ x)
        bonus = self.alpha * math.sqrt(max(0.0, float(x @ inv @ x)))
        return mean + bonus

    def choose(self, x, valid_actions, rng, epsilon):
        valid = list(valid_actions)
        if rng.random() < epsilon:
            return rng.choice(valid), "epsilon"
        return max(valid, key=lambda a: self.score(a, x)), "ucb"

    def update(self, action, x, reward):
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def state_dict(self):
        return {
            "alpha": self.alpha,
            "A": {k: v.tolist() for k, v in self.A.items()},
            "b": {k: v.tolist() for k, v in self.b.items()},
        }


def valid_actions(state, sample, args, ar_used, direct_threshold=None):
    actions = []
    direct_allowed = state["llda_valid"]
    if direct_threshold is not None and state["checkpoint"] <= args.selective_max_checkpoint:
        direct_allowed = direct_allowed and state_risk(state) <= direct_threshold
    if direct_allowed:
        actions.append("D")
    if state["has_next_state"] and state["checkpoint"] < args.max_checkpoint:
        actions.append("R")
    if state["checkpoint"] >= args.min_ar_checkpoint and not ar_used:
        actions.append("A")
    if not actions:
        actions.append("A" if not ar_used else "D")
    return actions


def terminal_reward(correct, spent_llda_calls, ar_called, args):
    return (
        float(correct)
        - args.lambda_llda_call * float(spent_llda_calls)
        - (args.lambda_ar_call if ar_called else 0.0)
    )


def run_episode(sample, router, feature_names, rng, args, learn=True, direct_threshold=None):
    pending = []
    ar_used = False
    states = sample["states"]
    idx = 0
    chosen_action = None
    decision_mode = None
    while True:
        state = states[idx]
        x = vectorize(state["router_features"], feature_names)
        actions = valid_actions(state, sample, args, ar_used, direct_threshold)
        action, mode = router.choose(x, actions, rng, args.epsilon)
        pending.append((action, x))
        chosen_action = action
        decision_mode = mode
        if action == "R" and state["has_next_state"]:
            idx += 1
            continue
        if action == "A":
            ar_used = True
            pred = sample["ar_pred"]
            correct = bool(sample["ar_correct"])
            terminal = "A"
        else:
            pred = state["llda_pred"]
            correct = bool(state["llda_correct"])
            terminal = "D"
        spent = state["spent_llda_calls"]
        reward = terminal_reward(correct, spent, ar_used, args)
        if learn:
            for a, feat in pending:
                router.update(a, feat, reward)
        return {
            "task": sample["task"],
            "id": sample["id"],
            "gold": sample["gold"],
            "pred": pred,
            "correct": correct,
            "reward": reward,
            "terminal_action": terminal,
            "last_chosen_action": chosen_action,
            "decision_mode": decision_mode,
            "ar_called": ar_used,
            "spent_llda_calls": spent,
            "checkpoint": state["checkpoint"],
            "direct_risk": state_risk(state),
            "updates": len(pending) if learn else 0,
        }


def baseline_episode(sample, policy, rng=None, random_rate=0.0):
    first = sample["states"][0]
    last = sample["states"][-1]
    if policy == "llda_first_valid":
        state = next((s for s in sample["states"] if s["llda_valid"]), last)
        ar = False
    elif policy == "llda_last_observed":
        state = last
        ar = False
    elif policy == "ar_only":
        state = first
        ar = True
    elif policy == "all_multichoice_ar":
        state = first
        ar = len(sample.get("label_space") or []) >= 3
    elif policy == "binary_llda32_multichoice_ar":
        state = last
        ar = len(sample.get("label_space") or []) >= 3
    elif policy == "random_same_budget":
        state = first
        ar = len(sample.get("label_space") or []) >= 3 and rng.random() < random_rate
    else:
        raise KeyError(policy)
    correct = bool(sample["ar_correct"]) if ar else bool(state["llda_correct"])
    pred = sample["ar_pred"] if ar else state["llda_pred"]
    return {
        "task": sample["task"],
        "id": sample["id"],
        "gold": sample["gold"],
        "pred": pred,
        "correct": correct,
        "reward": float(correct),
        "terminal_action": "A" if ar else "D",
        "last_chosen_action": "A" if ar else "D",
        "decision_mode": policy,
        "ar_called": ar,
        "spent_llda_calls": state["spent_llda_calls"],
        "checkpoint": state["checkpoint"],
        "updates": 0,
    }


def summarize(rows, policy, args=None):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    task_rows = []
    for task in sorted(by_task):
        group = by_task[task]
        n = len(group)
        task_rows.append(
            {
                "policy": policy,
                "task": task,
                "n": n,
                "accuracy": sum(r["correct"] for r in group) / n,
                "avg_reward": sum(r["reward"] for r in group) / n,
                "ar_trigger_rate": sum(r["ar_called"] for r in group) / n,
                "avg_llda_calls": sum(r["spent_llda_calls"] for r in group) / n,
                "estimated_seconds_per_sample": (
                    (sum(r["spent_llda_calls"] for r in group) / n) * args.seconds_per_llda_call
                    + (sum(r["ar_called"] for r in group) / n) * args.seconds_per_ar_call
                    if args
                    else 0.0
                ),
                "accept_risk": 1.0
                - (
                    sum(r["correct"] for r in group if not r["ar_called"]) / max(1, sum(1 for r in group if not r["ar_called"]))
                ),
            }
        )
    return {
        "policy": policy,
        "n": len(rows),
        "macro_accuracy": sum(r["accuracy"] for r in task_rows) / len(task_rows),
        "micro_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "avg_reward": sum(r["reward"] for r in rows) / len(rows),
        "ar_trigger_rate": sum(r["ar_called"] for r in rows) / len(rows),
        "avg_llda_calls": sum(r["spent_llda_calls"] for r in rows) / len(rows),
        "estimated_seconds_per_sample": (
            (sum(r["spent_llda_calls"] for r in rows) / len(rows)) * args.seconds_per_llda_call
            + (sum(r["ar_called"] for r in rows) / len(rows)) * args.seconds_per_ar_call
            if args
            else 0.0
        ),
        "direct_accept_risk": 1.0
        - (
            sum(r["correct"] for r in rows if not r["ar_called"]) / max(1, sum(1 for r in rows if not r["ar_called"]))
        ),
        "task_rows": task_rows,
    }


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


def plot_pareto(summary_rows, out_png, out_pdf):
    import matplotlib.pyplot as plt

    xs = [100 * float(r["ar_trigger_rate"]) for r in summary_rows]
    ys = [100 * float(r["macro_accuracy"]) for r in summary_rows]
    labels = [r["policy"] for r in summary_rows]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(xs, ys, s=70, edgecolor="black")
    for x, y, label in zip(xs, ys, labels):
        ax.text(x + 0.5, y + 0.05, label, fontsize=8)
    ax.set_xlabel("AR trigger rate (%)")
    ax.set_ylabel("Macro accuracy (%)")
    ax.set_title("Online bandit router: accuracy-cost tradeoff")
    ax.grid(color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-table", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-task", required=True)
    ap.add_argument("--out-router", required=True)
    ap.add_argument("--out-figure", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--calibration-fraction", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--lambda-llda-call", type=float, default=1 / 32)
    ap.add_argument("--lambda-ar-call", type=float, default=0.20)
    ap.add_argument("--seconds-per-llda-call", type=float, default=1.0)
    ap.add_argument("--seconds-per-ar-call", type=float, default=1.0)
    ap.add_argument("--min-ar-checkpoint", type=int, default=8)
    ap.add_argument("--max-checkpoint", type=int, default=32)
    ap.add_argument("--selective-epsilon", type=float, default=None)
    ap.add_argument("--selective-max-checkpoint", type=int, default=8)
    args = ap.parse_args()

    payload = json.loads(Path(args.decision_table).read_text())
    samples = payload["samples"]
    names = feature_schema(samples)
    calibration = [s for s in samples if split_name(s, args.seed, args.calibration_fraction) == "calibration"]
    test = [s for s in samples if split_name(s, args.seed, args.calibration_fraction) == "test"]

    rng = random.Random(args.seed)
    router = LinUCBRouter(len(names), alpha=args.alpha)
    direct_threshold = None
    if args.selective_epsilon is not None:
        direct_threshold = calibrate_direct_threshold(calibration, args.selective_epsilon, args.selective_max_checkpoint)
    for sample in calibration:
        run_episode(sample, router, names, rng, args, learn=True, direct_threshold=direct_threshold)

    rows = []
    for sample in test:
        rows.append(run_episode(sample, router, names, rng, args, learn=True, direct_threshold=direct_threshold))

    bandit = summarize(rows, "online_linucb", args)
    baselines = []
    for policy in ["llda_first_valid", "llda_last_observed", "ar_only", "all_multichoice_ar", "binary_llda32_multichoice_ar"]:
        baselines.append(summarize([baseline_episode(s, policy) for s in test], policy, args))
    random_rate = bandit["ar_trigger_rate"] / max(1e-9, sum(1 for s in test if len(s.get("label_space") or []) >= 3) / len(test))
    rand_rng = random.Random(args.seed)
    baselines.append(summarize([baseline_episode(s, "random_same_budget", rand_rng, random_rate) for s in test], "random_same_budget", args))
    summaries = [bandit] + baselines
    summary_rows = [
        {
            "policy": s["policy"],
            "n": s["n"],
            "macro_accuracy": s["macro_accuracy"],
            "micro_accuracy": s["micro_accuracy"],
            "avg_reward": s["avg_reward"],
            "ar_trigger_rate": s["ar_trigger_rate"],
            "avg_llda_calls": s["avg_llda_calls"],
            "estimated_seconds_per_sample": s["estimated_seconds_per_sample"],
            "direct_accept_risk": s["direct_accept_risk"],
        }
        for s in summaries
    ]
    task_rows = []
    for s in summaries:
        task_rows.extend(s["task_rows"])

    write_csv(args.out_summary, summary_rows)
    write_csv(args.out_task, task_rows)
    Path(args.out_rows).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_rows).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
    Path(args.out_router).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_router).write_text(
        json.dumps(
            {
                "feature_schema": names,
                "router": router.state_dict(),
                "args": vars(args),
                "n_calibration": len(calibration),
                "n_test": len(test),
                "direct_risk_threshold": direct_threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    plot_pareto(summary_rows, args.out_figure, str(Path(args.out_figure).with_suffix(".pdf")))
    print(json.dumps({"summary": summary_rows, "n_calibration": len(calibration), "n_test": len(test)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
