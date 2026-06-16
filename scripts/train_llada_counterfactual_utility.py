#!/usr/bin/env python3
"""Train/evaluate an offline utility policy for LLaDA counterfactual actions."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_name(sample, seed, frac):
    return "calibration" if stable_value(seed, sample["task"], sample["id"]) < frac else "test"


def flatten(payload, lambda_call, lambda_remask):
    rows = []
    feature_names = []
    seen = set()
    for sample in payload["samples"]:
        split_key = (sample["task"], sample["id"])
        for state in sample["states"]:
            base_feats = state.get("features") or {}
            for key in base_feats:
                if key not in seen:
                    seen.add(key)
                    feature_names.append(key)
            for action in state["actions"]:
                reward = float(action["correct"]) - lambda_call * float(action["total_calls"]) - lambda_remask * float(action["remasked_tokens"])
                family = action["action"].split("_to_")[0]
                rows.append(
                    {
                        "task": sample["task"],
                        "id": sample["id"],
                        "checkpoint": state["checkpoint"],
                        "action": action["action"],
                        "family": family,
                        "target_budget": action["target_budget"],
                        "correct": bool(action["correct"]),
                        "total_calls": float(action["total_calls"]),
                        "extra_calls": float(action["extra_calls"]),
                        "remasked_tokens": float(action["remasked_tokens"]),
                        "reward": reward,
                        "features": base_feats,
                    }
                )
    return rows, feature_names


def action_features(row):
    action = row["action"]
    family = row["family"]
    return {
        "action_accept": 1.0 if action == "accept_current" else 0.0,
        "action_projection": 1.0 if action == "label_projection" else 0.0,
        "action_restart": 1.0 if family == "restart" else 0.0,
        "action_remask_label_only": 1.0 if family == "remask_label_only" else 0.0,
        "action_remask_lowconf": 1.0 if family == "remask_lowconf" else 0.0,
        "action_remask_answer": 1.0 if family == "remask_answer" else 0.0,
        "action_remask_structured": 1.0 if family == "remask_structured" else 0.0,
        "action_remask_random": 1.0 if family == "remask_random" else 0.0,
        "target_norm": float(row["target_budget"]) / 32.0,
        "budget_delta_norm": max(0.0, float(row["target_budget"]) - float(row["checkpoint"])) / 32.0,
        "extra_calls_norm": float(row["extra_calls"]) / 32.0,
        "remasked_norm": float(row["remasked_tokens"]) / 32.0,
    }


def build_schema(feature_names, rows):
    names = list(feature_names)
    for key in action_features(rows[0]).keys():
        names.append(key)
    return names


def vectorize(row, names):
    feats = dict(row["features"])
    feats.update(action_features(row))
    x = np.array([float(feats.get(name, 0.0) or 0.0) for name in names], dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x


def fit_model(X, y, seed):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_leaf_nodes=9,
            max_iter=80,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=seed,
        )
        model.fit(X, y)
        return model, "hist_gradient_boosting"
    except Exception:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X, y)
        return model, "ridge_fallback"


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["id"], row["checkpoint"])].append(row)
    return groups


def summarize(chosen, policy):
    by_task = defaultdict(list)
    for row in chosen:
        by_task[row["task"]].append(row)
    task_rows = []
    for task, group in sorted(by_task.items()):
        n = len(group)
        task_rows.append(
            {
                "policy": policy,
                "task": task,
                "n": n,
                "accuracy": sum(r["correct"] for r in group) / n,
                "avg_reward": sum(r["reward"] for r in group) / n,
                "avg_calls": sum(r["total_calls"] for r in group) / n,
                "avg_remasked": sum(r["remasked_tokens"] for r in group) / n,
            }
        )
    return {
        "policy": policy,
        "n": len(chosen),
        "macro_accuracy": sum(r["accuracy"] for r in task_rows) / len(task_rows),
        "micro_accuracy": sum(r["correct"] for r in chosen) / len(chosen),
        "avg_reward": sum(r["reward"] for r in chosen) / len(chosen),
        "avg_calls": sum(r["total_calls"] for r in chosen) / len(chosen),
        "avg_remasked": sum(r["remasked_tokens"] for r in chosen) / len(chosen),
        "task_rows": task_rows,
    }


def choose_by_action(groups, action_name):
    chosen = []
    for group in groups.values():
        match = [r for r in group if r["action"] == action_name]
        if match:
            chosen.append(match[0])
    return chosen


def choose_by_prefix(groups, prefix):
    chosen = []
    for group in groups.values():
        match = [r for r in group if r["action"].startswith(prefix)]
        if match:
            chosen.append(sorted(match, key=lambda r: (r["target_budget"], r["action"]))[-1])
    return chosen


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
    ap.add_argument("--out-choices", required=True)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--calibration-fraction", type=float, default=0.5)
    ap.add_argument("--decision-checkpoint", type=int, default=4)
    ap.add_argument("--lambda-call", type=float, default=1 / 32)
    ap.add_argument("--lambda-remask", type=float, default=0.0)
    args = ap.parse_args()

    payload = json.loads(Path(args.action_table).read_text())
    rows, base_names = flatten(payload, args.lambda_call, args.lambda_remask)
    rows = [r for r in rows if int(r["checkpoint"]) == args.decision_checkpoint]
    names = build_schema(base_names, rows)
    cal_keys = {
        (s["task"], s["id"])
        for s in payload["samples"]
        if split_name(s, args.seed, args.calibration_fraction) == "calibration"
    }
    cal_rows = [r for r in rows if (r["task"], r["id"]) in cal_keys]
    test_rows = [r for r in rows if (r["task"], r["id"]) not in cal_keys]
    X = np.vstack([vectorize(r, names) for r in cal_rows])
    y = np.array([r["reward"] for r in cal_rows], dtype=np.float64)
    model, model_name = fit_model(X, y, args.seed)

    test_groups = group_rows(test_rows)
    learned = []
    for group in test_groups.values():
        Xg = np.vstack([vectorize(r, names) for r in group])
        pred = model.predict(Xg)
        learned.append(group[int(np.argmax(pred))])
    oracle = [max(group, key=lambda r: (r["reward"], r["correct"], -r["total_calls"])) for group in test_groups.values()]

    policies = [
        summarize(learned, "learned_utility"),
        summarize(oracle, "oracle_action"),
    ]
    for action in ["accept_current", "label_projection", "restart_to_8", "restart_to_16", "restart_to_32"]:
        chosen = choose_by_action(test_groups, action)
        if chosen:
            policies.append(summarize(chosen, action))
    for prefix in ["remask_label_only_to_", "remask_lowconf_to_", "remask_answer_to_", "remask_structured_to_", "remask_random_to_"]:
        chosen = choose_by_prefix(test_groups, prefix)
        if chosen:
            policies.append(summarize(chosen, prefix + "max"))

    summary_rows = [
        {
            "policy": p["policy"],
            "n": p["n"],
            "macro_accuracy": p["macro_accuracy"],
            "micro_accuracy": p["micro_accuracy"],
            "avg_reward": p["avg_reward"],
            "avg_calls": p["avg_calls"],
            "avg_remasked": p["avg_remasked"],
        }
        for p in policies
    ]
    task_rows = []
    for p in policies:
        task_rows.extend(p["task_rows"])
    choice_rows = [
        {
            "task": r["task"],
            "id": r["id"],
            "checkpoint": r["checkpoint"],
            "chosen_action": r["action"],
            "correct": r["correct"],
            "reward": r["reward"],
            "total_calls": r["total_calls"],
            "remasked_tokens": r["remasked_tokens"],
        }
        for r in learned
    ]
    write_csv(args.out_summary, summary_rows)
    write_csv(args.out_task, task_rows)
    write_csv(args.out_choices, choice_rows)
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_model).write_text(
        json.dumps(
            {
                "model": model_name,
                "feature_schema": names,
                "args": vars(args),
                "n_calibration_action_rows": len(cal_rows),
                "n_test_action_rows": len(test_rows),
                "n_test_states": len(test_groups),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"summary": summary_rows, "model": model_name}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
