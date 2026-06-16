#!/usr/bin/env python3
"""Train a repair-aware LLaDA selective remasking policy.

The policy is decomposed into two supervised questions:

1. Is the current early LLaDA answer likely wrong?
2. If it is wrong, which LLaDA-only refinement action is likely to repair it?

This is closer to the intended controller than direct reward regression.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from train_llada_counterfactual_utility import flatten, group_rows, summarize, vectorize, write_csv


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_name(sample, seed, frac):
    return "calibration" if stable_value(seed, sample["task"], sample["id"]) < frac else "test"


def state_feature_names(rows):
    names = []
    seen = set()
    for row in rows:
        for key in row["features"]:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def state_vector(row, names):
    x = np.array([float(row["features"].get(name, 0.0) or 0.0) for name in names], dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x


def action_feature(row):
    a = row["action"]
    return {
        "is_restart": float(a.startswith("restart")),
        "is_projection": float(a == "label_projection"),
        "is_lowconf": float(a.startswith("remask_lowconf")),
        "is_label_only": float(a.startswith("remask_label_only")),
        "is_answer": float(a.startswith("remask_answer")),
        "is_structured": float(a.startswith("remask_structured")),
        "is_random": float(a.startswith("remask_random")),
        "target_norm": float(row["target_budget"]) / 32.0,
        "extra_calls_norm": float(row["extra_calls"]) / 32.0,
        "remasked_norm": float(row["remasked_tokens"]) / 32.0,
    }


def action_schema(rows):
    keys = list(action_feature(rows[0]).keys())
    return keys


def action_vector(row, state_names, action_names):
    vals = dict(row["features"])
    vals.update(action_feature(row))
    x = np.array([float(vals.get(name, 0.0) or 0.0) for name in [*state_names, *action_names]], dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x


def fit_classifier(X, y, seed):
    if len(set(y)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.75, class_weight="balanced", random_state=seed),
    )
    clf.fit(X, y)
    return clf


def fit_repair_model(cal_groups, state_names, action_names, seed, mode):
    X, y = [], []
    for group in cal_groups.values():
        accept = next((r for r in group if r["action"] == "accept_current"), None)
        if accept is None:
            continue
        accept_correct = bool(accept["correct"])
        for row in group:
            if row["action"] == "accept_current":
                continue
            if mode == "fix_wrong_only" and accept_correct:
                continue
            # Positive means the action repairs an accept error. For accept-correct
            # states, actions are not useful positives because they can only keep or harm.
            target = (not accept_correct) and bool(row["correct"])
            X.append(action_vector(row, state_names, action_names))
            y.append(int(target))
    return fit_classifier(np.vstack(X), np.array(y), seed) if X else None, len(X), int(sum(y))


def choose_policy(group, state_names, action_names, wrong_model, repair_model, threshold, lambda_call, min_repair_prob):
    accept = next((r for r in group if r["action"] == "accept_current"), None)
    if accept is None:
        return max(group, key=lambda r: r["reward"])
    if wrong_model is None or repair_model is None:
        return accept
    p_wrong = float(wrong_model.predict_proba(state_vector(accept, state_names)[None, :])[0, 1])
    if p_wrong < threshold:
        return accept
    candidates = [r for r in group if r["action"] != "accept_current"]
    scored = []
    for row in candidates:
        p_fix = float(repair_model.predict_proba(action_vector(row, state_names, action_names)[None, :])[0, 1])
        # Expected advantage over accepting current answer.
        utility = p_wrong * p_fix - lambda_call * max(0.0, float(row["total_calls"]) - float(accept["total_calls"]))
        scored.append((utility, p_fix, row))
    if not scored:
        return accept
    utility, p_fix, row = max(scored, key=lambda x: (x[0], x[1], -float(x[2]["total_calls"])))
    if p_fix < min_repair_prob or utility <= 0.0:
        return accept
    return row


def tune_threshold(cal_groups, state_names, action_names, wrong_model, repair_model, lambda_call, min_repair_prob):
    best = (0.5, -1e9)
    for threshold in [i / 100 for i in range(5, 96, 2)]:
        chosen = [
            choose_policy(group, state_names, action_names, wrong_model, repair_model, threshold, lambda_call, min_repair_prob)
            for group in cal_groups.values()
        ]
        reward = sum(r["reward"] for r in chosen) / max(1, len(chosen))
        if reward > best[1]:
            best = (threshold, reward)
    return best


def choose_fixed(groups, action):
    out = []
    for group in groups.values():
        match = [r for r in group if r["action"] == action]
        if match:
            out.append(match[0])
    return out


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
    ap.add_argument("--repair-mode", choices=["fix_wrong_only", "all_states"], default="fix_wrong_only")
    ap.add_argument("--min-repair-prob", type=float, default=0.0)
    args = ap.parse_args()

    payload = json.loads(Path(args.action_table).read_text())
    rows, _ = flatten(payload, args.lambda_call, args.lambda_remask)
    rows = [r for r in rows if int(r["checkpoint"]) == args.decision_checkpoint]
    cal_keys = {
        (s["task"], s["id"])
        for s in payload["samples"]
        if split_name(s, args.seed, args.calibration_fraction) == "calibration"
    }
    cal_rows = [r for r in rows if (r["task"], r["id"]) in cal_keys]
    test_rows = [r for r in rows if (r["task"], r["id"]) not in cal_keys]
    cal_groups = group_rows(cal_rows)
    test_groups = group_rows(test_rows)

    state_names = state_feature_names(rows)
    action_names = action_schema([r for r in rows if r["action"] != "accept_current"])

    accept_cal = [next(r for r in group if r["action"] == "accept_current") for group in cal_groups.values()]
    X_wrong = np.vstack([state_vector(r, state_names) for r in accept_cal])
    y_wrong = np.array([int(not r["correct"]) for r in accept_cal])
    wrong_model = fit_classifier(X_wrong, y_wrong, args.seed)
    repair_model, n_repair, n_repair_pos = fit_repair_model(cal_groups, state_names, action_names, args.seed, args.repair_mode)
    threshold, cal_reward = tune_threshold(
        cal_groups, state_names, action_names, wrong_model, repair_model, args.lambda_call, args.min_repair_prob
    )
    chosen = [
        choose_policy(group, state_names, action_names, wrong_model, repair_model, threshold, args.lambda_call, args.min_repair_prob)
        for group in test_groups.values()
    ]
    oracle = [max(group, key=lambda r: (r["reward"], r["correct"], -float(r["total_calls"]))) for group in test_groups.values()]

    policies = [
        summarize(chosen, "repair_aware_policy"),
        summarize(oracle, "oracle_action"),
    ]
    for action in [
        "accept_current",
        "restart_to_8",
        "label_projection",
        "remask_label_only_to_8",
        "remask_lowconf_to_8",
        "remask_answer_to_8",
        "remask_structured_to_8",
        "remask_random_to_8",
    ]:
        fixed = choose_fixed(test_groups, action)
        if fixed:
            policies.append(summarize(fixed, action))

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
            "chosen_action": r["action"],
            "correct": r["correct"],
            "reward": r["reward"],
            "total_calls": r["total_calls"],
            "remasked_tokens": r["remasked_tokens"],
        }
        for r in chosen
    ]
    write_csv(args.out_summary, summary_rows)
    write_csv(args.out_task, task_rows)
    write_csv(args.out_choices, choice_rows)
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_model).write_text(
        json.dumps(
            {
                "args": vars(args),
                "threshold": threshold,
                "cal_reward": cal_reward,
                "state_features": state_names,
                "action_features": action_names,
                "n_wrong_train": len(y_wrong),
                "wrong_positive": int(y_wrong.sum()),
                "n_repair_train": n_repair,
                "repair_positive": n_repair_pos,
                "chosen_actions": Counter(r["action"] for r in chosen),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"summary": summary_rows, "threshold": threshold, "chosen": Counter(r["action"] for r in chosen)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
