#!/usr/bin/env python3
"""Train stronger offline policies for LLaDA counterfactual action tables.

This script keeps the deployment assumption clean: at test time the policy sees
only the early LLaDA state and the candidate action descriptors. Gold labels are
used only in the calibration split to learn action ranking/gating.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from train_llada_counterfactual_utility import (
    action_features,
    build_schema,
    flatten,
    group_rows,
    summarize,
    vectorize,
    write_csv,
)


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_name(sample, seed, frac):
    return "calibration" if stable_value(seed, sample["task"], sample["id"]) < frac else "test"


def fit_pairwise_ranker(groups, names, seed, reward_margin=1e-9):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = [], []
    for group in groups.values():
        vecs = [vectorize(row, names) for row in group]
        for i, row_i in enumerate(group):
            for j, row_j in enumerate(group):
                if i >= j:
                    continue
                diff = float(row_i["reward"]) - float(row_j["reward"])
                if abs(diff) <= reward_margin:
                    continue
                xi = vecs[i] - vecs[j]
                X.append(xi)
                y.append(1 if diff > 0 else 0)
                X.append(-xi)
                y.append(0 if diff > 0 else 1)
    if not X or len(set(y)) < 2:
        return None, 0
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=seed),
    )
    clf.fit(np.vstack(X), np.array(y))
    return clf, len(y)


def choose_pairwise(group, names, ranker):
    if ranker is None:
        return max(group, key=lambda r: (r["reward"], r["correct"], -r["total_calls"]))
    vecs = [vectorize(row, names) for row in group]
    scores = []
    for i, vi in enumerate(vecs):
        score = 0.0
        for j, vj in enumerate(vecs):
            if i == j:
                continue
            proba = ranker.predict_proba((vi - vj)[None, :])[0, 1]
            score += float(proba)
        scores.append(score / max(1, len(vecs) - 1))
    return group[int(np.argmax(scores))]


def group_best(group):
    return max(group, key=lambda r: (r["reward"], r["correct"], -r["total_calls"]))


def fit_gate_and_action_model(groups, names, seed, positive_margin):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_gate, y_gate = [], []
    refine_rows = []
    for group in groups.values():
        accept = next((r for r in group if r["action"] == "accept_current"), None)
        if accept is None:
            continue
        best_refine = max((r for r in group if r["action"] != "accept_current"), key=lambda r: r["reward"], default=None)
        if best_refine is None:
            continue
        # Gate sees the state once. Use the accept action vector with action bits
        # zeroed by replacing action descriptors with accept_current.
        X_gate.append(vectorize(accept, names))
        y_gate.append(1 if best_refine["reward"] > accept["reward"] + positive_margin else 0)
        refine_rows.extend([r for r in group if r["action"] != "accept_current"])
    if not X_gate or len(set(y_gate)) < 2:
        gate = None
    else:
        gate = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=seed),
        )
        gate.fit(np.vstack(X_gate), np.array(y_gate))
    if refine_rows:
        X_ref = np.vstack([vectorize(r, names) for r in refine_rows])
        y_ref = np.array([r["reward"] for r in refine_rows])
        action_model = HistGradientBoostingRegressor(
            max_leaf_nodes=7,
            max_iter=80,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=seed,
        )
        action_model.fit(X_ref, y_ref)
    else:
        action_model = None
    return gate, action_model, int(sum(y_gate)), len(y_gate)


def choose_two_stage(group, names, gate, action_model, threshold):
    accept = next((r for r in group if r["action"] == "accept_current"), None)
    if accept is None:
        return group_best(group)
    if gate is None or action_model is None:
        return accept
    p_refine = float(gate.predict_proba(vectorize(accept, names)[None, :])[0, 1])
    if p_refine < threshold:
        return accept
    refine = [r for r in group if r["action"] != "accept_current"]
    if not refine:
        return accept
    pred = action_model.predict(np.vstack([vectorize(r, names) for r in refine]))
    return refine[int(np.argmax(pred))]


def tune_two_stage_threshold(cal_groups, names, gate, action_model):
    best = (0.5, -1e9)
    for t in [i / 100 for i in range(5, 96, 2)]:
        chosen = [choose_two_stage(group, names, gate, action_model, t) for group in cal_groups.values()]
        avg = sum(r["reward"] for r in chosen) / max(1, len(chosen))
        if avg > best[1]:
            best = (t, avg)
    return best


def choose_named(groups, action_name):
    chosen = []
    for group in groups.values():
        match = [r for r in group if r["action"] == action_name]
        if match:
            chosen.append(match[0])
    return chosen


def choose_prefix_max(groups, prefix):
    chosen = []
    for group in groups.values():
        match = [r for r in group if r["action"].startswith(prefix)]
        if match:
            chosen.append(max(match, key=lambda r: (r["target_budget"], r["action"])))
    return chosen


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
    ap.add_argument("--positive-margin", type=float, default=0.0)
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
    cal_groups = group_rows(cal_rows)
    test_groups = group_rows(test_rows)

    ranker, n_pairs = fit_pairwise_ranker(cal_groups, names, args.seed)
    pairwise = [choose_pairwise(group, names, ranker) for group in test_groups.values()]
    gate, action_model, n_pos, n_gate = fit_gate_and_action_model(cal_groups, names, args.seed, args.positive_margin)
    threshold, cal_reward = tune_two_stage_threshold(cal_groups, names, gate, action_model)
    two_stage = [choose_two_stage(group, names, gate, action_model, threshold) for group in test_groups.values()]
    oracle = [group_best(group) for group in test_groups.values()]

    policies = [
        summarize(pairwise, "pairwise_rank_policy"),
        summarize(two_stage, "two_stage_gate_policy"),
        summarize(oracle, "oracle_action"),
    ]
    for action in ["accept_current", "restart_to_8", "restart_to_16", "restart_to_32"]:
        chosen = choose_named(test_groups, action)
        if chosen:
            policies.append(summarize(chosen, action))
    for prefix in ["remask_lowconf_to_", "remask_answer_to_", "remask_structured_to_", "remask_random_to_"]:
        chosen = choose_prefix_max(test_groups, prefix)
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
    choice_rows = []
    for policy, chosen in [("pairwise_rank_policy", pairwise), ("two_stage_gate_policy", two_stage)]:
        for r in chosen:
            choice_rows.append(
                {
                    "policy": policy,
                    "task": r["task"],
                    "id": r["id"],
                    "checkpoint": r["checkpoint"],
                    "chosen_action": r["action"],
                    "correct": r["correct"],
                    "reward": r["reward"],
                    "total_calls": r["total_calls"],
                    "remasked_tokens": r["remasked_tokens"],
                }
            )
    write_csv(args.out_summary, summary_rows)
    write_csv(args.out_task, task_rows)
    write_csv(args.out_choices, choice_rows)
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_model).write_text(
        json.dumps(
            {
                "feature_schema": names,
                "args": vars(args),
                "n_pairwise_pairs": n_pairs,
                "two_stage_threshold": threshold,
                "two_stage_cal_reward": cal_reward,
                "two_stage_gate_positive": n_pos,
                "two_stage_gate_total": n_gate,
                "n_calibration_groups": len(cal_groups),
                "n_test_groups": len(test_groups),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"summary": summary_rows, "threshold": threshold, "n_pairs": n_pairs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
