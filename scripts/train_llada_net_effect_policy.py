#!/usr/bin/env python3
"""Train a net-effect policy for LLaDA-only selective remasking.

The earlier repair policy asks whether a refinement can fix a currently wrong
answer. That misses the main failure mode of selective remasking: a refinement
can also damage a currently correct answer. This policy instead learns the
action-level treatment effect

    delta(a, x) = 1[action a is correct] - 1[accept_current is correct].

At inference time it selects a refinement only when the predicted lower-bound
net effect exceeds the extra LLaDA cost. All features are early-state features
plus action descriptors from the counterfactual table; gold labels are used only
on the calibration split to fit the effect model.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from train_llada_counterfactual_utility import flatten, group_rows, summarize, write_csv


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_name(sample, seed, frac):
    return "calibration" if stable_value(seed, sample["task"], sample["id"]) < frac else "test"


def state_feature_names(rows):
    names, seen = [], set()
    for row in rows:
        for key in row["features"]:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def action_feature(row):
    action = row["action"]
    plan = row.get("remask_plan") or {}
    return {
        "is_restart": float(action.startswith("restart")),
        "is_projection": float(action == "label_projection"),
        "is_label_only": float(action.startswith("remask_label_only")),
        "is_lowconf": float(action.startswith("remask_lowconf")),
        "is_answer": float(action.startswith("remask_answer")),
        "is_structured": float(action.startswith("remask_structured")),
        "is_random": float(action.startswith("remask_random")),
        "target_norm": float(row.get("target_budget", 0.0)) / 32.0,
        "extra_calls_norm": float(row.get("extra_calls", 0.0)) / 32.0,
        "remasked_norm": float(row.get("remasked_tokens", 0.0)) / 32.0,
        "label_token_hits_norm": float(plan.get("label_token_hits", 0.0) or 0.0) / 4.0,
        "marker_hits_norm": float(plan.get("marker_hits", 0.0) or 0.0) / 4.0,
    }


def action_schema(rows):
    names, seen = [], set()
    for row in rows:
        for key in action_feature(row):
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def vector(row, state_names, action_names):
    vals = dict(row["features"])
    vals.update(action_feature(row))
    out = np.array([float(vals.get(name, 0.0) or 0.0) for name in [*state_names, *action_names]], dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    return out


def fit_effect_model(groups, state_names, action_names, seed, model_type):
    X, y = [], []
    for group in groups.values():
        accept = next((r for r in group if r["action"] == "accept_current"), None)
        if accept is None:
            continue
        accept_correct = int(bool(accept["correct"]))
        for row in group:
            if row["action"] == "accept_current":
                continue
            X.append(vector(row, state_names, action_names))
            y.append(int(bool(row["correct"])) - accept_correct)
    if not X:
        return None, np.array([]), np.array([])
    X = np.vstack(X)
    y = np.asarray(y, dtype=np.float64)
    if model_type == "linear":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=8,
            l2_regularization=0.2,
            random_state=seed,
        )
    model.fit(X, y)
    pred = np.asarray(model.predict(X), dtype=np.float64)
    return model, pred, y


def choose(group, model, state_names, action_names, lambda_call, lambda_remask, lcb_margin):
    accept = next((r for r in group if r["action"] == "accept_current"), None)
    if accept is None or model is None:
        return max(group, key=lambda r: r["reward"])
    candidates = [r for r in group if r["action"] != "accept_current"]
    if not candidates:
        return accept
    X = np.vstack([vector(r, state_names, action_names) for r in candidates])
    pred_delta = np.asarray(model.predict(X), dtype=np.float64)
    scored = []
    for row, delta in zip(candidates, pred_delta):
        extra_calls = max(0.0, float(row["total_calls"]) - float(accept["total_calls"]))
        remask_cost = float(row.get("remasked_tokens", 0.0)) / 32.0
        net = float(delta) - lcb_margin - lambda_call * extra_calls - lambda_remask * remask_cost
        scored.append((net, float(delta), -extra_calls, row))
    net, _, _, row = max(scored, key=lambda item: (item[0], item[1], item[2]))
    if net <= 0.0:
        return accept
    return row


def evaluate_choice(groups, model, state_names, action_names, lambda_call, lambda_remask, lcb_margin):
    chosen = [
        choose(group, model, state_names, action_names, lambda_call, lambda_remask, lcb_margin)
        for group in groups.values()
    ]
    return chosen, summarize(chosen, "net_effect_policy")


def tune_margin(groups, model, state_names, action_names, pred_cal, y_cal, args):
    residual_margin = 0.0
    if len(y_cal):
        # One-sided conservative margin: how much the model overestimates delta.
        over = pred_cal - y_cal
        residual_margin = float(np.quantile(over, args.conformal_quantile))
        residual_margin = max(0.0, residual_margin)
    candidates = sorted(set([0.0, residual_margin, *[i / 100 for i in range(0, 51, 5)]]))
    best = (0.0, -1e9, None)
    for margin in candidates:
        chosen, stats = evaluate_choice(
            groups, model, state_names, action_names, args.lambda_call, args.lambda_remask, margin
        )
        reward = float(stats["avg_reward"])
        if reward > best[1]:
            best = (margin, reward, chosen)
    return best[0], best[1], residual_margin


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
    ap.add_argument("--model-type", choices=["gbrt", "linear"], default="gbrt")
    ap.add_argument("--conformal-quantile", type=float, default=0.80)
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
    model, pred_cal, y_cal = fit_effect_model(cal_groups, state_names, action_names, args.seed, args.model_type)
    margin, cal_reward, conformal_margin = tune_margin(
        cal_groups, model, state_names, action_names, pred_cal, y_cal, args
    )
    chosen, policy_stats = evaluate_choice(
        test_groups, model, state_names, action_names, args.lambda_call, args.lambda_remask, margin
    )
    oracle = [max(group, key=lambda r: (r["reward"], r["correct"], -float(r["total_calls"]))) for group in test_groups.values()]

    policies = [policy_stats, summarize(oracle, "oracle_action")]
    action_names_present = sorted({r["action"] for r in rows})
    for action in action_names_present:
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
                "state_features": state_names,
                "action_features": action_names,
                "cal_reward": cal_reward,
                "selected_lcb_margin": margin,
                "conformal_overestimate_margin": conformal_margin,
                "effect_train_n": int(len(y_cal)),
                "effect_train_positive": int((y_cal > 0).sum()),
                "effect_train_negative": int((y_cal < 0).sum()),
                "chosen_actions": Counter(r["action"] for r in chosen),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "summary": summary_rows,
                "selected_lcb_margin": margin,
                "chosen_actions": Counter(r["action"] for r in chosen),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
