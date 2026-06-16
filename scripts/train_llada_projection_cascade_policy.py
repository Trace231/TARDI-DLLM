#!/usr/bin/env python3
"""Evaluate a projection-first cascade for LLaDA fixed-label inference.

The cascade models a deployable sequence:

1. Run probe + 4-step scout.
2. Optionally spend one extra LLaDA call to project the scout context onto the
   legal final-label posterior.
3. After observing projection confidence/margin, choose either the original
   scout label or the projected label.

This separates two questions that a one-shot action ranker conflates:

- pre-call value: is a projection call worth paying for this state?
- post-call selection: after seeing the projected posterior, should we switch?
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from train_llada_counterfactual_utility import summarize, write_csv


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_name(sample, seed, frac):
    return "calibration" if stable_value(seed, sample["task"], sample["id"]) < frac else "test"


def groups_from_payload(payload, checkpoint, lambda_call):
    rows = []
    for sample in payload["samples"]:
        for state in sample["states"]:
            if int(state["checkpoint"]) != checkpoint:
                continue
            acts = {a["action"]: dict(a) for a in state["actions"]}
            if "accept_current" not in acts or "label_projection" not in acts:
                continue
            feats = dict(state.get("features") or {})
            projection_plan = acts["label_projection"].get("remask_plan") or {}
            row = {
                "task": sample["task"],
                "id": sample["id"],
                "features": feats,
                "accept": acts["accept_current"],
                "projection": acts["label_projection"],
                "projection_top_prob": float(projection_plan.get("top_prob", 0.0) or 0.0),
                "projection_margin": float(projection_plan.get("margin", 0.0) or 0.0),
                "projection_disagrees": float(acts["accept_current"].get("pred") != acts["label_projection"].get("pred")),
                "lambda_call": lambda_call,
            }
            rows.append(row)
    return rows


def state_names(rows):
    names, seen = [], set()
    for row in rows:
        for key in row["features"]:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def pre_vector(row, names):
    vals = row["features"]
    x = np.array([float(vals.get(name, 0.0) or 0.0) for name in names], dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x


def post_vector(row, names):
    vals = dict(row["features"])
    vals.update(
        {
            "projection_top_prob": row["projection_top_prob"],
            "projection_margin": row["projection_margin"],
            "projection_disagrees": row["projection_disagrees"],
        }
    )
    full_names = [*names, "projection_top_prob", "projection_margin", "projection_disagrees"]
    x = np.array([float(vals.get(name, 0.0) or 0.0) for name in full_names], dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x


def fit_regressor(X, y, seed, model_type):
    if len(X) == 0:
        return None
    if model_type == "linear":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.04,
            max_leaf_nodes=8,
            l2_regularization=0.2,
            random_state=seed,
        )
    model.fit(X, y)
    return model


def action_row(row, source, total_calls=None):
    base = row[source]
    out = {
        "task": row["task"],
        "id": row["id"],
        "checkpoint": 4,
        "action": source,
        "correct": bool(base["correct"]),
        "total_calls": float(base["total_calls"] if total_calls is None else total_calls),
        "remasked_tokens": float(base.get("remasked_tokens", 0.0)),
    }
    out["reward"] = float(out["correct"]) - row["lambda_call"] * out["total_calls"]
    return out


def choose(rows, pre_model, post_model, names, lambda_call, pre_margin, post_margin, mode):
    chosen = []
    for row in rows:
        accept = action_row(row, "accept")
        projection = action_row(row, "projection")
        if mode == "accept":
            chosen.append(accept)
            continue
        if mode == "projection":
            chosen.append(projection)
            continue
        call_projection = True
        if mode == "gated":
            pred_pre = float(pre_model.predict(pre_vector(row, names)[None, :])[0]) if pre_model is not None else 0.0
            call_projection = pred_pre - pre_margin > lambda_call
        if not call_projection:
            chosen.append(accept)
            continue
        pred_post = float(post_model.predict(post_vector(row, names)[None, :])[0]) if post_model is not None else 0.0
        # If we called projection but keep the scout answer, the extra call has
        # still been paid.
        if pred_post - post_margin > 0.0:
            chosen.append(projection)
        else:
            chosen.append(action_row(row, "accept", total_calls=float(projection["total_calls"])))
    return chosen


def tune(cal_rows, pre_model, post_model, names, lambda_call):
    best = ("gated", 0.0, 0.0, -1e9)
    for mode in ["projection_switch", "gated"]:
        for pre_margin in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
            for post_margin in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
                chosen = choose(cal_rows, pre_model, post_model, names, lambda_call, pre_margin, post_margin, mode)
                reward = sum(r["reward"] for r in chosen) / max(1, len(chosen))
                if reward > best[3]:
                    best = (mode, pre_margin, post_margin, reward)
    return best


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
    ap.add_argument("--model-type", choices=["gbrt", "linear"], default="gbrt")
    args = ap.parse_args()

    payload = json.loads(Path(args.action_table).read_text())
    rows = groups_from_payload(payload, args.decision_checkpoint, args.lambda_call)
    cal_keys = {
        (s["task"], s["id"])
        for s in payload["samples"]
        if split_name(s, args.seed, args.calibration_fraction) == "calibration"
    }
    cal = [r for r in rows if (r["task"], r["id"]) in cal_keys]
    test = [r for r in rows if (r["task"], r["id"]) not in cal_keys]
    names = state_names(rows)

    X_pre = np.vstack([pre_vector(r, names) for r in cal])
    y_pre = np.array([float(r["projection"]["correct"]) - float(r["accept"]["correct"]) for r in cal], dtype=np.float64)
    pre_model = fit_regressor(X_pre, y_pre, args.seed, args.model_type)
    X_post = np.vstack([post_vector(r, names) for r in cal])
    y_post = y_pre
    post_model = fit_regressor(X_post, y_post, args.seed, args.model_type)

    mode, pre_margin, post_margin, cal_reward = tune(cal, pre_model, post_model, names, args.lambda_call)
    policies = []
    for label, chosen in [
        ("accept_current", choose(test, pre_model, post_model, names, args.lambda_call, pre_margin, post_margin, "accept")),
        ("label_projection", choose(test, pre_model, post_model, names, args.lambda_call, pre_margin, post_margin, "projection")),
        ("projection_cascade", choose(test, pre_model, post_model, names, args.lambda_call, pre_margin, post_margin, mode)),
    ]:
        policies.append(summarize(chosen, label))

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
    chosen = choose(test, pre_model, post_model, names, args.lambda_call, pre_margin, post_margin, mode)
    choice_rows = [
        {
            "task": r["task"],
            "id": r["id"],
            "chosen_action": r["action"],
            "correct": r["correct"],
            "reward": r["reward"],
            "total_calls": r["total_calls"],
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
                "mode": mode,
                "pre_margin": pre_margin,
                "post_margin": post_margin,
                "cal_reward": cal_reward,
                "features": names,
                "chosen_actions": Counter(r["action"] for r in chosen),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"summary": summary_rows, "mode": mode, "pre_margin": pre_margin, "post_margin": post_margin}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
