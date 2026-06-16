#!/usr/bin/env python3
"""Multi-step Markov stopping policy over LLaDA reverse-diffusion checkpoints.

At each checkpoint the policy observes features and chooses STOP (commit current
label) or CONTINUE (diffuse to the next checkpoint). Solved by fitted backward
induction (Longstaff-Schwartz): regress the continuation value on features at each
checkpoint; stop iff immediate stop-reward >= predicted continuation value. Fitted
on a TRAIN table, rolled out on a disjoint EVAL table (no leakage).
Reward = correct - lambda_call * spent_calls.

Accepts both the `online_decision_table_v1` schema (llda_correct / spent_llda_calls
/ router_features) and the `llda_counterfactual_actions_v1` schema (state_correct /
calls_so_far / features). The {STOP, CONTINUE} core here uses only the natural
trajectory snapshot at each checkpoint.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def stable_value(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def _norm_state(st):
    c = int(st["checkpoint"])
    correct = st.get("state_correct", st.get("llda_correct"))
    correct = 1.0 if correct else 0.0
    spent = st.get("calls_so_far", st.get("spent_llda_calls"))
    spent = float(spent if spent is not None else c)
    feats = st.get("features") or st.get("router_features") or {}
    pred = st.get("state_pred", st.get("llda_pred", ""))
    valid = st.get("llda_valid")
    if valid is None:
        valid = bool(str(pred).strip())
    return {"checkpoint": c, "correct": correct, "spent": spent, "features": feats, "valid": bool(valid)}


def load_states(payload):
    out = []
    for s in payload["samples"]:
        states = [_norm_state(st) for st in sorted(s["states"], key=lambda st: int(st["checkpoint"]))]
        out.append({"task": s["task"], "id": str(s["id"]), "states": states})
    return out


def feature_names(samples):
    keys = set()
    for s in samples:
        for st in s["states"]:
            keys.update(st["features"].keys())
    return sorted(keys)


def vec(features, names):
    return np.array([float(features.get(k, 0.0) or 0.0) for k in names], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-table", required=True)
    ap.add_argument("--train-table", default=None, help="separate train table; else hash-split eval-table")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--calibration-fraction", type=float, default=0.5)
    ap.add_argument("--lambdas", default="0.0,0.005,0.01,0.02,0.03")
    ap.add_argument("--require-valid-stop", action="store_true")
    ap.add_argument("--out-summary", default=None)
    args = ap.parse_args()

    eval_samples = load_states(json.loads(Path(args.eval_table).read_text()))
    if args.train_table:
        train_samples = load_states(json.loads(Path(args.train_table).read_text()))
        test_samples = eval_samples
    else:
        train_samples = [s for s in eval_samples if stable_value(args.seed, s["task"], s["id"]) < args.calibration_fraction]
        test_samples = [s for s in eval_samples if stable_value(args.seed, s["task"], s["id"]) >= args.calibration_fraction]

    names = feature_names(train_samples)
    checkpoints = sorted({st["checkpoint"] for s in train_samples for st in s["states"]})

    def rollout(samples, stop_models, cont_models, lam):
        # LEAK-FREE: stop decision uses ONLY feature-predicted stop value vs continuation value.
        # The realized `correct` is never observed at decision time; it is only read out AFTER committing.
        calls = []
        per_task = {}
        for s in samples:
            sts = s["states"]
            chosen = sts[-1]
            for i, st in enumerate(sts):
                if args.require_valid_stop and not st["valid"] and i < len(sts) - 1:
                    continue
                if st["checkpoint"] not in cont_models or i == len(sts) - 1:
                    chosen = st
                    break
                x = vec(st["features"], names).reshape(1, -1)
                stop_hat = float(stop_models[st["checkpoint"]].predict(x)[0])
                cont_hat = float(cont_models[st["checkpoint"]].predict(x)[0])
                if stop_hat >= cont_hat:
                    chosen = st
                    break
            calls.append(chosen["spent"])
            per_task.setdefault(s["task"], []).append(chosen["correct"])
        macro = float(np.mean([np.mean(v) for v in per_task.values()]))
        return macro, float(np.mean(calls))

    def fixed_stop(samples, c):
        pt = {}
        sp = []
        for s in samples:
            st = next((x for x in s["states"] if x["checkpoint"] == c), None)
            if st:
                pt.setdefault(s["task"], []).append(st["correct"]); sp.append(st["spent"])
        if not pt:
            return None
        return float(np.mean([np.mean(v) for v in pt.values()])), float(np.mean(sp))

    print(f"checkpoints={checkpoints}  n_train={len(train_samples)} n_test={len(test_samples)}  feat_dim={len(names)}")
    out_rows = []
    for lam in [float(x) for x in args.lambdas.split(",")]:
        V = {j: (s["states"][-1]["correct"] - lam * s["states"][-1]["spent"]) for j, s in enumerate(train_samples)}
        stop_models, cont_models = {}, {}
        for c in reversed(checkpoints[:-1]):
            X, Vcont, Rstop, idxs = [], [], [], []
            for j, s in enumerate(train_samples):
                st = next((x for x in s["states"] if x["checkpoint"] == c), None)
                if st is None:
                    continue
                X.append(vec(st["features"], names))
                Vcont.append(V[j])
                Rstop.append(st["correct"] - lam * st["spent"])
                idxs.append((j, st))
            if len(X) < 20:
                continue
            Xm = np.vstack(X)
            cont_m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=3,
                                                   min_samples_leaf=20, random_state=args.seed).fit(Xm, np.array(Vcont))
            stop_m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=3,
                                                   min_samples_leaf=20, random_state=args.seed).fit(Xm, np.array(Rstop))
            cont_models[c] = cont_m
            stop_models[c] = stop_m
            # value accounting under the LEAK-FREE (feature-predicted) policy:
            for k, (j, st) in enumerate(idxs):
                x = Xm[k].reshape(1, -1)
                if float(stop_m.predict(x)[0]) >= float(cont_m.predict(x)[0]):
                    V[j] = Rstop[k]  # policy stops -> realized stop reward (decision used only predictions)
        seq_acc, seq_calls = rollout(test_samples, stop_models, cont_models, lam)
        fx = [(c, *fixed_stop(test_samples, c)) for c in checkpoints if fixed_stop(test_samples, c)]
        bf = " ".join(f"@{c}:{a:.3f}/{s:.1f}" for c, a, s in fx)
        print(f"  lambda={lam:<6} seq_acc={seq_acc:.3f}  seq_calls={seq_calls:.2f}   | fixed: {bf}")
        out_rows.append({"lambda": lam, "seq_acc": round(seq_acc, 4), "seq_calls": round(seq_calls, 3)})
    if args.out_summary:
        import csv
        with open(args.out_summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["lambda", "seq_acc", "seq_calls"]); w.writeheader(); w.writerows(out_rows)
        print("saved", args.out_summary)


if __name__ == "__main__":
    main()
