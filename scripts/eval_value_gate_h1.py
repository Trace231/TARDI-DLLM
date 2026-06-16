#!/usr/bin/env python3
"""Honest H=1 VALUE policy (the real 创新点2, not the single-threshold toy).

For each action a, fit q_a(x) = E[reward | x, a] = E[correct - lam*calls | x, a] via
GBDT on a TRAIN action table; at eval pick a* = argmax_a q_a(x) per sample (leak-free:
decision uses predicted q only; realized correctness read out after committing).
Compare to: single-threshold gate (best action,score,threshold), best fixed action,
accept-only. Answers "is H=1 more than a threshold, or is it intrinsically thresholdy?"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_llada_counterfactual_gate_policy as G


def feat_names(rows):
    k = set()
    for r in rows:
        k.update((r["features"] or {}).keys())
    return sorted(k)


def vec(f, names):
    return np.array([float((f or {}).get(n, 0.0) or 0.0) for n in names], dtype=np.float64)


def reward(a, lam):
    return float(a["correct"]) - lam * float(a["total_calls"])


def macro_acc(per_task):
    return float(np.mean([np.mean(v) for v in per_task.values()]))


def value_gate(train_rows, eval_rows, names, lam, seed):
    actions = sorted({a for r in train_rows for a in r["actions"]})
    qmods = {}
    for act in actions:
        X = [vec(r["features"], names) for r in train_rows if act in r["actions"]]
        Y = [reward(r["actions"][act], lam) for r in train_rows if act in r["actions"]]
        if len(X) < 20:
            continue
        qmods[act] = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=3, min_samples_leaf=20,
            random_state=seed).fit(np.vstack(X), np.array(Y))
    per_task, calls, picks = {}, [], {}
    for r in eval_rows:
        x = vec(r["features"], names).reshape(1, -1)
        cand = [(act, float(qmods[act].predict(x)[0])) for act in r["actions"] if act in qmods]
        if not cand:
            chosen = r["actions"]["accept_current"]
            act = "accept_current"
        else:
            act = max(cand, key=lambda t: t[1])[0]
            chosen = r["actions"][act]
        per_task.setdefault(r["task"], []).append(float(chosen["correct"]))
        calls.append(float(chosen["total_calls"]))
        picks[act] = picks.get(act, 0) + 1
    return macro_acc(per_task), float(np.mean(calls)), picks


def fixed_action(eval_rows, act):
    per_task, calls = {}, []
    for r in eval_rows:
        a = r["actions"].get(act, r["actions"]["accept_current"])
        per_task.setdefault(r["task"], []).append(float(a["correct"]))
        calls.append(float(a["total_calls"]))
    return macro_acc(per_task), float(np.mean(calls))


def thresh_gate(train_rows, eval_rows, lam, scores):
    actions = sorted({a for r in train_rows for a in r["actions"] if a != "accept_current"})
    best = G.tune(train_rows, actions, scores, "reward", lam)
    chosen = [G.choose(r, best["action"], best["score"], best["threshold"]) for r in eval_rows]
    per_task = {}
    for r, a in zip(eval_rows, chosen):
        per_task.setdefault(r["task"], []).append(float(a["correct"]))
    calls = float(np.mean([float(a["total_calls"]) for a in chosen]))
    return macro_acc(per_task), calls, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-table", required=True)
    ap.add_argument("--eval-table", required=True)
    ap.add_argument("--checkpoint", type=int, default=2)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--lambdas", default="0.0,0.005,0.01,0.02,0.03125,0.05")
    ap.add_argument("--scores", default="risk,probe_entropy,margin_deficit")
    args = ap.parse_args()
    scores = [s.strip() for s in args.scores.split(",") if s.strip()]

    tr = G.rows_from_payload(json.loads(Path(args.train_table).read_text()), args.checkpoint)
    ev = G.rows_from_payload(json.loads(Path(args.eval_table).read_text()), args.checkpoint)
    names = feat_names(tr)
    print(f"train_rows={len(tr)} eval_rows={len(ev)} feats={len(names)} "
          f"actions={sorted({a for r in tr for a in r['actions']})}")
    accept_acc, accept_calls = fixed_action(ev, "accept_current")
    print(f"accept_only                 acc={accept_acc:.3f}  calls={accept_calls:.2f}")
    for act in sorted({a for r in ev for a in r['actions']} - {"accept_current"}):
        fa, fc = fixed_action(ev, act)
        print(f"fixed_{act:<22} acc={fa:.3f}  calls={fc:.2f}")
    print("-" * 72)
    for lam in [float(x) for x in args.lambdas.split(",")]:
        ta, tc, best = thresh_gate(tr, ev, lam, scores)
        va, vc, picks = value_gate(tr, ev, names, lam, args.seed)
        print(f"lam={lam:<8} THRESH acc={ta:.3f} calls={tc:.2f} "
              f"[{best['action']} {best['score']}>={best['threshold']:.3g}]   "
              f"|| VALUE acc={va:.3f} calls={vc:.2f} picks={picks}")


if __name__ == "__main__":
    main()
