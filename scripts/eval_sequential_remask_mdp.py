#!/usr/bin/env python3
"""Multi-step Markov policy over LLaDA checkpoints with 3 actions:
   STOP (commit current label), REMASK (local lowconf remask -> commit), CONTINUE (advance).
Leak-free fitted backward induction: at each checkpoint fit stop / remask / continuation
value models from FEATURES only; choose argmax of predicted values. Realized correctness
is never observed at decision time (only read out after committing).
Fitted on a TRAIN table, rolled out on a disjoint EVAL table.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def stable_value(seed, *p):
    return int(hashlib.md5((":".join([str(seed), *map(str, p)])).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def nearest_remask(st, c, lam):
    """Return (correct, spent) of the nearest available lowconf remask (smallest target > c)."""
    best = None
    for a in st.get("actions", []):
        name = a.get("action", "")
        if name.startswith("remask_lowconf_to_"):
            tgt = int(name.rsplit("_", 1)[-1])
            if tgt > c and (best is None or tgt < best[0]):
                best = (tgt, a)
    if best is None:
        return None
    a = best[1]
    correct = 1.0 if a.get("correct") else 0.0
    spent = float(a.get("total_calls", c))
    return correct, spent


TRAJ_KEYS = ["margin_deficit", "low_fill_confidence", "probe_entropy", "risk_score",
             "flip_instability", "probe_margin", "probe_bayes_risk", "probe_uncertainty"]


def load(path, lam):
    d = json.loads(Path(path).read_text())
    out = []
    for s in d["samples"]:
        sts = []
        for st in sorted(s["states"], key=lambda x: int(x["checkpoint"])):
            c = int(st["checkpoint"])
            correct = 1.0 if st.get("state_correct") else 0.0
            spent = float(st.get("calls_so_far", c) or c)
            feats = dict(st.get("features") or {})
            pred = str(st.get("state_pred", "") or "")
            valid = bool(pred.strip())
            rk = nearest_remask(st, c, lam)
            sts.append({
                "c": c, "feats": feats, "valid": valid, "pred": pred,
                "stop_r": correct - lam * spent, "stop_correct": correct, "stop_spent": spent,
                "remask_r": (rk[0] - lam * rk[1]) if rk else (correct - lam * spent),
                "remask_correct": rk[0] if rk else correct, "remask_spent": rk[1] if rk else spent,
                "has_remask": rk is not None,
            })
        # causal cross-checkpoint trajectory features (uses only states <= current; no leakage)
        for i in range(len(sts)):
            f = sts[i]["feats"]
            flips = sum(1 for j in range(1, i + 1)
                        if sts[j]["pred"] and sts[j - 1]["pred"] and sts[j]["pred"] != sts[j - 1]["pred"])
            run = 1
            for j in range(i, 0, -1):
                if sts[j]["pred"] and sts[j - 1]["pred"] and sts[j]["pred"] == sts[j - 1]["pred"]:
                    run += 1
                else:
                    break
            f["traj_flip_count"] = float(flips)
            f["traj_stable_run"] = float(run)
            f["traj_changed_last"] = float(bool(i > 0 and sts[i]["pred"] and sts[i - 1]["pred"]
                                                and sts[i]["pred"] != sts[i - 1]["pred"]))
            f["traj_n_seen"] = float(i + 1)
            f["traj_distinct_preds"] = float(len({sts[j]["pred"] for j in range(i + 1) if sts[j]["pred"]}))
            if i > 0:
                prev = sts[i - 1]["feats"]
                for k in TRAJ_KEYS:
                    if k in f and k in prev:
                        try:
                            f[f"traj_d_{k}"] = float(f[k]) - float(prev[k])
                        except (TypeError, ValueError):
                            pass
        out.append({"task": s["task"], "id": str(s["id"]), "states": sts})
    return out


def feat_names(samples):
    k = set()
    for s in samples:
        for st in s["states"]:
            k.update(st["feats"].keys())
    return sorted(k)


def vec(f, names):
    return np.array([float(f.get(n, 0.0) or 0.0) for n in names], dtype=np.float64)


VM = {"iters": 200, "lr": 0.06, "depth": 3, "leaf": 20}


def fit(X, Y, seed):
    return HistGradientBoostingRegressor(max_iter=VM["iters"], learning_rate=VM["lr"], max_depth=VM["depth"],
                                         min_samples_leaf=VM["leaf"], random_state=seed).fit(np.vstack(X), np.array(Y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-table", required=True)
    ap.add_argument("--eval-table", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--lambdas", default="0.0,0.005,0.01,0.02,0.03,0.05")
    ap.add_argument("--require-valid-stop", action="store_true")
    ap.add_argument("--vm-iters", type=int, default=200)
    ap.add_argument("--vm-lr", type=float, default=0.06)
    ap.add_argument("--vm-depth", type=int, default=3)
    ap.add_argument("--vm-leaf", type=int, default=20)
    args = ap.parse_args()
    VM.update({"iters": args.vm_iters, "lr": args.vm_lr, "depth": args.vm_depth, "leaf": args.vm_leaf})

    for lam in [float(x) for x in args.lambdas.split(",")]:
        tr = load(args.train_table, lam)
        ev = load(args.eval_table, lam)
        names = feat_names(tr)
        cps = sorted({st["c"] for s in tr for st in s["states"]})

        # backward induction on TRAIN: fit stop/remask/cont value models, decide by predicted values
        V = {}
        for j, s in enumerate(tr):
            last = s["states"][-1]
            V[j] = max(last["stop_r"], last["remask_r"] if last["has_remask"] else last["stop_r"])
        stop_m, remask_m, cont_m = {}, {}, {}
        for c in reversed(cps):
            rows = [(j, next((x for x in s["states"] if x["c"] == c), None)) for j, s in enumerate(tr)]
            rows = [(j, st) for j, st in rows if st is not None]
            if len(rows) < 20:
                continue
            X = [vec(st["feats"], names) for _, st in rows]
            stop_m[c] = fit(X, [st["stop_r"] for _, st in rows], args.seed)
            remask_m[c] = fit(X, [st["remask_r"] for _, st in rows], args.seed)
            is_last = (c == cps[-1])
            if not is_last:
                cont_m[c] = fit(X, [V[j] for j, _ in rows], args.seed)
            for k, (j, st) in enumerate(rows):
                x = X[k].reshape(1, -1)
                qs = float(stop_m[c].predict(x)[0])
                qr = float(remask_m[c].predict(x)[0]) if st["has_remask"] else -1e9
                qc = float(cont_m[c].predict(x)[0]) if (not is_last and c in cont_m) else -1e9
                best = max(qs, qr, qc)
                if best == qc:
                    pass  # continue: V[j] unchanged
                elif best == qr:
                    V[j] = st["remask_r"]
                else:
                    V[j] = st["stop_r"]

        # rollout on EVAL (leak-free)
        per_task, calls, acts = {}, [], {"STOP": 0, "REMASK": 0, "CONTINUE": 0}
        for s in ev:
            chosen = None
            sts = s["states"]
            for i, st in enumerate(sts):
                c = st["c"]
                if args.require_valid_stop and not st["valid"] and i < len(sts) - 1:
                    continue
                last = (i == len(sts) - 1) or (c not in stop_m)
                x = vec(st["feats"], names).reshape(1, -1)
                qs = float(stop_m[c].predict(x)[0]) if c in stop_m else st["stop_r"]
                qr = float(remask_m[c].predict(x)[0]) if (st["has_remask"] and c in remask_m) else -1e9
                qc = float(cont_m[c].predict(x)[0]) if (not last and c in cont_m) else -1e9
                best = max(qs, qr, qc)
                if best == qc and not last:
                    acts["CONTINUE"] += 1
                    continue
                if best == qr:
                    acts["REMASK"] += 1
                    chosen = (st["remask_correct"], st["remask_spent"])
                else:
                    acts["STOP"] += 1
                    chosen = (st["stop_correct"], st["stop_spent"])
                break
            if chosen is None:
                last = sts[-1]
                chosen = (last["stop_correct"], last["stop_spent"])
            per_task.setdefault(s["task"], []).append(chosen[0])
            calls.append(chosen[1])
        macro = float(np.mean([np.mean(v) for v in per_task.values()]))
        print(f"lambda={lam:<6} seq_remask_acc={macro:.3f}  avg_calls={np.mean(calls):.2f}   actions={acts}")


if __name__ == "__main__":
    main()
