#!/usr/bin/env python3
"""Full-action multi-step MDP: a STRICT generalization of the H=1 value policy.
At every checkpoint the action set is {STOP} U {REMASK_to_t for every available target t} U {CONTINUE}
and we value EACH action separately (per-target remask, not the collapsed 'nearest remask' of the old
solver). Leak-free fitted backward induction: decisions use predicted values only; realized correctness
read out after committing. Fit on TRAIN table, roll out on disjoint EVAL table.
H=1 value gate == this with a single checkpoint; old multi-step == this with remask collapsed to nearest.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

TRAJ_KEYS = ["margin_deficit", "low_fill_confidence", "probe_entropy", "risk_score",
             "flip_instability", "probe_margin", "probe_bayes_risk", "probe_uncertainty"]
VM = {"iters": 300, "lr": 0.06, "depth": 3, "leaf": 20}


def fit(X, Y, seed):
    return HistGradientBoostingRegressor(max_iter=VM["iters"], learning_rate=VM["lr"], max_depth=VM["depth"],
                                         min_samples_leaf=VM["leaf"], random_state=seed).fit(np.vstack(X), np.array(Y))


def remasks_at(st, c):
    """All available lowconf remask targets t>c -> {t: action dict}."""
    out = {}
    for a in st.get("actions", []):
        name = a.get("action", "")
        if name.startswith("remask_lowconf_to_"):
            t = int(name.rsplit("_", 1)[-1])
            if t > c:
                out[t] = a
    return out


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
            rms = {}
            for t, a in remasks_at(st, c).items():
                rc = 1.0 if a.get("correct") else 0.0
                rs = float(a.get("total_calls", c))
                rms[t] = {"correct": rc, "spent": rs, "r": rc - lam * rs}
            sts.append({
                "c": c, "feats": feats, "valid": bool(pred.strip()), "pred": pred,
                "stop": {"correct": correct, "spent": spent, "r": correct - lam * spent},
                "remasks": rms,
            })
        for i in range(len(sts)):  # causal trajectory features (states <= current; leak-free)
            f = sts[i]["feats"]
            flips = sum(1 for j in range(1, i + 1)
                        if sts[j]["pred"] and sts[j - 1]["pred"] and sts[j]["pred"] != sts[j - 1]["pred"])
            run = 1
            for j in range(i, 0, -1):
                if sts[j]["pred"] and sts[j - 1]["pred"] and sts[j]["pred"] == sts[j - 1]["pred"]:
                    run += 1
                else:
                    break
            f["traj_flip_count"] = float(flips); f["traj_stable_run"] = float(run)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-table", required=True)
    ap.add_argument("--eval-table", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--lambdas", default="0.0,0.005,0.01,0.02,0.03")
    ap.add_argument("--vm-depth", type=int, default=3)
    ap.add_argument("--vm-iters", type=int, default=300)
    ap.add_argument("--vm-leaf", type=int, default=20)
    args = ap.parse_args()
    VM.update({"depth": args.vm_depth, "iters": args.vm_iters, "leaf": args.vm_leaf})

    for lam in [float(x) for x in args.lambdas.split(",")]:
        tr = load(args.train_table, lam)
        ev = load(args.eval_table, lam)
        names = feat_names(tr)
        cps = sorted({st["c"] for s in tr for st in s["states"]})
        all_targets = sorted({t for s in tr for st in s["states"] for t in st["remasks"]})

        def immediate_best(st):
            r = st["stop"]["r"]
            for t in st["remasks"]:
                r = max(r, st["remasks"][t]["r"])
            return r

        V = {j: immediate_best(s["states"][-1]) for j, s in enumerate(tr)}
        stop_m, rem_m, cont_m = {}, {}, {}
        for c in reversed(cps):
            rows = [(j, next((x for x in s["states"] if x["c"] == c), None)) for j, s in enumerate(tr)]
            rows = [(j, st) for j, st in rows if st is not None]
            if len(rows) < 20:
                continue
            X = [vec(st["feats"], names) for _, st in rows]
            stop_m[c] = fit(X, [st["stop"]["r"] for _, st in rows], args.seed)
            rem_m[c] = {}
            for t in all_targets:
                sub = [(x, st["remasks"][t]["r"]) for x, (_, st) in zip(X, rows) if t in st["remasks"]]
                if len(sub) >= 20:
                    rem_m[c][t] = fit([x for x, _ in sub], [y for _, y in sub], args.seed)
            is_last = (c == cps[-1])
            if not is_last:
                cont_m[c] = fit(X, [V[j] for j, _ in rows], args.seed)
            for k, (j, st) in enumerate(rows):
                x = X[k].reshape(1, -1)
                qs = float(stop_m[c].predict(x)[0])
                best, real = qs, st["stop"]["r"]
                for t in st["remasks"]:
                    if t in rem_m[c]:
                        q = float(rem_m[c][t].predict(x)[0])
                        if q > best:
                            best, real = q, st["remasks"][t]["r"]
                if not is_last and c in cont_m:
                    if float(cont_m[c].predict(x)[0]) > best:
                        real = V[j]  # continue: keep propagated value
                V[j] = real

        # rollout on EVAL (leak-free)
        per_task, calls, acts = {}, [], {"STOP": 0, "REMASK": 0, "CONTINUE": 0}
        tgt_hist = {}
        for s in ev:
            chosen, sts = None, s["states"]
            for i, st in enumerate(sts):
                c = st["c"]
                last = (i == len(sts) - 1) or (c not in stop_m)
                x = vec(st["feats"], names).reshape(1, -1)
                qs = float(stop_m[c].predict(x)[0]) if c in stop_m else st["stop"]["r"]
                best_a, best_q = ("STOP", None), qs
                for t in st["remasks"]:
                    if c in rem_m and t in rem_m[c]:
                        q = float(rem_m[c][t].predict(x)[0])
                        if q > best_q:
                            best_q, best_a = q, ("REMASK", t)
                if not last and c in cont_m:
                    if float(cont_m[c].predict(x)[0]) > best_q:
                        best_a = ("CONTINUE", None)
                if best_a[0] == "CONTINUE":
                    acts["CONTINUE"] += 1
                    continue
                if best_a[0] == "REMASK":
                    acts["REMASK"] += 1
                    tgt_hist[best_a[1]] = tgt_hist.get(best_a[1], 0) + 1
                    rm = st["remasks"][best_a[1]]
                    chosen = (rm["correct"], rm["spent"])
                else:
                    acts["STOP"] += 1
                    chosen = (st["stop"]["correct"], st["stop"]["spent"])
                break
            if chosen is None:
                last = sts[-1]
                chosen = (last["stop"]["correct"], last["stop"]["spent"])
            per_task.setdefault(s["task"], []).append(chosen[0])
            calls.append(chosen[1])
        macro = float(np.mean([np.mean(v) for v in per_task.values()]))
        print(f"lambda={lam:<6} fullaction_acc={macro:.3f}  avg_calls={np.mean(calls):.2f}  "
              f"actions={acts}  remask_targets={tgt_hist}")


if __name__ == "__main__":
    main()
