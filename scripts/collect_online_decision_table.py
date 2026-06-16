#!/usr/bin/env python3
"""Build an online-decision table from paired LLaDA and AR logs.

This script is intentionally conservative: `router_features` only contains
features observable at the current probe/scout checkpoint. Gold labels and
counterfactual action outcomes are stored outside the feature dictionary so the
router can be tested for leakage.
"""

import argparse
import json
import math
from pathlib import Path


def load_rows(path):
    payload = json.loads(Path(path).read_text())
    return payload, {(r["task"], str(r["id"])): r for r in payload.get("rows", [])}


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        out = float(value)
        return default if math.isnan(out) or math.isinf(out) else out
    except Exception:
        return default


def norm_entropy(probs):
    vals = [max(1e-12, safe_float(v)) for v in (probs or {}).values()]
    if not vals:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    vals = [v / total for v in vals]
    ent = -sum(v * math.log(v) for v in vals)
    return ent / math.log(len(vals)) if len(vals) > 1 else 0.0


def profile_features(lrow, checkpoint, spent_calls):
    profile = lrow.get("profile") or {}
    probe = lrow.get("probe") or {}
    probs = probe.get("probs") or {}
    n_labels = int(profile.get("n_labels") or 0)
    top = safe_float(probe.get("top_prob"))
    margin = safe_float(probe.get("margin"))
    entropy = norm_entropy(probs)
    return {
        "bias": 1.0,
        "checkpoint_norm": checkpoint / 32.0,
        "spent_llda_calls_norm": spent_calls / 32.0,
        "n_labels": float(n_labels),
        "log_n_labels": math.log(max(1, n_labels)),
        "is_binary": 1.0 if n_labels == 2 else 0.0,
        "is_multichoice": 1.0 if n_labels >= 3 else 0.0,
        "prompt_tokens_norm": safe_float(profile.get("prompt_tokens")) / 256.0,
        "probe_available": 1.0 if probe.get("available") else 0.0,
        "probe_top_prob": top,
        "probe_margin": margin,
        "probe_entropy": entropy,
        "probe_bayes_risk": 1.0 - top,
    }


def rolling_history_features(history, checkpoint, probe_top_label):
    prefix = [h for h in history if int(h.get("local_step", -1)) <= checkpoint]
    valid = [h for h in prefix if h.get("valid")]
    preds = [h.get("pred") for h in valid if h.get("pred")]
    flips = sum(1 for a, b in zip(preds, preds[1:]) if a != b)
    first_valid = safe_float(valid[0].get("local_step")) if valid else 0.0
    pred = preds[-1] if preds else ""
    return {
        "scout_seen": 1.0 if prefix else 0.0,
        "scout_valid": 1.0 if pred else 0.0,
        "scout_filled_ratio": safe_float(prefix[-1].get("filled_ratio")) if prefix else 0.0,
        "early_flip_count_norm": min(flips, 4) / 4.0,
        "first_valid_step_norm": first_valid / 32.0,
        "valid_seen_norm": min(len(valid), 8) / 8.0,
        "probe_scout_disagree": 1.0 if pred and probe_top_label and pred != probe_top_label else 0.0,
    }, pred, bool(pred)


def risk_feature_projection(features):
    allowed = [
        "probe_uncertainty",
        "probe_entropy",
        "margin_deficit",
        "probe_scout_disagree",
        "invalid_or_empty",
        "flip_instability",
        "late_first_final",
        "low_fill_confidence",
        "label_complexity",
        "prompt_complexity",
    ]
    return {f"risk_{k}": safe_float((features or {}).get(k)) for k in allowed}


def route_step_states(lrow):
    states = {}
    for step in lrow.get("route_steps") or []:
        budget = int(step.get("budget") or 0)
        states[budget] = {
            "pred": step.get("pred") or "",
            "valid": bool(step.get("pred")),
            "risk_score_visible": safe_float(step.get("risk_score")),
            "risk_features_visible": risk_feature_projection(step.get("features") or {}),
        }
    return states


def build_sample(lrow, jrow, checkpoints):
    gold = lrow.get("gold")
    probe = lrow.get("probe") or {}
    probe_pred = probe.get("top_label") or ""
    probe_valid = bool(probe_pred)
    history = (lrow.get("scout_stats") or {}).get("history") or []
    route_states = route_step_states(lrow)
    observed = set()
    if probe.get("available"):
        observed.add(0)
    for h in history:
        observed.add(int(h.get("local_step") or 0))
    observed.update(route_states)
    observed = sorted(c for c in observed if c in checkpoints)
    if not observed:
        observed = [0]

    states = []
    probe_top_label = probe.get("top_label")
    for checkpoint in observed:
        spent = 1 if checkpoint == 0 else max(1, checkpoint + 1)
        feats = profile_features(lrow, checkpoint, spent)
        hist_feats, hist_pred, hist_valid = rolling_history_features(history, checkpoint, probe_top_label)
        feats.update(hist_feats)
        pred = probe_pred if checkpoint == 0 else hist_pred
        valid = probe_valid if checkpoint == 0 else hist_valid
        if checkpoint in route_states:
            pred = route_states[checkpoint]["pred"] or pred
            valid = route_states[checkpoint]["valid"] or valid
            feats["visible_risk_score"] = route_states[checkpoint]["risk_score_visible"]
            feats.update(route_states[checkpoint]["risk_features_visible"])
        else:
            feats["visible_risk_score"] = 0.0
            feats.update(risk_feature_projection({}))
        states.append(
            {
                "checkpoint": checkpoint,
                "spent_llda_calls": spent,
                "llda_pred": pred,
                "llda_valid": bool(valid),
                "llda_correct": bool(pred and pred == gold),
                "router_features": feats,
            }
        )

    for i, state in enumerate(states):
        state["has_next_state"] = i + 1 < len(states)
        state["next_checkpoint"] = states[i + 1]["checkpoint"] if i + 1 < len(states) else None

    ar_pred = jrow.get("pred") or ""
    return {
        "task": lrow.get("task"),
        "id": str(lrow.get("id")),
        "metric": lrow.get("metric"),
        "label_space": (lrow.get("profile") or {}).get("label_space") or [],
        "gold": gold,
        "ar_pred": ar_pred,
        "ar_correct": bool(ar_pred and ar_pred == gold),
        "states": states,
    }


def validate_no_leakage(samples):
    banned = {
        "gold",
        "correct",
        "route",
        "final_budget",
        "forward_calls",
        "full_forward_calls",
        "oracle",
    }
    offenders = []
    for sample in samples:
        for state in sample["states"]:
            for key in state["router_features"]:
                if key in banned or any(part == key for part in banned):
                    offenders.append((sample["task"], sample["id"], state["checkpoint"], key))
    return offenders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("LLADA_JSON", "AR_JSON"))
    ap.add_argument("--checkpoints", default="0,4,8,16,24,32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checkpoints = sorted({int(x) for x in args.checkpoints.split(",") if x.strip()})
    samples = []
    source_files = []
    for llada_path, ar_path in args.pair:
        source_files.extend([llada_path, ar_path])
        _, llada = load_rows(llada_path)
        _, ar = load_rows(ar_path)
        for key in sorted(set(llada) & set(ar)):
            samples.append(build_sample(llada[key], ar[key], checkpoints))

    offenders = validate_no_leakage(samples)
    if offenders:
        raise SystemExit(f"router feature leakage detected: {offenders[:5]}")

    payload = {
        "schema": "online_decision_table_v1",
        "checkpoints": checkpoints,
        "source_files": source_files,
        "n": len(samples),
        "samples": samples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"out": args.out, "n": len(samples), "checkpoints": checkpoints}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
