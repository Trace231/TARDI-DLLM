import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as domain
import eval_subset as base
from eval_llada_adaptive_router import (
    apply_prompt,
    choose_policy,
    infer_label_space,
    parse_for_metric,
    probe_label_distribution,
    run_decode,
    task_profile,
)


def normalized_entropy(probs):
    vals = np.array(list(probs.values()), dtype=np.float64)
    if len(vals) <= 1:
        return 0.0
    vals = np.clip(vals, 1e-12, 1.0)
    ent = float(-(vals * np.log(vals)).sum())
    return ent / float(np.log(len(vals)))


def budget_policy(first_policy, steps):
    policy = dict(first_policy)
    policy["steps"] = int(steps)
    policy["fast_candidate"] = steps < 32
    if steps >= 32:
        policy["schedule"] = "uniform"
        policy["fast_candidate"] = False
    policy["reason"] = f"multi_budget_{steps}_step"
    return policy


def parse_budgets(text):
    budgets = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not budgets:
        raise ValueError("at least one budget is required")
    if budgets[-1] != 32:
        budgets.append(32)
    return budgets


def next_budget(current, budgets):
    for budget in budgets:
        if budget > current:
            return budget
    return current


def choose_initial_budget(profile, probe, first_policy, budgets, args):
    if not first_policy.get("fast_candidate"):
        return 32, "profile_conservative"
    if not probe.get("available"):
        return first_policy["steps"], "probe_unavailable"

    top_prob = probe.get("top_prob", 1.0)
    margin = probe.get("margin", 1.0)
    ent = normalized_entropy(probe.get("probs", {}))
    n_labels = profile.get("n_labels", 0)

    if n_labels <= 2:
        if top_prob < args.binary_full_threshold:
            return 32, "binary_very_low_probe_confidence"
        if top_prob < args.binary_24_threshold:
            return 24, "binary_low_probe_confidence"
        if top_prob < args.binary_16_threshold or margin < args.binary_margin_threshold:
            return 16, "binary_medium_probe_confidence"
        return min(budgets), "binary_probe_confident"

    if n_labels >= 5:
        if top_prob < args.multi_full_threshold or ent > args.multi_entropy_full:
            return 32, "multi_high_entropy_or_low_confidence"
        if top_prob < args.multi_24_threshold or ent > args.multi_entropy_24:
            return 24, "multi_moderate_high_entropy"
        if top_prob < args.multi_16_threshold or ent > args.multi_entropy_16:
            return 16, "multi_moderate_entropy"
        return min(budgets), "multi_probe_confident"

    return first_policy["steps"], "default_fast_policy"


def risk_signal(pred, labels, probe, profile, args):
    if labels and pred not in labels:
        return True, "invalid_label", 2
    if not probe.get("available"):
        return False, "no_probe_available", 0
    if not pred:
        return True, "empty_prediction", 2

    top_prob = probe.get("top_prob", 1.0)
    margin = probe.get("margin", 1.0)
    ent = normalized_entropy(probe.get("probs", {}))
    n_labels = profile.get("n_labels", 0)
    disagree = pred != probe.get("top_label")

    if n_labels <= 2:
        if top_prob < args.binary_full_threshold:
            return True, "binary_probe_very_uncertain", 2
        if disagree and top_prob >= args.binary_disagree_confidence:
            return True, "binary_probe_scout_disagreement", 1
        if top_prob < args.binary_16_threshold and margin < args.binary_margin_threshold:
            return True, "binary_low_margin", 1
        return False, "binary_accepted", 0

    if n_labels >= 5:
        if args.multi_disagreement_policy == "fallback" and disagree and top_prob >= args.multi_disagree_confidence:
            return True, "multi_probe_scout_disagreement", 1
        if top_prob < args.multi_full_threshold or ent > args.multi_entropy_full:
            return True, "multi_high_uncertainty", 2
        return False, "multi_accepted", 0

    return False, "accepted", 0


def run_one_budget(model, tokenizer, sample, first_policy, budget, args, labels):
    policy = budget_policy(first_policy, budget)
    output, dt, calls = run_decode(model, tokenizer, sample, policy, args)
    pred = parse_for_metric(sample, output)
    return output, pred, dt, calls


def summarize_route(route_steps):
    return "->".join(str(step["steps"]) for step in route_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="winogrande,commonsenseqa")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="8,16,24,32")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--max-escalations", type=int, default=3)
    ap.add_argument("--binary-16-threshold", type=float, default=0.78)
    ap.add_argument("--binary-24-threshold", type=float, default=0.70)
    ap.add_argument("--binary-full-threshold", type=float, default=0.60)
    ap.add_argument("--binary-margin-threshold", type=float, default=0.12)
    ap.add_argument("--binary-disagree-confidence", type=float, default=0.70)
    ap.add_argument("--multi-16-threshold", type=float, default=0.62)
    ap.add_argument("--multi-24-threshold", type=float, default=0.52)
    ap.add_argument("--multi-full-threshold", type=float, default=0.42)
    ap.add_argument("--multi-entropy-16", type=float, default=0.82)
    ap.add_argument("--multi-entropy-24", type=float, default=0.90)
    ap.add_argument("--multi-entropy-full", type=float, default=0.96)
    ap.add_argument("--multi-disagree-confidence", type=float, default=0.62)
    ap.add_argument("--multi-disagreement-policy", choices=["fallback", "ignore"], default="ignore")
    args = ap.parse_args()

    budgets = parse_budgets(args.budgets)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = base.load_llada(args.model)
    torch.cuda.reset_peak_memory_stats()

    rows = []
    summary = {}
    for task in [t for t in args.tasks.split(",") if t]:
        raw_samples = domain.build_samples(task, args.limit, args.seed)
        correct = 0
        total_time = 0.0
        total_calls = 0
        route_counts = {}
        final_budget_counts = {}
        for raw in raw_samples:
            profile = task_profile(raw, tokenizer)
            labels = infer_label_space(raw)
            first_policy = choose_policy(profile)
            sample = apply_prompt(raw, first_policy["prompt"])
            t0 = time.time()
            probe = probe_label_distribution(model, tokenizer, sample, labels)
            calls = 1 if probe.get("available") else 0
            route_steps = []

            budget, reason = choose_initial_budget(profile, probe, first_policy, budgets, args)
            escalations = 0
            output = ""
            pred = ""
            final_risk_reason = reason
            while True:
                output, pred, dt_decode, decode_calls = run_one_budget(
                    model, tokenizer, sample, first_policy, budget, args, labels
                )
                calls += decode_calls
                risky, risk_reason, severity = risk_signal(pred, labels, probe, profile, args)
                route_steps.append(
                    {
                        "steps": budget,
                        "reason": reason,
                        "pred": pred,
                        "risk_after_decode": risky,
                        "risk_reason": risk_reason,
                    }
                )
                final_risk_reason = risk_reason
                if not risky or budget >= max(budgets) or escalations >= args.max_escalations:
                    break
                old_budget = budget
                if severity >= 2:
                    budget = max(budgets)
                else:
                    budget = next_budget(budget, budgets)
                if budget == old_budget:
                    break
                reason = risk_reason
                escalations += 1

            dt = time.time() - t0
            ok = pred == raw["gold"]
            correct += int(ok)
            total_time += dt
            total_calls += calls
            route = summarize_route(route_steps)
            route_counts[route] = route_counts.get(route, 0) + 1
            final_budget_counts[str(route_steps[-1]["steps"])] = final_budget_counts.get(str(route_steps[-1]["steps"]), 0) + 1
            row = {
                "task": task,
                "id": raw["id"],
                "gold": raw["gold"],
                "pred": pred,
                "correct": ok,
                "output": output,
                "metric": raw["metric"],
                "profile": profile,
                "probe": probe,
                "probe_entropy_norm": normalized_entropy(probe.get("probs", {})) if probe.get("available") else None,
                "first_policy": first_policy,
                "route": route,
                "route_steps": route_steps,
                "final_budget": route_steps[-1]["steps"],
                "final_risk_reason": final_risk_reason,
                "seconds": dt,
                "forward_calls": calls,
                "meta": raw.get("meta", {}),
            }
            rows.append(row)
            print(
                json.dumps(
                    {k: row[k] for k in ["task", "id", "gold", "pred", "correct", "route", "forward_calls"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )

        n = len(raw_samples)
        summary[task] = {
            "accuracy": correct / n,
            "n": n,
            "seconds": total_time,
            "avg_forward_calls": total_calls / n,
            "route_rates": {k: v / n for k, v in sorted(route_counts.items())},
            "final_budget_rates": {k: v / n for k, v in sorted(final_budget_counts.items())},
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    payload = {
        "method": "multi_budget_controller",
        "model": args.model,
        "args": vars(args),
        "summary": summary,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
