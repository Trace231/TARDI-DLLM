import argparse
import json
import sys
import time
from pathlib import Path

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


def conservative_policy(prompt_kind="typed"):
    return {
        "steps": 32,
        "schedule": "uniform",
        "prompt": prompt_kind,
        "fast_candidate": False,
        "reason": "forward-aware fallback",
    }


def pre_scout_risk(profile, probe, binary_confidence_threshold):
    if not probe.get("available"):
        return False, "probe_unavailable"
    if profile.get("n_labels", 0) <= 2 and probe.get("top_prob", 1.0) < binary_confidence_threshold:
        return True, "binary_low_probe_confidence"
    return False, "pre_scout_safe"


def post_scout_risk(pred, labels, probe):
    if labels and pred not in labels:
        return True, "invalid_label"
    if probe.get("available") and pred and pred != probe.get("top_label"):
        return True, "probe_scout_disagreement"
    return False, "post_scout_safe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="winogrande,commonsenseqa")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--binary-confidence-threshold", type=float, default=0.70)
    args = ap.parse_args()

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
        pre_fallback_count = 0
        post_fallback_count = 0
        accepted_count = 0
        for raw in raw_samples:
            profile = task_profile(raw, tokenizer)
            labels = infer_label_space(raw)
            first_policy = choose_policy(profile)
            sample = apply_prompt(raw, first_policy["prompt"])

            t0 = time.time()
            probe = probe_label_distribution(model, tokenizer, sample, labels)
            calls = 1 if probe.get("available") else 0
            output = ""
            pred = ""
            route = "fast"
            route_reason = "not_started"
            final_policy = dict(first_policy)

            if not first_policy.get("fast_candidate"):
                route = "conservative"
                route_reason = "policy_not_fast_candidate"
                final_policy = conservative_policy(first_policy["prompt"])
                final_sample = apply_prompt(raw, final_policy["prompt"])
                output, dt_decode, decode_calls = run_decode(model, tokenizer, final_sample, final_policy, args)
                pred = parse_for_metric(final_sample, output)
                calls += decode_calls
            else:
                pre_risk, pre_reason = pre_scout_risk(profile, probe, args.binary_confidence_threshold)
                if pre_risk:
                    route = "pre_fallback"
                    route_reason = pre_reason
                    pre_fallback_count += 1
                    final_policy = conservative_policy(first_policy["prompt"])
                    final_sample = apply_prompt(raw, final_policy["prompt"])
                    output, dt_decode, decode_calls = run_decode(model, tokenizer, final_sample, final_policy, args)
                    pred = parse_for_metric(final_sample, output)
                    calls += decode_calls
                else:
                    output, dt_decode, decode_calls = run_decode(model, tokenizer, sample, first_policy, args)
                    pred = parse_for_metric(sample, output)
                    calls += decode_calls
                    post_risk, post_reason = post_scout_risk(pred, labels, probe)
                    if post_risk:
                        route = "post_fallback"
                        route_reason = post_reason
                        post_fallback_count += 1
                        final_policy = conservative_policy(first_policy["prompt"])
                        final_sample = apply_prompt(raw, final_policy["prompt"])
                        output, dt2, decode_calls2 = run_decode(model, tokenizer, final_sample, final_policy, args)
                        pred = parse_for_metric(final_sample, output)
                        calls += decode_calls2
                    else:
                        route = "accepted_fast"
                        route_reason = post_reason
                        accepted_count += 1

            dt = time.time() - t0
            ok = pred == raw["gold"]
            correct += int(ok)
            total_time += dt
            total_calls += calls
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
                "first_policy": first_policy,
                "final_policy": final_policy,
                "route": route,
                "route_reason": route_reason,
                "seconds": dt,
                "forward_calls": calls,
                "meta": raw.get("meta", {}),
            }
            rows.append(row)
            print(
                json.dumps(
                    {k: row[k] for k in ["task", "id", "gold", "pred", "correct", "route", "route_reason", "forward_calls"]},
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
            "pre_fallback_rate": pre_fallback_count / n,
            "post_fallback_rate": post_fallback_count / n,
            "accepted_fast_rate": accepted_count / n,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    payload = {"model": args.model, "args": vars(args), "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
