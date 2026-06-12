import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as domain
import eval_subset as base
from eval_llada_adaptive_router import apply_prompt, choose_policy, infer_label_space, task_profile
from eval_llada_refinement_controller import maybe_enable_compact_choice_scout
from eval_llada_risk_controller import decode_with_scout_stats


def make_policy(first_policy, steps, schedule):
    policy = dict(first_policy)
    policy["steps"] = int(steps)
    policy["schedule"] = schedule
    if steps >= 32:
        policy["schedule"] = "uniform"
    return policy


def valid(pred, labels):
    return bool(pred) and (not labels or pred in labels)


def decode(model, tokenizer, sample, first_policy, steps, schedule, args, labels):
    policy = make_policy(first_policy, steps, schedule)
    output, pred, dt, calls, stats = decode_with_scout_stats(model, tokenizer, sample, policy, args, labels, {steps})
    return {"budget": steps, "schedule": policy["schedule"], "pred": pred, "output": output, "seconds": dt, "calls": calls, "stats": stats}


def choose_schedules(first_policy, args):
    primary = first_policy.get("schedule", args.primary_schedule)
    secondary = args.secondary_schedule
    if secondary == primary:
        secondary = "uniform" if primary != "uniform" else "middle_heavy"
    return primary, secondary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default="winogrande,commonsenseqa")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--compact-choice-fast", action="store_true")
    ap.add_argument("--compact-choice-max-labels", type=int, default=5)
    ap.add_argument("--compact-choice-max-prompt-tokens", type=int, default=512)
    ap.add_argument("--primary-schedule", default="back_loaded")
    ap.add_argument("--secondary-schedule", default="uniform")
    ap.add_argument("--mid-budget", type=int, default=16)
    ap.add_argument("--full-budget", type=int, default=32)
    ap.add_argument("--full-on-disagree", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = base.load_llada(args.model, args.adapter)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    summary = {}
    for task in [x for x in args.tasks.split(",") if x]:
        samples = domain.build_samples(task, args.limit, args.seed)
        correct = 0
        total_calls = 0
        total_time = 0.0
        route_counts = {}
        for raw in samples:
            profile = task_profile(raw, tokenizer)
            labels = infer_label_space(raw)
            first_policy = maybe_enable_compact_choice_scout(profile, choose_policy(profile), args)
            sample = apply_prompt(raw, first_policy["prompt"])
            primary, secondary = choose_schedules(first_policy, args)
            t0 = time.time()
            runs = []
            a = decode(model, tokenizer, sample, first_policy, 8, primary, args, labels)
            b = decode(model, tokenizer, sample, first_policy, 8, secondary, args, labels)
            runs.extend([a, b])
            calls = a["calls"] + b["calls"]
            route = ["8x2"]
            pred = a["pred"]
            output = a["output"]
            reason = "dual8_agree"
            if not (valid(a["pred"], labels) and a["pred"] == b["pred"]):
                if args.full_on_disagree:
                    c = decode(model, tokenizer, sample, first_policy, args.full_budget, "uniform", args, labels)
                    route.append(str(args.full_budget))
                else:
                    c = decode(model, tokenizer, sample, first_policy, args.mid_budget, "uniform", args, labels)
                    route.append(str(args.mid_budget))
                    if not valid(c["pred"], labels):
                        d = decode(model, tokenizer, sample, first_policy, args.full_budget, "uniform", args, labels)
                        runs.append(d)
                        calls += d["calls"]
                        c = d
                        route.append(str(args.full_budget))
                runs.append(c)
                calls += c["calls"]
                pred = c["pred"]
                output = c["output"]
                reason = "dual8_disagree_upgrade"
            dt = time.time() - t0
            ok = pred == raw["gold"]
            correct += int(ok)
            total_calls += calls
            total_time += dt
            route_s = "->".join(route)
            route_counts[route_s] = route_counts.get(route_s, 0) + 1
            row = {
                "task": task,
                "id": raw["id"],
                "gold": raw["gold"],
                "pred": pred,
                "correct": ok,
                "output": output,
                "route": route_s,
                "reason": reason,
                "runs": runs,
                "forward_calls": calls,
                "seconds": dt,
                "metric": raw["metric"],
                "meta": raw.get("meta", {}),
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ["task", "id", "gold", "pred", "correct", "route", "forward_calls"]}, ensure_ascii=False), flush=True)
        n = len(samples)
        summary[task] = {
            "accuracy": correct / n,
            "n": n,
            "seconds": total_time,
            "avg_forward_calls": total_calls / n,
            "route_rates": {k: v / n for k, v in sorted(route_counts.items())},
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        Path(args.out).write_text(json.dumps({"method": "schedule_ensemble_controller", "model": args.model, "args": vars(args), "complete": False, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    payload = {"method": "schedule_ensemble_controller", "model": args.model, "args": vars(args), "complete": True, "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
