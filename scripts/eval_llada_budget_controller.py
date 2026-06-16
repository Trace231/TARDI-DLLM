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
from eval_llada_sampler_variants import MASK_ID, model_logits, token_budget, x0_and_conf


def budget_policy(first_policy, steps):
    policy = dict(first_policy)
    policy["steps"] = steps
    policy["fast_candidate"] = steps < 32
    if steps >= 32:
        policy["schedule"] = "uniform"
        policy["fast_candidate"] = False
    policy["reason"] = f"budget_controller_{steps}_step"
    return policy


def normalized_entropy(probs):
    vals = np.array(list(probs.values()), dtype=np.float64)
    vals = np.clip(vals, 1e-12, 1.0)
    ent = float(-(vals * np.log(vals)).sum())
    return ent / float(np.log(len(vals))) if len(vals) > 1 else 0.0


def initial_budget(profile, probe, first_policy, args):
    if not first_policy.get("fast_candidate"):
        return 32, "profile_conservative"
    if not probe.get("available"):
        return first_policy["steps"], "probe_unavailable"
    top_prob = probe.get("top_prob", 1.0)
    if profile.get("n_labels", 0) <= 2:
        if top_prob < args.binary_direct_fallback_threshold:
            return 32, "binary_low_probe_confidence"
        if top_prob < args.binary_medium_threshold:
            return 16, "binary_medium_probe_confidence"
    return first_policy["steps"], "fast_probe_confident"


def invalid_or_disagree(pred, labels, probe, profile, multi_disagreement_policy):
    if labels and pred not in labels:
        return True, "invalid_label"
    if probe.get("available") and pred and pred != probe.get("top_label"):
        if profile.get("n_labels", 0) > 2 and multi_disagreement_policy == "ignore":
            return False, "multi_choice_disagreement_ignored"
        return True, "probe_scout_disagreement"
    return False, "consistent"


@torch.no_grad()
def run_decode_with_trace(model, tokenizer, sample, policy, args, labels, trace_stride=1):
    text = base.chat_prompt(tokenizer, sample["prompt"])
    enc = tokenizer([text], add_special_tokens=False, padding=True, return_tensors="pt")
    prompt = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    x = torch.full((prompt.shape[0], prompt.shape[1] + args.gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((prompt.shape[0], args.gen_length), dtype=attention_mask.dtype, device=model.device),
        ],
        dim=-1,
    )
    prompt_index = x != MASK_ID
    assert args.gen_length % args.block_length == 0
    num_blocks = args.gen_length // args.block_length
    assert policy["steps"] % num_blocks == 0
    steps_per_block = policy["steps"] // num_blocks

    trace = []
    forward_calls = 0
    t0 = time.time()
    for nb in range(num_blocks):
        block_start = prompt.shape[1] + nb * args.block_length
        block_end = prompt.shape[1] + (nb + 1) * args.block_length
        block_mask = x[:, block_start:block_end] == MASK_ID
        budgets = token_budget(block_mask, steps_per_block, policy["schedule"])
        for i in range(steps_per_block):
            mask_index = x == MASK_ID
            logits = model_logits(model, x, attention_mask, prompt_index, args.cfg)
            forward_calls += 1
            x0, conf, _ = x0_and_conf(logits, x, mask_index, prompt.shape[1], block_end, args.temperature, args.remasking)
            transfer = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            for j in range(conf.shape[0]):
                k = int(budgets[j, i].item())
                if k <= 0:
                    continue
                _, idx = torch.topk(conf[j], k=k)
                transfer[j, idx] = True
            x[transfer] = x0[transfer]
            global_step = nb * steps_per_block + i + 1
            if trace_stride > 0 and (global_step == 1 or global_step == policy["steps"] or global_step % trace_stride == 0):
                gen_ids = x[:, prompt.shape[1] :]
                output_now = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
                pred_now = parse_for_metric(sample, output_now)
                filled = float((gen_ids != MASK_ID).float().mean().item())
                trace.append(
                    {
                        "step": global_step,
                        "filled_ratio": filled,
                        "pred": pred_now,
                        "valid_label": (not labels) or pred_now in labels,
                    }
                )
    dt = time.time() - t0
    output = tokenizer.batch_decode(x[:, prompt.shape[1] :], skip_special_tokens=True)[0]
    return output, dt, forward_calls, trace


def decode_once(model, tokenizer, sample, policy, args, labels, collect_trace):
    if collect_trace:
        return run_decode_with_trace(
            model,
            tokenizer,
            sample,
            policy,
            args,
            labels,
            trace_stride=args.trace_stride,
        )
    output, dt, calls = run_decode(model, tokenizer, sample, policy, args)
    return output, dt, calls, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="optional PEFT adapter path")
    ap.add_argument("--tasks", default="winogrande,commonsenseqa")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--binary-direct-fallback-threshold", type=float, default=0.70)
    ap.add_argument("--binary-medium-threshold", type=float, default=0.80)
    ap.add_argument("--multi-disagreement-policy", choices=["fallback", "ignore"], default="fallback")
    ap.add_argument("--use-intermediate-on-disagreement", action="store_true")
    ap.add_argument("--collect-trace", action="store_true")
    ap.add_argument("--trace-stride", type=int, default=1)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = base.load_llada(args.model, args.adapter)
    torch.cuda.reset_peak_memory_stats()

    rows = []
    summary = {}
    for task in [t for t in args.tasks.split(",") if t]:
        raw_samples = domain.build_samples(task, args.limit, args.seed)
        correct = 0
        total_time = 0.0
        total_calls = 0
        route_counts = {}
        for raw in raw_samples:
            profile = task_profile(raw, tokenizer)
            labels = infer_label_space(raw)
            first_policy = choose_policy(profile)
            sample = apply_prompt(raw, first_policy["prompt"])
            t0 = time.time()
            probe = probe_label_distribution(model, tokenizer, sample, labels)
            calls = 1 if probe.get("available") else 0
            route_steps = []
            traces = []

            budget, budget_reason = initial_budget(profile, probe, first_policy, args)
            policy = budget_policy(first_policy, budget)
            output, dt_decode, decode_calls, trace = decode_once(model, tokenizer, sample, policy, args, labels, args.collect_trace)
            pred = parse_for_metric(sample, output)
            calls += decode_calls
            route_steps.append({"steps": budget, "reason": budget_reason, "pred": pred})
            if trace:
                traces.append({"steps": budget, "trace": trace})

            risky, risk_reason = invalid_or_disagree(pred, labels, probe, profile, args.multi_disagreement_policy)
            if risky and budget < 32:
                if args.use_intermediate_on_disagreement and budget < 16:
                    policy16 = budget_policy(first_policy, 16)
                    out16, dt16, calls16, trace16 = decode_once(model, tokenizer, sample, policy16, args, labels, args.collect_trace)
                    pred16 = parse_for_metric(sample, out16)
                    calls += calls16
                    output = out16
                    pred = pred16
                    route_steps.append({"steps": 16, "reason": risk_reason, "pred": pred16})
                    if trace16:
                        traces.append({"steps": 16, "trace": trace16})
                    risky16, risk16_reason = invalid_or_disagree(
                        pred16,
                        labels,
                        probe,
                        profile,
                        args.multi_disagreement_policy,
                    )
                    risk_reason = risk16_reason
                    risky = risky16
                if risky:
                    policy32 = budget_policy(first_policy, 32)
                    out32, dt32, calls32, trace32 = decode_once(model, tokenizer, sample, policy32, args, labels, args.collect_trace)
                    pred32 = parse_for_metric(sample, out32)
                    calls += calls32
                    output = out32
                    pred = pred32
                    route_steps.append({"steps": 32, "reason": risk_reason, "pred": pred32})
                    if trace32:
                        traces.append({"steps": 32, "trace": trace32})

            dt = time.time() - t0
            ok = pred == raw["gold"]
            correct += int(ok)
            total_time += dt
            total_calls += calls
            route = "->".join(str(s["steps"]) for s in route_steps)
            route_counts[route] = route_counts.get(route, 0) + 1
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
                "seconds": dt,
                "forward_calls": calls,
                "traces": traces,
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
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    payload = {"model": args.model, "args": vars(args), "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
