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
    task_profile,
)
from eval_llada_sampler_variants import MASK_ID, model_logits, token_budget, x0_and_conf
from eval_llada_risk_controller import norm_entropy, risk_features, risk_score, budget_from_score


def budget_policy(first_policy, steps):
    policy = dict(first_policy)
    policy["steps"] = int(steps)
    if steps >= 32:
        policy["schedule"] = "uniform"
    return policy


def parse_budgets(text):
    budgets = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if 32 not in budgets:
        budgets.append(32)
    return budgets


def next_budget(current, budgets):
    for b in budgets:
        if b > current:
            return b
    return current


def make_state(tokenizer, sample, args, device):
    text = base.chat_prompt(tokenizer, sample["prompt"])
    enc = tokenizer([text], add_special_tokens=False, padding=True, return_tensors="pt")
    prompt = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    x = torch.full((prompt.shape[0], prompt.shape[1] + args.gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, : prompt.shape[1]] = prompt.clone()
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((prompt.shape[0], args.gen_length), dtype=attention_mask.dtype, device=device),
        ],
        dim=-1,
    )
    return x, attention_mask, prompt.shape[1], x != MASK_ID


@torch.no_grad()
def fill_masks(model, tokenizer, sample, x, attention_mask, prompt_len, prompt_index, steps, schedule, args, labels, checkpoints):
    assert args.gen_length % args.block_length == 0
    num_blocks = args.gen_length // args.block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    last_conf = torch.zeros_like(x, dtype=torch.float32)
    history = []
    forward_calls = 0
    for nb in range(num_blocks):
        block_start = prompt_len + nb * args.block_length
        block_end = prompt_len + (nb + 1) * args.block_length
        block_mask = x[:, block_start:block_end] == MASK_ID
        budgets = token_budget(block_mask, steps_per_block, schedule)
        for i in range(steps_per_block):
            mask_index = x == MASK_ID
            if not bool(mask_index[:, prompt_len:block_end].any().item()):
                continue
            logits = model_logits(model, x, attention_mask, prompt_index, args.cfg)
            forward_calls += 1
            x0, conf, _ = x0_and_conf(logits, x, mask_index, prompt_len, block_end, args.temperature, args.remasking)
            transfer = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            for j in range(conf.shape[0]):
                k = int(budgets[j, i].item())
                if k <= 0:
                    continue
                finite = torch.isfinite(conf[j])
                k = min(k, int(finite.sum().item()))
                if k <= 0:
                    continue
                vals, idx = torch.topk(conf[j], k=k)
                transfer[j, idx] = True
                last_conf[j, idx] = vals.detach().float()
            x[transfer] = x0[transfer]
            global_step = i + 1
            if global_step in checkpoints or global_step == steps:
                gen_ids = x[:, prompt_len:]
                output_now = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
                pred_now = parse_for_metric(sample, output_now)
                history.append(
                    {
                        "local_step": global_step,
                        "pred": pred_now,
                        "valid": (not labels) or pred_now in labels,
                        "filled_ratio": float((gen_ids != MASK_ID).float().mean().item()),
                    }
                )
    output = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
    pred = parse_for_metric(sample, output)
    return x, output, pred, forward_calls, last_conf, history


def scout_stats(history, args, last_conf=None, prompt_len=0):
    preds = [h["pred"] for h in history if h.get("pred")]
    flips = sum(1 for a, b in zip(preds, preds[1:]) if a != b)
    final = preds[-1] if preds else ""
    first_final = None
    for h in history:
        if h.get("pred") == final and final:
            first_final = h["local_step"]
            break
    out = {
        "history": history,
        "flip_count": flips,
        "first_final_step": first_final,
        "valid_seen": sum(1 for h in history if h.get("valid") and h.get("pred")),
    }
    if last_conf is not None:
        vals = last_conf[:, prompt_len:]
        vals = vals[vals > 0]
        if vals.numel() > 0:
            out["mean_fill_confidence"] = float(vals.detach().float().mean().item())
            out["min_fill_confidence"] = float(vals.detach().float().min().item())
    return out


def remask_low_confidence(x, prompt_len, last_conf, fraction, min_tokens):
    gen_conf = last_conf[:, prompt_len:].clone()
    gen_tokens = x[:, prompt_len:]
    valid = gen_tokens != MASK_ID
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return 0
    k = max(min_tokens, int(round(valid_count * fraction)))
    k = min(k, valid_count)
    gen_conf = torch.where(valid, gen_conf, torch.full_like(gen_conf, float("inf")))
    _, idx = torch.topk(-gen_conf[0], k=k)
    gen_tokens[0, idx] = MASK_ID
    return k


def post_risky(pred, labels, probe, profile, args):
    if labels and pred not in labels:
        return True, "invalid_label", 2
    if not pred:
        return True, "empty_prediction", 2
    if not probe.get("available"):
        return False, "accepted", 0
    disagree = pred != probe.get("top_label")
    if profile.get("n_labels", 0) <= 2 and disagree and probe.get("top_prob", 0.0) >= args.binary_post_disagree_confidence:
        return True, "binary_probe_scout_disagreement", 1
    if (
        profile.get("n_labels", 0) > 2
        and args.multi_disagreement_policy == "fallback"
        and disagree
        and probe.get("top_prob", 0.0) >= args.multi_post_disagree_confidence
    ):
        return True, "multi_probe_scout_disagreement", 1
    return False, "accepted", 0


def direct_full(profile, probe, first_policy, args):
    if not first_policy.get("fast_candidate"):
        return True, "profile_conservative"
    if not probe.get("available"):
        return False, "probe_unavailable"
    ent = norm_entropy(probe.get("probs", {}))
    if profile.get("n_labels", 0) <= 2 and probe.get("top_prob", 1.0) < args.binary_direct_full_threshold:
        return True, "binary_direct_full"
    if profile.get("n_labels", 0) > 2 and (probe.get("top_prob", 1.0) < args.multi_direct_full_threshold or ent > args.multi_direct_full_entropy):
        return True, "multi_direct_full"
    return False, "scout_allowed"


def refinement_fraction(score, target_budget, args):
    base = args.remask_min_fraction + (args.remask_max_fraction - args.remask_min_fraction) * max(0.0, score)
    if target_budget >= 32:
        return min(args.remask_max_fraction, base + 0.10)
    if target_budget >= 24:
        return min(args.remask_max_fraction, base + 0.05)
    return base


def maybe_enable_compact_choice_scout(profile, first_policy, args):
    if not args.compact_choice_fast:
        return first_policy
    if first_policy.get("fast_candidate"):
        return first_policy
    if profile.get("metric") not in {"letter", "bool"}:
        return first_policy
    if profile.get("prompt_tokens", 10**9) > args.compact_choice_max_prompt_tokens:
        return first_policy
    n_labels = profile.get("n_labels", 0)
    if n_labels <= 0 or n_labels > args.compact_choice_max_labels:
        return first_policy
    return {
        "steps": args.scout_steps,
        "schedule": "back_loaded" if n_labels <= 2 else "middle_heavy",
        "prompt": "typed",
        "fast_candidate": True,
        "reason": "compact low-cardinality choice task uses calibrated scout",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default="winogrande,commonsenseqa")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="8,16,24,32")
    ap.add_argument("--scout-steps", type=int, default=8)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--risk-t16", type=float, default=0.24)
    ap.add_argument("--risk-t24", type=float, default=0.38)
    ap.add_argument("--risk-t32", type=float, default=0.56)
    ap.add_argument("--margin-target", type=float, default=0.18)
    ap.add_argument("--fill-confidence-target", type=float, default=0.70)
    ap.add_argument("--binary-direct-full-threshold", type=float, default=0.52)
    ap.add_argument("--multi-direct-full-threshold", type=float, default=0.36)
    ap.add_argument("--multi-direct-full-entropy", type=float, default=0.98)
    ap.add_argument("--binary-post-disagree-confidence", type=float, default=0.72)
    ap.add_argument("--multi-post-disagree-confidence", type=float, default=0.70)
    ap.add_argument("--multi-disagreement-policy", choices=["fallback", "ignore"], default="ignore")
    ap.add_argument("--remask-min-fraction", type=float, default=0.20)
    ap.add_argument("--remask-max-fraction", type=float, default=0.55)
    ap.add_argument("--remask-min-tokens", type=int, default=4)
    ap.add_argument("--max-refinements", type=int, default=3)
    ap.add_argument("--compact-choice-fast", action="store_true")
    ap.add_argument("--compact-choice-max-labels", type=int, default=5)
    ap.add_argument("--compact-choice-max-prompt-tokens", type=int, default=512)
    args = ap.parse_args()

    budgets = parse_budgets(args.budgets)
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
        budget_counts = {}
        for raw in raw_samples:
            profile = task_profile(raw, tokenizer)
            labels = infer_label_space(raw)
            first_policy = choose_policy(profile)
            first_policy = maybe_enable_compact_choice_scout(profile, first_policy, args)
            sample = apply_prompt(raw, first_policy["prompt"])
            t0 = time.time()
            probe = probe_label_distribution(model, tokenizer, sample, labels)
            calls = 1 if probe.get("available") else 0
            x, attention_mask, prompt_len, prompt_index = make_state(tokenizer, sample, args, model.device)
            route_steps = []
            features = {}
            score = None

            go_full, full_reason = direct_full(profile, probe, first_policy, args)
            if go_full:
                policy = budget_policy(first_policy, 32)
                x, output, pred, c, last_conf, hist = fill_masks(
                    model, tokenizer, sample, x, attention_mask, prompt_len, prompt_index, 32, policy["schedule"], args, labels, {8, 16, 24, 32}
                )
                calls += c
                route_steps.append({"budget": 32, "mode": "direct_full", "reason": full_reason, "pred": pred})
                stats = scout_stats(hist, args, last_conf, prompt_len)
            else:
                scout_policy = budget_policy(first_policy, args.scout_steps)
                x, output, pred, c, last_conf, hist = fill_masks(
                    model,
                    tokenizer,
                    sample,
                    x,
                    attention_mask,
                    prompt_len,
                    prompt_index,
                    args.scout_steps,
                    scout_policy["schedule"],
                    args,
                    labels,
                    {1, 2, 4, args.scout_steps},
                )
                calls += c
                stats = scout_stats(hist, args, last_conf, prompt_len)
                features = risk_features(profile, probe, pred, labels, stats, args)
                score = risk_score(features, profile)
                target, reason = budget_from_score(score, budgets, args)
                current_budget = args.scout_steps
                route_steps.append({"budget": current_budget, "mode": "scout", "reason": "scout", "pred": pred, "risk_score": score, "target_budget": target})

                refinements = 0
                while target > current_budget and refinements < args.max_refinements:
                    extra_steps = target - current_budget
                    frac = refinement_fraction(score, target, args)
                    remasked = remask_low_confidence(x, prompt_len, last_conf, frac, args.remask_min_tokens)
                    if remasked <= 0:
                        break
                    policy = budget_policy(first_policy, target)
                    x, output, pred, c, last_conf, hist2 = fill_masks(
                        model,
                        tokenizer,
                        sample,
                        x,
                        attention_mask,
                        prompt_len,
                        prompt_index,
                        extra_steps,
                        policy["schedule"],
                        args,
                        labels,
                        {extra_steps},
                    )
                    calls += c
                    current_budget = target
                    route_steps.append({"budget": current_budget, "mode": "refine", "reason": reason, "pred": pred, "remasked_tokens": remasked})
                    risky, risk_reason, severity = post_risky(pred, labels, probe, profile, args)
                    if not risky:
                        break
                    target = 32 if severity >= 2 else next_budget(current_budget, budgets)
                    reason = risk_reason
                    refinements += 1

            dt = time.time() - t0
            ok = pred == raw["gold"]
            correct += int(ok)
            total_time += dt
            total_calls += calls
            route = "->".join(str(x["budget"]) for x in route_steps)
            route_counts[route] = route_counts.get(route, 0) + 1
            final_budget = route_steps[-1]["budget"]
            budget_counts[str(final_budget)] = budget_counts.get(str(final_budget), 0) + 1
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
                "risk_features": features,
                "risk_score": score,
                "scout_stats": stats,
                "route": route,
                "route_steps": route_steps,
                "final_budget": final_budget,
                "seconds": dt,
                "forward_calls": calls,
                "meta": raw.get("meta", {}),
            }
            rows.append(row)
            print(
                json.dumps(
                    {k: row[k] for k in ["task", "id", "gold", "pred", "correct", "route", "forward_calls", "risk_score"]},
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
            "final_budget_rates": {k: v / n for k, v in sorted(budget_counts.items())},
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    payload = {"method": "selective_remask_refinement_controller", "model": args.model, "args": vars(args), "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
