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


def norm_entropy(probs):
    vals = np.array(list(probs.values()), dtype=np.float64)
    if len(vals) <= 1:
        return 0.0
    vals = np.clip(vals, 1e-12, 1.0)
    return float(-(vals * np.log(vals)).sum() / np.log(len(vals)))


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def budget_policy(first_policy, steps):
    policy = dict(first_policy)
    policy["steps"] = int(steps)
    policy["fast_candidate"] = steps < 32
    if steps >= 32:
        policy["schedule"] = "uniform"
        policy["fast_candidate"] = False
    policy["reason"] = f"risk_controller_{steps}_step"
    return policy


def parse_budgets(text):
    budgets = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not budgets:
        raise ValueError("empty budget set")
    if 32 not in budgets:
        budgets.append(32)
    return budgets


def next_budget(current, budgets):
    for b in budgets:
        if b > current:
            return b
    return current


@torch.no_grad()
def decode_with_scout_stats(model, tokenizer, sample, policy, args, labels, checkpoints):
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

    history = []
    fill_conf = []
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
                vals, idx = torch.topk(conf[j], k=k)
                transfer[j, idx] = True
                finite_vals = vals[torch.isfinite(vals)]
                if finite_vals.numel() > 0:
                    fill_conf.extend(float(v.item()) for v in finite_vals.detach().float().cpu())
            x[transfer] = x0[transfer]
            global_step = nb * steps_per_block + i + 1
            if global_step in checkpoints or global_step == policy["steps"]:
                gen_ids = x[:, prompt.shape[1] :]
                output_now = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
                pred_now = parse_for_metric(sample, output_now)
                history.append(
                    {
                        "step": global_step,
                        "pred": pred_now,
                        "valid": (not labels) or pred_now in labels,
                        "filled_ratio": float((gen_ids != MASK_ID).float().mean().item()),
                    }
                )
    dt = time.time() - t0
    output = tokenizer.batch_decode(x[:, prompt.shape[1] :], skip_special_tokens=True)[0]
    pred = parse_for_metric(sample, output)
    preds = [h["pred"] for h in history if h.get("pred")]
    flips = sum(1 for a, b in zip(preds, preds[1:]) if a != b)
    final_pred = preds[-1] if preds else pred
    first_final_step = None
    for h in history:
        if h.get("pred") == final_pred and h.get("pred"):
            first_final_step = h["step"]
            break
    valid_seen = sum(1 for h in history if h.get("valid") and h.get("pred"))
    stats = {
        "history": history,
        "flip_count": flips,
        "first_final_step": first_final_step,
        "valid_seen": valid_seen,
        "mean_fill_confidence": float(np.mean(fill_conf)) if fill_conf else None,
        "min_fill_confidence": float(np.min(fill_conf)) if fill_conf else None,
    }
    return output, pred, dt, forward_calls, stats


def fast_decode(model, tokenizer, sample, first_policy, steps, args, labels):
    policy = budget_policy(first_policy, steps)
    return decode_with_scout_stats(
        model,
        tokenizer,
        sample,
        policy,
        args,
        labels,
        checkpoints={steps},
    )


def direct_full_gate(profile, probe, first_policy, args):
    if not first_policy.get("fast_candidate"):
        return True, "profile_conservative"
    if not probe.get("available"):
        return False, "probe_unavailable"
    top_prob = probe.get("top_prob", 1.0)
    ent = norm_entropy(probe.get("probs", {}))
    n_labels = profile.get("n_labels", 0)
    if n_labels <= 2 and top_prob < args.binary_direct_full_threshold:
        return True, "binary_probe_direct_full"
    if n_labels >= 5 and (top_prob < args.multi_direct_full_threshold or ent > args.multi_direct_full_entropy):
        return True, "multi_probe_direct_full"
    return False, "probe_allows_scout"


def risk_features(profile, probe, scout_pred, labels, scout_stats, args):
    top_prob = probe.get("top_prob", 1.0) if probe.get("available") else 1.0
    margin = probe.get("margin", 1.0) if probe.get("available") else 1.0
    ent = norm_entropy(probe.get("probs", {})) if probe.get("available") else 0.0
    n_labels = profile.get("n_labels", 0)
    disagree = bool(probe.get("available") and scout_pred and scout_pred != probe.get("top_label"))
    invalid = bool(labels and scout_pred not in labels)
    empty = not bool(scout_pred)
    flip_norm = clip01(scout_stats.get("flip_count", 0) / max(1, len(scout_stats.get("history", [])) - 1))
    first_final = scout_stats.get("first_final_step")
    late_final = 1.0 if first_final is None else clip01(first_final / max(1, args.scout_steps))
    mean_conf = scout_stats.get("mean_fill_confidence")
    low_fill_conf = 0.5 if mean_conf is None else clip01((args.fill_confidence_target - mean_conf) / args.fill_confidence_target)
    margin_deficit = clip01((args.margin_target - margin) / args.margin_target)
    label_complexity = clip01((n_labels - 2) / 6) if n_labels else 0.0
    prompt_complexity = clip01((profile.get("prompt_tokens", 0) - 256) / 512)
    return {
        "probe_uncertainty": 1.0 - top_prob,
        "probe_entropy": ent,
        "margin_deficit": margin_deficit,
        "probe_scout_disagree": float(disagree),
        "invalid_or_empty": float(invalid or empty),
        "flip_instability": flip_norm,
        "late_first_final": late_final,
        "low_fill_confidence": low_fill_conf,
        "label_complexity": label_complexity,
        "prompt_complexity": prompt_complexity,
    }


def risk_score(features, profile):
    if profile.get("n_labels", 0) <= 2:
        weights = {
            "probe_uncertainty": 0.22,
            "probe_entropy": 0.10,
            "margin_deficit": 0.14,
            "probe_scout_disagree": 0.20,
            "invalid_or_empty": 0.24,
            "flip_instability": 0.08,
            "late_first_final": 0.08,
            "low_fill_confidence": 0.05,
            "prompt_complexity": 0.04,
        }
    else:
        weights = {
            "probe_uncertainty": 0.16,
            "probe_entropy": 0.18,
            "margin_deficit": 0.08,
            "probe_scout_disagree": 0.06,
            "invalid_or_empty": 0.28,
            "flip_instability": 0.08,
            "late_first_final": 0.06,
            "low_fill_confidence": 0.04,
            "label_complexity": 0.10,
            "prompt_complexity": 0.04,
        }
    return clip01(sum(weights.get(k, 0.0) * float(v) for k, v in features.items()))


def budget_from_score(score, budgets, args):
    if score < args.risk_t16:
        return min(budgets), "risk_accept_scout"
    if score < args.risk_t24:
        return 16 if 16 in budgets else next_budget(min(budgets), budgets), "risk_medium_16"
    if score < args.risk_t32:
        return 24 if 24 in budgets else next_budget(16, budgets), "risk_high_24"
    return 32, "risk_full_32"


def post_decode_risk(pred, labels, probe, profile, args):
    if labels and pred not in labels:
        return True, "post_invalid_label", 2
    if not pred:
        return True, "post_empty_prediction", 2
    if not probe.get("available"):
        return False, "post_no_probe", 0
    disagree = pred != probe.get("top_label")
    top_prob = probe.get("top_prob", 1.0)
    if profile.get("n_labels", 0) <= 2 and disagree and top_prob >= args.binary_post_disagree_confidence:
        return True, "post_binary_probe_disagreement", 1
    if (
        profile.get("n_labels", 0) > 2
        and args.multi_disagreement_policy == "fallback"
        and disagree
        and top_prob >= args.multi_post_disagree_confidence
    ):
        return True, "post_multi_probe_disagreement", 1
    return False, "post_accepted", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
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
    ap.add_argument("--max-escalations", type=int, default=2)
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
        direct_full_count = 0
        for raw in raw_samples:
            profile = task_profile(raw, tokenizer)
            labels = infer_label_space(raw)
            first_policy = choose_policy(profile)
            sample = apply_prompt(raw, first_policy["prompt"])
            t0 = time.time()
            probe = probe_label_distribution(model, tokenizer, sample, labels)
            calls = 1 if probe.get("available") else 0
            route_steps = []
            scout_stats = {}
            features = {}
            score = None

            direct_full, gate_reason = direct_full_gate(profile, probe, first_policy, args)
            if direct_full:
                output, pred, _, decode_calls, stats = fast_decode(model, tokenizer, sample, first_policy, 32, args, labels)
                calls += decode_calls
                direct_full_count += 1
                route_steps.append({"steps": 32, "reason": gate_reason, "pred": pred})
                scout_stats = stats
            else:
                scout_policy = budget_policy(first_policy, args.scout_steps)
                output, pred, _, scout_calls, scout_stats = decode_with_scout_stats(
                    model,
                    tokenizer,
                    sample,
                    scout_policy,
                    args,
                    labels,
                    checkpoints={1, 2, 4, args.scout_steps},
                )
                calls += scout_calls
                features = risk_features(profile, probe, pred, labels, scout_stats, args)
                score = risk_score(features, profile)
                target_budget, budget_reason = budget_from_score(score, budgets, args)
                route_steps.append(
                    {
                        "steps": args.scout_steps,
                        "reason": "scout",
                        "pred": pred,
                        "risk_score": score,
                        "target_budget": target_budget,
                    }
                )
                current_budget = args.scout_steps
                escalations = 0
                if target_budget > args.scout_steps:
                    current_budget = target_budget
                    output, pred, _, decode_calls, stats = fast_decode(
                        model, tokenizer, sample, first_policy, current_budget, args, labels
                    )
                    calls += decode_calls
                    route_steps.append({"steps": current_budget, "reason": budget_reason, "pred": pred})
                    scout_stats["selected_budget_stats"] = stats

                while current_budget < max(budgets) and escalations < args.max_escalations:
                    risky, risk_reason, severity = post_decode_risk(pred, labels, probe, profile, args)
                    if not risky:
                        break
                    new_budget = max(budgets) if severity >= 2 else next_budget(current_budget, budgets)
                    if new_budget == current_budget:
                        break
                    current_budget = new_budget
                    output, pred, _, decode_calls, stats = fast_decode(
                        model, tokenizer, sample, first_policy, current_budget, args, labels
                    )
                    calls += decode_calls
                    route_steps.append({"steps": current_budget, "reason": risk_reason, "pred": pred})
                    scout_stats["escalated_budget_stats"] = stats
                    escalations += 1

            dt = time.time() - t0
            ok = pred == raw["gold"]
            correct += int(ok)
            total_time += dt
            total_calls += calls
            route = "->".join(str(x["steps"]) for x in route_steps)
            route_counts[route] = route_counts.get(route, 0) + 1
            final_budget = route_steps[-1]["steps"]
            final_budget_counts[str(final_budget)] = final_budget_counts.get(str(final_budget), 0) + 1
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
                "scout_stats": scout_stats,
                "first_policy": first_policy,
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
            "final_budget_rates": {k: v / n for k, v in sorted(final_budget_counts.items())},
            "direct_full_rate": direct_full_count / n,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    payload = {
        "method": "trajectory_risk_controller",
        "model": args.model,
        "args": vars(args),
        "summary": summary,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
