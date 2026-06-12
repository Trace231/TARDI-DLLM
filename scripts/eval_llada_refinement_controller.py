import argparse
import csv
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


def clip01(value):
    return max(0.0, min(1.0, float(value)))


def resolve_prior_path(text):
    if text:
        path = Path(text)
        return path if path.exists() else None
    return None


def load_task_priors(path, budgets):
    if path is None:
        return {}
    by_task = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                task = row["task"]
                step = int(row["step"])
                acc = float(row["accuracy"])
            except (KeyError, TypeError, ValueError):
                continue
            by_task.setdefault(task, {})[step] = acc
    priors = {}
    for task, curve in by_task.items():
        if not curve:
            continue
        min_budget = min(budgets)
        max_budget = max(budgets)
        base_step = min(curve, key=lambda s: abs(s - min_budget))
        full_step = min(curve, key=lambda s: abs(s - max_budget))
        base_acc = curve[base_step]
        full_acc = curve[full_step]
        best_acc = max(curve.values())
        max_gain = max(1e-6, best_acc - base_acc)
        budget_gain = {}
        for budget in budgets:
            near = min(curve, key=lambda s: abs(s - budget))
            budget_gain[budget] = clip01((curve[near] - base_acc) / max_gain)
        positive_span = max(0.0, full_acc - base_acc)
        volatility = 0.0
        ordered = sorted(curve)
        if len(ordered) > 1:
            volatility = float(np.mean([abs(curve[b] - curve[a]) for a, b in zip(ordered, ordered[1:])]))
        saturation_step = max_budget
        for step in ordered:
            if curve[step] >= best_acc - 0.01:
                saturation_step = step
                break
        priors[task] = {
            "source": str(path),
            "base_step": base_step,
            "full_step": full_step,
            "base_acc": base_acc,
            "full_acc": full_acc,
            "best_acc": best_acc,
            "budget_gain": {str(k): v for k, v in budget_gain.items()},
            "positive_span": positive_span,
            "sensitivity": clip01(positive_span / 0.20 + volatility / 0.08),
            "volatility": volatility,
            "saturation_step": saturation_step,
        }
    return priors


def fallback_prior(task, profile, budgets):
    n_labels = profile.get("n_labels", 0)
    if profile.get("metric") not in {"letter", "bool", "decision"}:
        sensitivity = 1.0
        saturation = max(budgets)
    elif profile.get("metric") == "decision":
        sensitivity = 0.9
        saturation = max(budgets)
    elif n_labels <= 2:
        sensitivity = 0.45
        saturation = 16 if 16 in budgets else max(budgets)
    elif n_labels >= 8:
        sensitivity = 0.75
        saturation = 24 if 24 in budgets else max(budgets)
    else:
        sensitivity = 0.55
        saturation = 16 if 16 in budgets else max(budgets)
    budget_gain = {}
    lo, hi = min(budgets), max(budgets)
    for budget in budgets:
        budget_gain[str(budget)] = clip01((budget - lo) / max(1, hi - lo))
    return {
        "source": "fallback_shape_prior",
        "base_step": min(budgets),
        "full_step": max(budgets),
        "base_acc": None,
        "full_acc": None,
        "best_acc": None,
        "budget_gain": budget_gain,
        "positive_span": None,
        "sensitivity": sensitivity,
        "volatility": 0.0,
        "saturation_step": saturation,
    }


def structured_budget_decision(task, profile, features, score, budgets, prior, args):
    if not args.structured_routing:
        target, reason = budget_from_score(score, budgets, args)
        return target, reason, {"mode": "threshold", "score": score}
    if features.get("invalid_or_empty", 0.0) >= 1.0:
        return max(budgets), "invalid_or_empty_full", {
            "mode": "risk_value",
            "score": score,
            "forced": "invalid_or_empty",
        }
    min_budget = min(budgets)
    max_budget = max(budgets)
    sensitivity = float(prior.get("sensitivity", 0.5))
    instability = 0.5 * float(features.get("flip_instability", 0.0)) + 0.5 * float(features.get("late_first_final", 0.0))
    uncertainty = 0.5 * float(features.get("probe_entropy", 0.0)) + 0.5 * float(features.get("margin_deficit", 0.0))
    disagreement = float(features.get("probe_scout_disagree", 0.0))
    risk_mass = clip01(score * (0.65 + 0.55 * sensitivity + 0.25 * instability + 0.20 * uncertainty + 0.15 * disagreement))
    objective = {}
    gain_curve = prior.get("budget_gain", {})
    saturation = float(prior.get("saturation_step", max_budget) or max_budget)
    for budget in budgets:
        normalized_cost = (budget - min_budget) / max(1, max_budget - min_budget)
        curve_gain = float(gain_curve.get(str(budget), 0.0))
        saturation_bonus = 0.0
        if budget <= saturation:
            saturation_bonus = args.saturation_bonus * (1.0 - abs(budget - saturation) / max(1.0, saturation - min_budget + 1.0))
        expected_gain = risk_mass * (args.prior_gain_weight * curve_gain + args.instability_gain_weight * instability * normalized_cost)
        residual = risk_mass - expected_gain
        objective[str(budget)] = {
            "curve_gain": curve_gain,
            "expected_gain": expected_gain,
            "residual_risk": residual,
            "cost": args.compute_cost * normalized_cost,
            "saturation_bonus": saturation_bonus,
            "value": residual + args.compute_cost * normalized_cost - saturation_bonus,
        }
    target = min(budgets, key=lambda b: (objective[str(b)]["value"], b))
    if args.prefer_saturation_budget and target == max_budget and score < args.structured_force_32_score and saturation < max_budget:
        saturation_budget = min((b for b in budgets if b >= saturation), default=max_budget)
        if objective[str(saturation_budget)]["value"] <= objective[str(max_budget)]["value"] + args.saturation_tolerance:
            target = saturation_budget
    if profile.get("n_labels", 0) <= 2 and score >= args.binary_min_16_score and target < 16 and 16 in budgets:
        target = 16
        reason = "binary_medium_risk_min_16"
    elif score >= args.structured_force_32_score and target < max_budget:
        target = max_budget
        reason = "high_residual_risk_full"
    elif score >= args.structured_min_24_score and 24 in budgets and target < 24:
        target = 24
        reason = "high_risk_min_24"
    elif disagreement and probe_confident_enough(profile, score, args) and target < 16 and 16 in budgets:
        target = 16
        reason = "probe_scout_disagreement_min_16"
    else:
        reason = f"risk_value_budget_{target}"
    return target, reason, {
        "mode": "risk_value",
        "task": task,
        "score": score,
        "risk_mass": risk_mass,
        "sensitivity": sensitivity,
        "instability": instability,
        "uncertainty": uncertainty,
        "objective": objective,
        "selected_budget": target,
        "prior": prior,
    }


def probe_confident_enough(profile, score, args):
    if profile.get("n_labels", 0) <= 2:
        return score >= args.risk_t16
    return score >= args.risk_t24


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


def scout_stats(history, args, last_conf=None, prompt_len=0, total_steps=None):
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
    out["observed_steps"] = int(total_steps or args.scout_steps)
    return out


def absolute_history(history, offset):
    out = []
    for item in history:
        row = dict(item)
        row["local_step"] = int(row.get("local_step", 0)) + int(offset)
        out.append(row)
    return out


def label_token_ids(tokenizer, labels):
    out = set()
    for label in labels:
        for text in (label, " " + label, "\n" + label):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(int(ids[0]))
    return out


def token_text_flags(tokenizer, token_ids):
    flags = []
    for token_id in token_ids:
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False).lower()
        flags.append(
            {
                "final_marker": any(x in piece for x in ["final", "answer", "答案", "答"]),
                "separator": any(x in piece for x in [":", "：", "<answer>", "</answer>"]),
            }
        )
    return flags


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


def remask_structured(x, tokenizer, prompt_len, last_conf, labels, fraction, min_tokens, args):
    gen_conf = last_conf[:, prompt_len:].clone()
    gen_tokens = x[:, prompt_len:]
    valid = gen_tokens != MASK_ID
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return 0, {"policy": "structured", "reason": "no_valid_tokens"}
    k = max(min_tokens, int(round(valid_count * fraction)))
    k = min(k, valid_count)
    label_ids = label_token_ids(tokenizer, labels)
    token_ids = gen_tokens[0].detach().cpu().tolist()
    flags = token_text_flags(tokenizer, token_ids)
    finite_conf = torch.where(valid, gen_conf, torch.full_like(gen_conf, 1.0))
    rank_score = 1.0 - finite_conf[0]
    label_bonus = torch.zeros_like(rank_score)
    marker_bonus = torch.zeros_like(rank_score)
    for idx, token_id in enumerate(token_ids):
        if not bool(valid[0, idx].item()):
            continue
        if int(token_id) in label_ids:
            label_bonus[idx] = args.label_remask_bonus
            left = max(0, idx - args.label_remask_window)
            right = min(label_bonus.numel(), idx + args.label_remask_window + 1)
            label_bonus[left:right] = torch.maximum(label_bonus[left:right], torch.full_like(label_bonus[left:right], args.label_context_bonus))
        if flags[idx]["final_marker"] or flags[idx]["separator"]:
            marker_bonus[idx] = args.answer_marker_bonus
    rank_score = torch.where(valid[0], rank_score + label_bonus + marker_bonus, torch.full_like(rank_score, -1.0))
    _, idx = torch.topk(rank_score, k=k)
    gen_tokens[0, idx] = MASK_ID
    selected = idx.detach().cpu().tolist()
    return k, {
        "policy": "structured",
        "fraction": fraction,
        "selected": selected,
        "label_token_hits": int(sum(1 for i in selected if int(token_ids[i]) in label_ids)),
        "marker_hits": int(sum(1 for i in selected if flags[i]["final_marker"] or flags[i]["separator"])),
        "valid_count": valid_count,
    }


def remask_answer_consistency(x, tokenizer, prompt_len, last_conf, labels, pred, probe, fraction, min_tokens, args):
    gen_conf = last_conf[:, prompt_len:].clone()
    gen_tokens = x[:, prompt_len:]
    valid = gen_tokens != MASK_ID
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return 0, {"policy": "answer_consistency", "reason": "no_valid_tokens"}
    k = max(min_tokens, int(round(valid_count * fraction)))
    k = min(k, valid_count)
    label_ids = label_token_ids(tokenizer, labels)
    pred_ids = set(label_token_ids(tokenizer, [pred])) if pred else set()
    probe_top = probe.get("top_label") if probe.get("available") else None
    probe_conf = float(probe.get("top_prob", 0.0)) if probe.get("available") else 0.0
    probe_margin = float(probe.get("margin", 0.0)) if probe.get("available") else 0.0
    agrees = bool(pred and probe_top == pred and probe_conf >= args.answer_protect_confidence and probe_margin >= args.answer_protect_margin)
    disagrees = bool(pred and probe_top and probe_top != pred and probe_conf >= args.answer_repair_confidence)
    token_ids = gen_tokens[0].detach().cpu().tolist()
    flags = token_text_flags(tokenizer, token_ids)
    finite_conf = torch.where(valid, gen_conf, torch.full_like(gen_conf, 1.0))
    rank_score = 1.0 - finite_conf[0]
    adjustment = torch.zeros_like(rank_score)
    for idx, token_id in enumerate(token_ids):
        if not bool(valid[0, idx].item()):
            continue
        is_label = int(token_id) in label_ids
        is_pred = int(token_id) in pred_ids
        is_marker = flags[idx]["final_marker"] or flags[idx]["separator"]
        if agrees and (is_pred or is_marker):
            adjustment[idx] -= args.answer_protect_penalty
        elif disagrees and (is_label or is_marker):
            adjustment[idx] += args.answer_repair_bonus
        elif is_label and not agrees:
            adjustment[idx] += args.answer_label_bonus
    rank_score = torch.where(valid[0], rank_score + adjustment, torch.full_like(rank_score, -1.0))
    selectable = int((rank_score > -0.5).sum().item())
    k = min(k, max(1, selectable))
    _, idx = torch.topk(rank_score, k=k)
    gen_tokens[0, idx] = MASK_ID
    selected = idx.detach().cpu().tolist()
    return k, {
        "policy": "answer_consistency",
        "fraction": fraction,
        "selected": selected,
        "probe_top": probe_top,
        "probe_confidence": probe_conf,
        "probe_margin": probe_margin,
        "agrees": agrees,
        "disagrees": disagrees,
        "label_token_hits": int(sum(1 for i in selected if int(token_ids[i]) in label_ids)),
        "marker_hits": int(sum(1 for i in selected if flags[i]["final_marker"] or flags[i]["separator"])),
        "valid_count": valid_count,
    }


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


def online_risk_features(profile, probe, pred, labels, stats, args, observed_steps):
    features = risk_features(profile, probe, pred, labels, stats, args)
    first_final = stats.get("first_final_step")
    features["late_first_final"] = 1.0 if first_final is None else clip01(first_final / max(1, observed_steps))
    return features


def online_continue_decision(profile, probe, pred, labels, features, score, current_budget, budgets, args):
    if labels and pred not in labels:
        target = max(budgets) if current_budget >= args.online_invalid_full_after else next_budget(current_budget, budgets)
        return True, "online_invalid_label", target
    if not pred:
        target = max(budgets) if current_budget >= args.online_invalid_full_after else next_budget(current_budget, budgets)
        return True, "online_empty_prediction", target

    next_b = next_budget(current_budget, budgets)
    if next_b <= current_budget:
        return False, "online_max_budget", current_budget
    if current_budget < args.online_min_budget:
        return True, "online_minimum_scout_budget", next_b
    if (
        profile.get("n_labels", 0) <= 2
        and current_budget < args.online_binary_hesitation_until
        and score >= args.online_binary_hesitation_score
    ):
        return True, "online_binary_hesitation", next_b

    probe_top = probe.get("top_prob", 0.0) if probe.get("available") else 0.0
    probe_margin = probe.get("margin", 1.0) if probe.get("available") else 1.0
    strong_disagreement = bool(
        probe.get("available")
        and pred != probe.get("top_label")
        and probe_top >= args.online_disagree_confidence
        and probe_margin >= args.online_disagree_margin
    )
    unstable = (
        features.get("flip_instability", 0.0) >= args.online_flip_threshold
        or (
            features.get("late_first_final", 0.0) >= args.online_late_threshold
            and score >= args.online_late_score
            and current_budget < args.online_late_ignore_after
        )
    )
    weak_conf = (
        features.get("low_fill_confidence", 0.0) >= args.online_low_fill_threshold
        or features.get("margin_deficit", 0.0) >= args.online_margin_deficit_threshold
    )
    high_uncertainty = features.get("probe_entropy", 0.0) >= args.online_entropy_threshold

    accept_score = args.online_accept_score
    if current_budget <= 4:
        accept_score *= 0.75
    elif current_budget >= 16:
        accept_score *= 1.15

    if score <= accept_score and not strong_disagreement and not unstable and not weak_conf:
        return False, "online_accept_low_risk", current_budget
    if strong_disagreement:
        return True, "online_probe_scout_disagreement", next_b
    if unstable:
        return True, "online_unstable_trajectory", next_b
    if weak_conf:
        return True, "online_weak_decision_confidence", next_b
    if high_uncertainty and score >= args.online_uncertain_score:
        return True, "online_probe_uncertainty", next_b
    if score >= args.online_continue_score:
        return True, "online_score_continue", next_b
    return False, "online_accept_marginal_gain_low", current_budget


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


def write_payload(path, args, prior_path, summary, rows, complete):
    payload = {
        "method": "selective_remask_refinement_controller",
        "model": args.model,
        "args": vars(args),
        "task_prior_csv": str(prior_path) if prior_path else None,
        "complete": complete,
        "summary": summary,
        "rows": rows,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


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
    ap.add_argument("--max-refinements", type=int, default=7)
    ap.add_argument("--online-control", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--online-step", type=int, default=4)
    ap.add_argument("--online-min-budget", type=int, default=8)
    ap.add_argument("--online-min-rerun", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--online-accept-score", type=float, default=0.24)
    ap.add_argument("--online-continue-score", type=float, default=0.42)
    ap.add_argument("--online-uncertain-score", type=float, default=0.30)
    ap.add_argument("--online-late-score", type=float, default=0.32)
    ap.add_argument("--online-late-ignore-after", type=int, default=16)
    ap.add_argument("--online-invalid-full-after", type=int, default=12)
    ap.add_argument("--online-binary-hesitation-score", type=float, default=0.21)
    ap.add_argument("--online-binary-hesitation-until", type=int, default=16)
    ap.add_argument("--online-disagree-confidence", type=float, default=0.62)
    ap.add_argument("--online-disagree-margin", type=float, default=0.08)
    ap.add_argument("--online-flip-threshold", type=float, default=0.34)
    ap.add_argument("--online-late-threshold", type=float, default=0.80)
    ap.add_argument("--online-low-fill-threshold", type=float, default=0.28)
    ap.add_argument("--online-margin-deficit-threshold", type=float, default=0.50)
    ap.add_argument("--online-entropy-threshold", type=float, default=0.82)
    ap.add_argument("--structured-routing", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--task-prior-csv", default=None)
    ap.add_argument("--compute-cost", type=float, default=0.055)
    ap.add_argument("--prior-gain-weight", type=float, default=0.75)
    ap.add_argument("--instability-gain-weight", type=float, default=0.30)
    ap.add_argument("--saturation-bonus", type=float, default=0.025)
    ap.add_argument("--prefer-saturation-budget", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--saturation-tolerance", type=float, default=0.08)
    ap.add_argument("--binary-min-16-score", type=float, default=0.24)
    ap.add_argument("--structured-min-24-score", type=float, default=0.48)
    ap.add_argument("--structured-force-32-score", type=float, default=0.70)
    ap.add_argument("--structured-remask", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--answer-consistency-remask", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--label-remask-bonus", type=float, default=0.45)
    ap.add_argument("--label-context-bonus", type=float, default=0.20)
    ap.add_argument("--answer-marker-bonus", type=float, default=0.12)
    ap.add_argument("--label-remask-window", type=int, default=2)
    ap.add_argument("--answer-protect-confidence", type=float, default=0.58)
    ap.add_argument("--answer-protect-margin", type=float, default=0.04)
    ap.add_argument("--answer-protect-penalty", type=float, default=0.55)
    ap.add_argument("--answer-repair-confidence", type=float, default=0.62)
    ap.add_argument("--answer-repair-bonus", type=float, default=0.35)
    ap.add_argument("--answer-label-bonus", type=float, default=0.10)
    ap.add_argument("--compact-choice-fast", action="store_true")
    ap.add_argument("--compact-choice-max-labels", type=int, default=5)
    ap.add_argument("--compact-choice-max-prompt-tokens", type=int, default=512)
    args = ap.parse_args()

    budgets = parse_budgets(args.budgets)
    prior_path = resolve_prior_path(args.task_prior_csv)
    task_priors = load_task_priors(prior_path, budgets)
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
            task_prior = task_priors.get(task, fallback_prior(task, profile, budgets))
            sample = apply_prompt(raw, first_policy["prompt"])
            t0 = time.time()
            probe = probe_label_distribution(model, tokenizer, sample, labels)
            calls = 1 if probe.get("available") else 0
            x, attention_mask, prompt_len, prompt_index = make_state(tokenizer, sample, args, model.device)
            route_steps = []
            features = {}
            score = None
            budget_decision = {}

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
                current_budget = args.scout_steps
                cumulative_history = absolute_history(hist, 0)
                stats = scout_stats(cumulative_history, args, last_conf, prompt_len, current_budget)
                features = online_risk_features(profile, probe, pred, labels, stats, args, current_budget)
                score = risk_score(features, profile)
                if args.online_control:
                    keep_going, reason, target = online_continue_decision(profile, probe, pred, labels, features, score, current_budget, budgets, args)
                    budget_decision = {
                        "mode": "online_sequential",
                        "observed_budget": current_budget,
                        "selected_next_budget": target,
                        "keep_going": keep_going,
                        "reason": reason,
                        "score": score,
                        "features": features,
                    }
                else:
                    target, reason, budget_decision = structured_budget_decision(task, profile, features, score, budgets, task_prior, args)
                    keep_going = target > current_budget
                route_steps.append(
                    {
                        "budget": current_budget,
                        "mode": "scout",
                        "reason": "scout",
                        "pred": pred,
                        "risk_score": score,
                        "target_budget": target,
                        "budget_decision": budget_decision,
                    }
                )

                refinements = 0
                while keep_going and target > current_budget and refinements < args.max_refinements:
                    extra_steps = target - current_budget
                    frac = refinement_fraction(score, target, args)
                    if args.online_control and args.online_min_rerun and current_budget < args.online_min_budget and target <= args.online_min_budget:
                        remasked = 0
                        remask_plan = {"policy": "fresh_min_budget_rerun", "from_budget": current_budget, "to_budget": target}
                        x, attention_mask, prompt_len, prompt_index = make_state(tokenizer, sample, args, model.device)
                        policy = budget_policy(first_policy, target)
                        x, output, pred, c, last_conf, hist2 = fill_masks(
                            model,
                            tokenizer,
                            sample,
                            x,
                            attention_mask,
                            prompt_len,
                            prompt_index,
                            target,
                            policy["schedule"],
                            args,
                            labels,
                            {1, 2, 4, target},
                        )
                        calls += c
                        hist2_abs = absolute_history(hist2, 0)
                    elif args.answer_consistency_remask:
                        remasked, remask_plan = remask_answer_consistency(
                            x, tokenizer, prompt_len, last_conf, labels, pred, probe, frac, args.remask_min_tokens, args
                        )
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
                        hist2_abs = absolute_history(hist2, current_budget)
                    elif args.structured_remask:
                        remasked, remask_plan = remask_structured(x, tokenizer, prompt_len, last_conf, labels, frac, args.remask_min_tokens, args)
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
                        hist2_abs = absolute_history(hist2, current_budget)
                    else:
                        remasked = remask_low_confidence(x, prompt_len, last_conf, frac, args.remask_min_tokens)
                        remask_plan = {"policy": "low_confidence", "fraction": frac}
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
                        hist2_abs = absolute_history(hist2, current_budget)
                    current_budget = target
                    cumulative_history.extend(hist2_abs)
                    stats = scout_stats(cumulative_history, args, last_conf, prompt_len, current_budget)
                    features = online_risk_features(profile, probe, pred, labels, stats, args, current_budget)
                    score = risk_score(features, profile)
                    route_steps.append(
                        {
                            "budget": current_budget,
                            "mode": "refine",
                            "reason": reason,
                            "pred": pred,
                            "risk_score": score,
                            "remasked_tokens": remasked,
                            "remask_plan": remask_plan,
                            "features": features,
                        }
                    )
                    if args.online_control:
                        keep_going, reason, target = online_continue_decision(profile, probe, pred, labels, features, score, current_budget, budgets, args)
                    else:
                        risky, risk_reason, severity = post_risky(pred, labels, probe, profile, args)
                        if not risky:
                            keep_going = False
                            break
                        target = 32 if severity >= 2 else next_budget(current_budget, budgets)
                        reason = risk_reason
                        keep_going = target > current_budget
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
                "budget_decision": budget_decision,
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
        write_payload(args.out, args, prior_path, summary, rows, complete=False)
    write_payload(args.out, args, prior_path, summary, rows, complete=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
