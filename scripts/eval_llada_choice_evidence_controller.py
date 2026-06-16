import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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
from eval_llada_refinement_controller import (
    budget_policy,
    direct_full,
    fill_masks,
    make_state,
    maybe_enable_compact_choice_scout,
    next_budget,
    online_risk_features,
    parse_budgets,
    probe_direct_accept,
    remask_low_confidence,
    refinement_fraction,
    risk_score,
    scout_stats,
    structured_budget_decision,
)
from eval_llada_risk_controller import norm_entropy
from eval_llada_sampler_variants import MASK_ID


def parse_options(prompt):
    out = {}
    for line in str(prompt).splitlines():
        m = re.match(r"\s*([A-J])\.\s+(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def extract_question(prompt):
    lines = []
    for line in str(prompt).splitlines():
        if re.match(r"\s*[A-J]\.\s+", line):
            continue
        if line.strip().lower().startswith("answer:"):
            continue
        if "final answer" in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines).strip()[-1800:]


def winogrande_evidence_prompt(sample, label, option):
    m = re.search(r"Sentence:\s*(.*)", sample["prompt"])
    sentence = m.group(1).strip() if m else sample["prompt"]
    completed = sentence.replace("____", option)
    return (
        "Judge whether the completed sentence is coherent and uses the intended referent.\n\n"
        f"Completed sentence: {completed}\n\n"
        "Answer yes or no only.\nFinal answer: "
    )


def generic_evidence_prompt(sample, label, option):
    question = extract_question(sample["prompt"])
    return (
        "Judge whether the candidate is the best answer to the question.\n\n"
        f"{question}\n\nCandidate answer ({label}): {option}\n\n"
        "Answer yes or no only.\nFinal answer: "
    )


def bool_token_ids(tokenizer):
    ids = {}
    for label in ["yes", "no"]:
        for text in [label, " " + label, label.capitalize(), " " + label.capitalize()]:
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(toks) == 1:
                ids[label] = toks[0]
                break
    return ids


@torch.no_grad()
def yes_no_distribution(model, tokenizer, prompt):
    ids = bool_token_ids(tokenizer)
    if len(ids) < 2:
        return {"available": False}
    text = base.chat_prompt(tokenizer, prompt)
    enc = tokenizer([text], add_special_tokens=False, return_tensors="pt")
    input_ids = torch.cat([enc["input_ids"], torch.tensor([[MASK_ID]], dtype=torch.long)], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    logits = model(input_ids, attention_mask=attention_mask).logits[0, -1]
    labels = list(ids.keys())
    label_logits = torch.stack([logits[ids[label]] for label in labels])
    probs = F.softmax(label_logits.float(), dim=0)
    return {label: float(prob.item()) for label, prob in zip(labels, probs)}


@torch.no_grad()
def choice_evidence_distribution(model, tokenizer, sample, labels, args):
    options = parse_options(sample["prompt"])
    if not options or not all(label in options for label in labels):
        return {"available": False}
    raw = {}
    calls = 0
    for label in labels:
        if sample.get("task") == "winogrande":
            prompt = winogrande_evidence_prompt(sample, label, options[label])
        else:
            prompt = generic_evidence_prompt(sample, label, options[label])
        dist = yes_no_distribution(model, tokenizer, prompt)
        if not dist.get("available", True) and "yes" not in dist:
            return {"available": False}
        raw[label] = max(1e-6, float(dist.get("yes", 0.0)))
        calls += 1
    logits = {label: math.log(score) / max(1e-6, args.evidence_temperature) for label, score in raw.items()}
    max_logit = max(logits.values())
    denom = sum(math.exp(v - max_logit) for v in logits.values())
    probs = {label: math.exp(logits[label] - max_logit) / denom for label in labels}
    top = max(probs, key=probs.get)
    ordered = sorted(probs.values(), reverse=True)
    return {
        "available": True,
        "calls": calls,
        "top_label": top,
        "top_prob": probs[top],
        "margin": ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0],
        "raw_yes": raw,
        "probs": probs,
        "options": options,
    }


def fuse_distributions(label_probe, evidence, labels, args):
    if not label_probe.get("available") or not evidence.get("available"):
        return {"available": False}
    fused_logits = {}
    for label in labels:
        lp = max(1e-8, float(label_probe.get("probs", {}).get(label, 0.0)))
        ep = max(1e-8, float(evidence.get("probs", {}).get(label, 0.0)))
        fused_logits[label] = args.label_weight * math.log(lp) + args.evidence_weight * math.log(ep)
    max_logit = max(fused_logits.values())
    denom = sum(math.exp(v - max_logit) for v in fused_logits.values())
    probs = {label: math.exp(fused_logits[label] - max_logit) / denom for label in labels}
    top = max(probs, key=probs.get)
    ordered = sorted(probs.values(), reverse=True)
    return {
        "available": True,
        "top_label": top,
        "top_prob": probs[top],
        "margin": ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0],
        "probs": probs,
    }


def accept_fused(fused, args):
    if not fused.get("available"):
        return False
    return fused.get("top_prob", 0.0) >= args.fused_accept_confidence and fused.get("margin", 0.0) >= args.fused_accept_margin


def run_refinement(model, tokenizer, sample, raw, profile, labels, first_policy, probe, task_prior, budgets, args):
    x, attention_mask, prompt_len, prompt_index = make_state(tokenizer, sample, args, model.device)
    calls = 0
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
        return output, pred, calls, route_steps, stats, features, score, budget_decision

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
    stats = scout_stats(hist, args, last_conf, prompt_len, current_budget)
    features = online_risk_features(profile, probe, pred, labels, stats, args, current_budget)
    score = risk_score(features, profile)
    target, reason, budget_decision = structured_budget_decision(raw["task"], profile, features, score, budgets, task_prior, args)
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
            model, tokenizer, sample, x, attention_mask, prompt_len, prompt_index, extra_steps, policy["schedule"], args, labels, {extra_steps}
        )
        calls += c
        current_budget = target
        stats = scout_stats(hist + hist2, args, last_conf, prompt_len, current_budget)
        features = online_risk_features(profile, probe, pred, labels, stats, args, current_budget)
        score = risk_score(features, profile)
        route_steps.append({"budget": current_budget, "mode": "refine", "reason": reason, "pred": pred, "risk_score": score, "remasked_tokens": remasked})
        risky = bool((labels and pred not in labels) or not pred)
        if not risky:
            break
        target = 32 if current_budget < 32 else current_budget
        refinements += 1
    return output, pred, calls, route_steps, stats, features, score, budget_decision


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
    ap.add_argument("--budgets", default="8,16,24,32")
    ap.add_argument("--scout-steps", type=int, default=8)
    ap.add_argument("--max-refinements", type=int, default=2)
    ap.add_argument("--remask-min-fraction", type=float, default=0.10)
    ap.add_argument("--remask-max-fraction", type=float, default=0.35)
    ap.add_argument("--remask-min-tokens", type=int, default=4)
    ap.add_argument("--fill-confidence-target", type=float, default=0.75)
    ap.add_argument("--margin-target", type=float, default=0.35)
    ap.add_argument("--risk-t16", type=float, default=0.24)
    ap.add_argument("--risk-t24", type=float, default=0.38)
    ap.add_argument("--risk-t32", type=float, default=0.58)
    ap.add_argument("--binary-direct-full-threshold", type=float, default=0.56)
    ap.add_argument("--multi-direct-full-threshold", type=float, default=0.42)
    ap.add_argument("--multi-direct-full-entropy", type=float, default=0.94)
    ap.add_argument("--binary-post-disagree-confidence", type=float, default=0.78)
    ap.add_argument("--multi-post-disagree-confidence", type=float, default=0.82)
    ap.add_argument("--multi-disagreement-policy", default="ignore")
    ap.add_argument("--structured-routing", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--compute-cost", type=float, default=0.055)
    ap.add_argument("--prior-gain-weight", type=float, default=0.75)
    ap.add_argument("--instability-gain-weight", type=float, default=0.30)
    ap.add_argument("--saturation-bonus", type=float, default=0.025)
    ap.add_argument("--prefer-saturation-budget", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--saturation-tolerance", type=float, default=0.08)
    ap.add_argument("--binary-min-16-score", type=float, default=0.24)
    ap.add_argument("--structured-min-24-score", type=float, default=0.48)
    ap.add_argument("--structured-force-32-score", type=float, default=0.70)
    ap.add_argument("--label-weight", type=float, default=0.55)
    ap.add_argument("--evidence-weight", type=float, default=0.45)
    ap.add_argument("--evidence-temperature", type=float, default=0.70)
    ap.add_argument("--fused-accept-confidence", type=float, default=0.78)
    ap.add_argument("--fused-accept-margin", type=float, default=0.28)
    ap.add_argument("--evidence-on-label-disagree", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compact-choice-fast", action="store_true")
    ap.add_argument("--compact-choice-max-labels", type=int, default=5)
    ap.add_argument("--compact-choice-max-prompt-tokens", type=int, default=512)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = base.load_llada(args.model, args.adapter)
    budgets = parse_budgets(args.budgets)
    rows = []
    summary = {}
    torch.cuda.reset_peak_memory_stats()
    for task in [t for t in args.tasks.split(",") if t]:
        raw_samples = domain.build_samples(task, args.limit, args.seed)
        correct = 0
        total_calls = 0
        total_time = 0.0
        routes = {}
        for raw in raw_samples:
            profile = task_profile(raw, tokenizer)
            labels = infer_label_space(raw)
            first_policy = maybe_enable_compact_choice_scout(profile, choose_policy(profile), args)
            sample = apply_prompt(raw, first_policy["prompt"])
            t0 = time.time()
            label_probe = probe_label_distribution(model, tokenizer, sample, labels)
            calls = 1 if label_probe.get("available") else 0
            evidence = choice_evidence_distribution(model, tokenizer, sample, labels, args)
            calls += evidence.get("calls", 0)
            fused = fuse_distributions(label_probe, evidence, labels, args)
            if accept_fused(fused, args):
                pred = fused["top_label"]
                output = f"Final answer: {pred}"
                route_steps = [{"budget": 0, "mode": "choice_evidence_direct", "reason": "fused_posterior_accept", "pred": pred}]
                stats = {"history": [], "flip_count": 0, "first_final_step": 0, "valid_seen": 1, "observed_steps": 0}
                features = {}
                score = None
                budget_decision = {"mode": "choice_evidence_direct"}
            else:
                output, pred, extra_calls, route_steps, stats, features, score, budget_decision = run_refinement(
                    model, tokenizer, sample, raw, profile, labels, first_policy, label_probe, {}, budgets, args
                )
                calls += extra_calls
            dt = time.time() - t0
            ok = pred == raw["gold"]
            correct += int(ok)
            total_calls += calls
            total_time += dt
            route = "->".join(str(x["budget"]) for x in route_steps)
            routes[route] = routes.get(route, 0) + 1
            row = {
                "task": task,
                "id": raw["id"],
                "gold": raw["gold"],
                "pred": pred,
                "correct": ok,
                "output": output,
                "metric": raw["metric"],
                "profile": profile,
                "label_probe": label_probe,
                "choice_evidence": evidence,
                "fused_probe": fused,
                "risk_features": features,
                "risk_score": score,
                "budget_decision": budget_decision,
                "scout_stats": stats,
                "route": route,
                "route_steps": route_steps,
                "seconds": dt,
                "forward_calls": calls,
                "meta": raw.get("meta", {}),
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ["task", "id", "gold", "pred", "correct", "route", "forward_calls"]}, ensure_ascii=False), flush=True)
        n = len(raw_samples)
        summary[task] = {
            "accuracy": correct / n,
            "n": n,
            "seconds": total_time,
            "avg_forward_calls": total_calls / n,
            "route_rates": {k: v / n for k, v in sorted(routes.items())},
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        }
    result = {
        "model": args.model,
        "args": vars(args),
        "method": "choice_evidence_fused_controller",
        "summary": summary,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
