import argparse
import json
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
    label_token_ids,
    parse_for_metric,
    probe_label_distribution,
    task_profile,
)
from eval_llada_refinement_controller import maybe_enable_compact_choice_scout
from eval_llada_risk_controller import decode_with_scout_stats, norm_entropy
from eval_llada_sampler_variants import MASK_ID


def budget_policy(first_policy, steps):
    policy = dict(first_policy)
    policy["steps"] = int(steps)
    if steps >= 32:
        policy["schedule"] = "uniform"
    return policy


def strip_final_answer(text):
    lines = []
    for line in str(text).splitlines():
        if re.search(r"(?i)^\s*(final\s+answer|answer)\s*[:：]", line):
            continue
        if re.search(r"(?i)<answer>.*?</answer>", line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned[:1600]


@torch.no_grad()
def trajectory_label_distribution(model, tokenizer, sample, labels, output):
    ids = label_token_ids(tokenizer, labels)
    if len(ids) < 2:
        return {"available": False}
    reasoning = strip_final_answer(output)
    prompt = sample["prompt"].rstrip()
    if reasoning:
        prompt += "\n\nDraft reasoning:\n" + reasoning
    prompt += "\n\nFinal answer: "
    text = base.chat_prompt(tokenizer, prompt)
    enc = tokenizer([text], add_special_tokens=False, return_tensors="pt")
    input_ids = torch.cat([enc["input_ids"], torch.tensor([[MASK_ID]], dtype=torch.long)], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    logits = model(input_ids, attention_mask=attention_mask).logits[0, -1]
    label_logits = torch.stack([logits[ids[label]] for label in ids])
    probs = F.softmax(label_logits.float(), dim=0)
    order = torch.argsort(probs, descending=True)
    ordered_labels = list(ids.keys())
    top_label = ordered_labels[int(order[0])]
    top_prob = float(probs[order[0]].item())
    second_prob = float(probs[order[1]].item()) if len(order) > 1 else 0.0
    return {
        "available": True,
        "top_label": top_label,
        "top_prob": top_prob,
        "margin": top_prob - second_prob,
        "entropy": norm_entropy({label: float(prob.item()) for label, prob in zip(ordered_labels, probs)}),
        "probs": {label: float(prob.item()) for label, prob in zip(ordered_labels, probs)},
    }


def probe_supports_pred(probe, pred, min_conf, min_margin):
    if not probe.get("available") or not pred:
        return False
    return probe.get("top_label") == pred and probe.get("top_prob", 0.0) >= min_conf and probe.get("margin", 0.0) >= min_margin


def probe_contradicts_pred(probe, pred, min_conf, min_margin):
    if not probe.get("available") or not pred:
        return False
    return probe.get("top_label") != pred and probe.get("top_prob", 0.0) >= min_conf and probe.get("margin", 0.0) >= min_margin


def valid_pred(pred, labels):
    return bool(pred) and (not labels or pred in labels)


def route_after_decode(pred, labels, prompt_probe, traj_probe, stats, args, stage):
    if not valid_pred(pred, labels):
        return False, f"{stage}_invalid"
    traj_support = probe_supports_pred(traj_probe, pred, args.traj_accept_confidence, args.traj_accept_margin)
    prompt_support = probe_supports_pred(prompt_probe, pred, args.prompt_accept_confidence, args.prompt_accept_margin)
    traj_contra = probe_contradicts_pred(traj_probe, pred, args.traj_reject_confidence, args.traj_reject_margin)
    prompt_contra = probe_contradicts_pred(prompt_probe, pred, args.prompt_reject_confidence, args.prompt_reject_margin)
    flips = stats.get("flip_count", 0)
    first_final = stats.get("first_final_step")
    stable = flips <= args.max_accept_flips and (first_final is not None)
    if traj_support and stable and not prompt_contra:
        return True, f"{stage}_trajectory_probe_accept"
    if prompt_support and traj_probe.get("available") and traj_probe.get("top_label") == pred and not traj_contra:
        return True, f"{stage}_dual_probe_accept"
    if stage != "scout8" and valid_pred(pred, labels) and not traj_contra and not prompt_contra and flips <= args.max_accept_flips:
        return True, f"{stage}_post_refine_accept"
    return False, f"{stage}_needs_more"


def run_decode(model, tokenizer, sample, first_policy, budget, args, labels):
    policy = budget_policy(first_policy, budget)
    checkpoints = {max(1, budget // 2), budget}
    output, pred, dt, calls, stats = decode_with_scout_stats(model, tokenizer, sample, policy, args, labels, checkpoints)
    return output, pred, dt, calls, stats


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
    ap.add_argument("--traj-accept-confidence", type=float, default=0.58)
    ap.add_argument("--traj-accept-margin", type=float, default=0.06)
    ap.add_argument("--prompt-accept-confidence", type=float, default=0.62)
    ap.add_argument("--prompt-accept-margin", type=float, default=0.08)
    ap.add_argument("--traj-reject-confidence", type=float, default=0.68)
    ap.add_argument("--traj-reject-margin", type=float, default=0.10)
    ap.add_argument("--prompt-reject-confidence", type=float, default=0.72)
    ap.add_argument("--prompt-reject-margin", type=float, default=0.12)
    ap.add_argument("--max-accept-flips", type=int, default=0)
    ap.add_argument("--mid-budget", type=int, default=16)
    ap.add_argument("--full-budget", type=int, default=32)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = base.load_llada(args.model, args.adapter)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    summary = {}
    for task in [x for x in args.tasks.split(",") if x]:
        raw_samples = domain.build_samples(task, args.limit, args.seed)
        correct = 0
        total_calls = 0
        total_time = 0.0
        route_counts = {}
        for raw in raw_samples:
            profile = task_profile(raw, tokenizer)
            labels = infer_label_space(raw)
            first_policy = maybe_enable_compact_choice_scout(profile, choose_policy(profile), args)
            sample = apply_prompt(raw, first_policy["prompt"])
            t0 = time.time()
            prompt_probe = probe_label_distribution(model, tokenizer, sample, labels)
            calls = 1 if prompt_probe.get("available") else 0
            route_steps = []

            output8, pred8, dt8, c8, stats8 = run_decode(model, tokenizer, sample, first_policy, 8, args, labels)
            calls += c8
            traj8 = trajectory_label_distribution(model, tokenizer, sample, labels, output8)
            calls += 1 if traj8.get("available") else 0
            accept, reason = route_after_decode(pred8, labels, prompt_probe, traj8, stats8, args, "scout8")
            route_steps.append({"budget": 8, "pred": pred8, "reason": reason, "accepted": accept, "stats": stats8, "trajectory_probe": traj8})
            output = output8
            pred = pred8

            if not accept:
                output_mid, pred_mid, dt_mid, c_mid, stats_mid = run_decode(model, tokenizer, sample, first_policy, args.mid_budget, args, labels)
                calls += c_mid
                traj_mid = trajectory_label_distribution(model, tokenizer, sample, labels, output_mid)
                calls += 1 if traj_mid.get("available") else 0
                accept_mid, reason_mid = route_after_decode(pred_mid, labels, prompt_probe, traj_mid, stats_mid, args, f"fresh{args.mid_budget}")
                route_steps.append({"budget": args.mid_budget, "pred": pred_mid, "reason": reason_mid, "accepted": accept_mid, "stats": stats_mid, "trajectory_probe": traj_mid})
                output = output_mid
                pred = pred_mid
                if not accept_mid:
                    output_full, pred_full, dt_full, c_full, stats_full = run_decode(model, tokenizer, sample, first_policy, args.full_budget, args, labels)
                    calls += c_full
                    route_steps.append({"budget": args.full_budget, "pred": pred_full, "reason": "fresh_full", "accepted": True, "stats": stats_full})
                    output = output_full
                    pred = pred_full

            dt = time.time() - t0
            ok = pred == raw["gold"]
            correct += int(ok)
            total_calls += calls
            total_time += dt
            route = "->".join(str(x["budget"]) for x in route_steps)
            route_counts[route] = route_counts.get(route, 0) + 1
            row = {
                "task": task,
                "id": raw["id"],
                "gold": raw["gold"],
                "pred": pred,
                "correct": ok,
                "output": output,
                "metric": raw["metric"],
                "prompt_probe": prompt_probe,
                "route": route,
                "route_steps": route_steps,
                "forward_calls": calls,
                "seconds": dt,
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
            "route_rates": {k: v / n for k, v in sorted(route_counts.items())},
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        Path(args.out).write_text(
            json.dumps({"method": "probe_cascade_controller", "model": args.model, "args": vars(args), "complete": False, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2)
        )
    payload = {"method": "probe_cascade_controller", "model": args.model, "args": vars(args), "complete": True, "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
