#!/usr/bin/env python3
"""Collect full-checkpoint online decision tables on GPU.

For each sample this runs one LLaDA reverse process to the maximum checkpoint and
records the online-visible state at probe/4/8/16/24/32. It also runs one AR
judge prediction. The output is compatible with eval_online_bandit_router.py.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as domain
import eval_subset as base
from eval_llada_adaptive_router import apply_prompt, choose_policy, infer_label_space, parse_for_metric, probe_label_distribution, task_profile
from eval_llada_sampler_variants import MASK_ID, model_logits, token_budget, x0_and_conf


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
    vals = [v / total for v in vals]
    ent = -sum(v * math.log(v) for v in vals)
    return ent / math.log(len(vals)) if len(vals) > 1 else 0.0


def final_label_prefix(sample):
    if sample.get("metric") == "decision":
        return domain.DECISION_FINAL_LABEL_EVAL_PROMPT
    if sample.get("metric") == "bool":
        return domain.BOOL_FINAL_LABEL_EVAL_PROMPT
    if sample.get("metric") == "number":
        return domain.NUMBER_FINAL_LABEL_EVAL_PROMPT
    if sample.get("metric") == "span_contains":
        return domain.SPAN_FINAL_LABEL_EVAL_PROMPT
    return domain.MC_FINAL_LABEL_EVAL_PROMPT


def prepare_sample(raw, tokenizer):
    # Preserve the same typed final-label prompt convention used by controller scripts.
    return apply_prompt(dict(raw), "typed")


def make_state(tokenizer, sample, model, gen_length):
    text = base.chat_prompt(tokenizer, sample["prompt"])
    enc = tokenizer([text], add_special_tokens=False, padding=True, return_tensors="pt")
    prompt = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    attention_mask = torch.cat(
        [attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)],
        dim=-1,
    )
    return x, attention_mask, prompt.shape[1], x != MASK_ID


def base_features(profile, probe, checkpoint, spent_calls):
    probs = probe.get("probs") or {}
    top = safe_float(probe.get("top_prob"))
    margin = safe_float(probe.get("margin"))
    entropy = norm_entropy(probs)
    n_labels = int(profile.get("n_labels") or 0)
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


def history_features(history, checkpoint, pred, probe):
    prefix = [h for h in history if h["local_step"] <= checkpoint]
    valid = [h for h in prefix if h.get("valid")]
    preds = [h["pred"] for h in valid if h.get("pred")]
    flips = sum(1 for a, b in zip(preds, preds[1:]) if a != b)
    first_valid = valid[0]["local_step"] if valid else 0
    probe_top = probe.get("top_label") if probe.get("available") else ""
    return {
        "scout_seen": 1.0 if prefix else 0.0,
        "scout_valid": 1.0 if pred else 0.0,
        "scout_filled_ratio": safe_float(prefix[-1].get("filled_ratio")) if prefix else 0.0,
        "early_flip_count_norm": min(flips, 4) / 4.0,
        "first_valid_step_norm": first_valid / 32.0,
        "valid_seen_norm": min(len(valid), 8) / 8.0,
        "probe_scout_disagree": 1.0 if pred and probe_top and pred != probe_top else 0.0,
    }


def confidence_features(conf_store, prompt_len):
    gen_conf = conf_store[:, prompt_len:]
    finite = gen_conf[torch.isfinite(gen_conf)]
    if finite.numel() == 0:
        return {"mean_fill_confidence": 0.0, "min_fill_confidence": 0.0}
    return {
        "mean_fill_confidence": float(finite.float().mean().item()),
        "min_fill_confidence": float(finite.float().min().item()),
    }


@torch.no_grad()
def llada_checkpoints(model, tokenizer, sample, profile, labels, probe, args):
    max_step = max(args.checkpoints)
    assert args.gen_length == args.block_length, "v1 collector expects a single generation block"
    x, attention_mask, prompt_len, prompt_index = make_state(tokenizer, sample, model, args.gen_length)
    block_mask = x[:, prompt_len : prompt_len + args.block_length] == MASK_ID
    budgets = token_budget(block_mask, max_step, args.schedule)
    conf_store = torch.full_like(x, -torch.inf, dtype=torch.float32)
    checkpoints = set(args.checkpoints)
    states = []
    history = []
    forward_calls = 0

    def spent_calls():
        # The final-label posterior probe is one LLaDA forward call before the
        # reverse process starts. Count it explicitly so reported budgets match
        # the deployed online policy.
        return 1 + forward_calls

    def record(step):
        output = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
        pred = parse_for_metric(sample, output)
        filled = float((x[:, prompt_len:] != MASK_ID).float().mean().item())
        valid = bool(pred and (not labels or pred in labels))
        history.append({"local_step": step, "pred": pred, "valid": valid, "filled_ratio": filled})
        spent = spent_calls()
        feats = base_features(profile, probe, step, spent)
        feats.update(history_features(history, step, pred, probe))
        feats.update(confidence_features(conf_store, prompt_len))
        states.append(
            {
                "checkpoint": step,
                "spent_llda_calls": spent,
                "llda_pred": pred,
                "llda_valid": valid,
                "llda_correct": pred == sample["gold"],
                "router_features": feats,
                "output": output,
            }
        )

    if 0 in checkpoints:
        probe_pred = probe.get("top_label") if probe.get("available") else ""
        feats = base_features(profile, probe, 0, 1)
        feats.update(
            {
                "scout_seen": 0.0,
                "scout_valid": 1.0 if probe_pred else 0.0,
                "scout_filled_ratio": 0.0,
                "early_flip_count_norm": 0.0,
                "first_valid_step_norm": 0.0,
                "valid_seen_norm": 1.0 if probe_pred else 0.0,
                "probe_scout_disagree": 0.0,
                "mean_fill_confidence": 0.0,
                "min_fill_confidence": 0.0,
            }
        )
        states.append(
            {
                "checkpoint": 0,
                "spent_llda_calls": 1,
                "llda_pred": probe_pred,
                "llda_valid": bool(probe_pred and (not labels or probe_pred in labels)),
                "llda_correct": probe_pred == sample["gold"],
                "router_features": feats,
                "output": f"Final answer: {probe_pred}" if probe_pred else "",
            }
        )

    for step in range(1, max_step + 1):
        mask_index = x == MASK_ID
        if not bool(mask_index[:, prompt_len:].any().item()):
            if step in checkpoints:
                record(step)
            continue
        logits = model_logits(model, x, attention_mask, prompt_index, args.cfg)
        forward_calls += 1
        x0, conf, _ = x0_and_conf(logits, x, mask_index, prompt_len, prompt_len + args.block_length, args.temperature, args.remasking)
        transfer = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for j in range(conf.shape[0]):
            k = int(budgets[j, step - 1].item())
            if k <= 0:
                continue
            _, idx = torch.topk(conf[j], k=k)
            transfer[j, idx] = True
        x[transfer] = x0[transfer]
        conf_store[transfer] = conf[transfer].float()
        if step in checkpoints:
            record(step)

    for i, state in enumerate(states):
        state["has_next_state"] = i + 1 < len(states)
        state["next_checkpoint"] = states[i + 1]["checkpoint"] if i + 1 < len(states) else None
    return states, forward_calls


@torch.no_grad()
def ar_judge(tokenizer, model, sample, args):
    text = base.chat_prompt(tokenizer, sample["prompt"])
    enc = tokenizer([text], padding=True, return_tensors="pt").to("cuda")
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    dt = time.time() - t0
    output = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)[0]
    pred, ok = domain.score(sample["metric"], output, sample["gold"])
    return pred, ok, output, dt


def validate_no_leakage(samples):
    banned = {"gold", "correct", "route", "final_budget", "forward_calls", "oracle"}
    offenders = []
    for sample in samples:
        for state in sample["states"]:
            for key in state["router_features"]:
                if key in banned:
                    offenders.append((sample["task"], sample["id"], state["checkpoint"], key))
    if offenders:
        raise RuntimeError(f"router feature leakage detected: {offenders[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llada-model", required=True)
    ap.add_argument("--ar-model", required=True)
    ap.add_argument("--ar-adapter", default=None)
    ap.add_argument("--tasks", default="winogrande,commonsenseqa,arc_challenge,sciq,hellaswag")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="0,4,8,16,24,32")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--schedule", choices=["uniform", "front_loaded", "back_loaded", "middle_heavy"], default="uniform")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    args.checkpoints = sorted({int(x) for x in args.checkpoints.split(",") if x.strip()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    llada_tok, llada_model = base.load_llada(args.llada_model)
    ar_tok, ar_model = base.load_ar(args.ar_model, args.ar_adapter)
    torch.cuda.reset_peak_memory_stats()
    samples_out = []
    summary = {}
    for task in [t for t in args.tasks.split(",") if t]:
        raw_samples = domain.build_samples(task, args.limit, args.seed)
        task_correct = {"llda32": 0, "ar": 0}
        task_time = {"llda": 0.0, "ar": 0.0}
        for raw in raw_samples:
            sample = prepare_sample(raw, llada_tok)
            profile = task_profile(sample, llada_tok)
            labels = infer_label_space(sample)
            t0 = time.time()
            probe = probe_label_distribution(llada_model, llada_tok, sample, labels)
            states, calls = llada_checkpoints(llada_model, llada_tok, sample, profile, labels, probe, args)
            task_time["llda"] += time.time() - t0
            ar_pred, ar_ok, ar_output, ar_dt = ar_judge(ar_tok, ar_model, sample, args)
            task_time["ar"] += ar_dt
            task_correct["llda32"] += int(states[-1]["llda_correct"])
            task_correct["ar"] += int(ar_ok)
            sample_out = {
                "task": task,
                "id": str(raw["id"]),
                "metric": raw["metric"],
                "label_space": labels,
                "gold": raw["gold"],
                "ar_pred": ar_pred,
                "ar_correct": bool(ar_ok),
                "ar_output": ar_output,
                "states": states,
            }
            samples_out.append(sample_out)
            print(
                json.dumps(
                    {
                        "task": task,
                        "id": raw["id"],
                        "gold": raw["gold"],
                        "llda32": states[-1]["llda_pred"],
                        "ar": ar_pred,
                        "llda32_correct": states[-1]["llda_correct"],
                        "ar_correct": ar_ok,
                        "llada_calls": 1 + calls,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        n = len(raw_samples)
        summary[task] = {
            "n": n,
            "llda32_accuracy": task_correct["llda32"] / n,
            "ar_accuracy": task_correct["ar"] / n,
            "llda_seconds": task_time["llda"],
            "ar_seconds": task_time["ar"],
        }

    validate_no_leakage(samples_out)
    payload = {
        "schema": "online_decision_table_v1",
        "collector": "full_checkpoint_gpu_v1",
        "args": vars(args),
        "summary": summary,
        "checkpoints": args.checkpoints,
        "n": len(samples_out),
        "samples": samples_out,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": summary, "out": args.out, "n": len(samples_out), "peak_mem_gb": payload["peak_mem_gb"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
