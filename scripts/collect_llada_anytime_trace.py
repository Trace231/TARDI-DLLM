#!/usr/bin/env python3
"""Collect same-trajectory LLaDA checkpoints for anytime stopping.

This script runs one max-budget masked diffusion trajectory and records the
current parsed answer at multiple checkpoints. It does not use an AR model or an
extra label probe, so simulated stopping costs are directly comparable with
fixed-step LLaDA decoding.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as domain
import eval_subset as base
from eval_llada_sampler_variants import MASK_ID, model_logits, token_budget, x0_and_conf
from eval_llada_adaptive_router import apply_prompt, infer_label_space, parse_for_metric


def parse_ints(text):
    return sorted({int(x.strip()) for x in text.split(",") if x.strip()})


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def build_payload(args, rows, t0):
    by_task = {}
    for row in rows:
        task = row["task"]
        by_task.setdefault(task, {"n": 0})
        by_task[task]["n"] += 1
    return {
        "schema": "llada_anytime_trace_v1",
        "model": args.model,
        "args": vars(args),
        "n": len(rows),
        "seconds": time.time() - t0,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "summary": by_task,
        "rows": rows,
    }


def confidence_stats(last_conf, prompt_len):
    vals = last_conf[:, prompt_len:]
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return {"mean_fill_confidence": 0.0, "min_fill_confidence": 0.0}
    vals = vals.detach().float()
    return {
        "mean_fill_confidence": float(vals.mean().item()),
        "min_fill_confidence": float(vals.min().item()),
    }


@torch.no_grad()
def run_trace(model, tokenizer, sample, args, labels):
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
    assert args.max_steps % num_blocks == 0
    steps_per_block = args.max_steps // num_blocks
    checkpoints = set(args.checkpoints)
    checkpoints.add(args.max_steps)
    last_conf = torch.zeros_like(x, dtype=torch.float32)
    states = []
    forward_calls = 0
    previous_pred = ""
    flip_count = 0
    stable_count = 0
    first_valid_step = None
    first_final_step = None

    for nb in range(num_blocks):
        block_start = prompt.shape[1] + nb * args.block_length
        block_end = prompt.shape[1] + (nb + 1) * args.block_length
        block_mask = x[:, block_start:block_end] == MASK_ID
        budgets = token_budget(block_mask, steps_per_block, args.schedule)
        for i in range(steps_per_block):
            global_step = nb * steps_per_block + i + 1
            mask_index = x == MASK_ID
            if bool(mask_index[:, prompt.shape[1]:block_end].any().item()):
                logits = model_logits(model, x, attention_mask, prompt_index, args.cfg)
                forward_calls += 1
                x0, conf, _ = x0_and_conf(
                    logits,
                    x,
                    mask_index,
                    prompt.shape[1],
                    block_end,
                    args.temperature,
                    args.remasking,
                )
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
            if global_step in checkpoints:
                gen_ids = x[:, prompt.shape[1]:]
                output = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
                pred = parse_for_metric(sample, output)
                valid = bool(pred and ((not labels) or pred in labels))
                if valid and first_valid_step is None:
                    first_valid_step = global_step
                if pred:
                    if previous_pred and pred != previous_pred:
                        flip_count += 1
                        stable_count = 1
                    elif pred == previous_pred:
                        stable_count += 1
                    else:
                        stable_count = 1
                    previous_pred = pred
                else:
                    stable_count = 0
                filled_ratio = float((gen_ids != MASK_ID).float().mean().item())
                conf_stats = confidence_stats(last_conf, prompt.shape[1])
                states.append(
                    {
                        "step": global_step,
                        "forward_calls": forward_calls,
                        "pred": pred,
                        "valid": valid,
                        "correct": bool(pred == sample["gold"]),
                        "filled_ratio": filled_ratio,
                        "stable_count": stable_count,
                        "flip_count": flip_count,
                        "first_valid_step": first_valid_step,
                        "mean_fill_confidence": conf_stats["mean_fill_confidence"],
                        "min_fill_confidence": conf_stats["min_fill_confidence"],
                        "output": output,
                    }
                )
    final_pred = states[-1]["pred"] if states else ""
    for state in states:
        if state["pred"] == final_pred and final_pred:
            first_final_step = state["step"]
            break
    for state in states:
        state["first_final_step"] = first_final_step
        state["late_first_final"] = 1.0 if first_final_step is None else first_final_step / max(1, args.max_steps)
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default="mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="2,4,6,8,12,16,24,32")
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--schedule", choices=["uniform", "front_loaded", "back_loaded", "middle_heavy"], default="uniform")
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args()
    args.checkpoints = [x for x in parse_ints(args.checkpoints) if x <= args.max_steps]
    if args.max_steps not in args.checkpoints:
        args.checkpoints.append(args.max_steps)
    args.checkpoints = sorted(set(args.checkpoints))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = base.load_llada(args.model, args.adapter)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    t0 = time.time()
    for task in [t for t in args.tasks.split(",") if t]:
        for raw in domain.build_samples(task, args.limit, args.seed):
            sample = apply_prompt(dict(raw), "typed")
            labels = infer_label_space(sample)
            states = run_trace(model, tokenizer, sample, args, labels)
            row = {
                "task": task,
                "id": str(raw["id"]),
                "gold": raw["gold"],
                "metric": raw["metric"],
                "label_space": labels,
                "states": states,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "task": row["task"],
                        "id": row["id"],
                        "gold": row["gold"],
                        "final_pred": states[-1]["pred"] if states else "",
                        "final_correct": states[-1]["correct"] if states else False,
                        "n_states": len(states),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.flush_every > 0 and len(rows) % args.flush_every == 0:
                payload = build_payload(args, rows, t0)
                payload["partial"] = True
                write_json_atomic(args.out, payload)
                write_json_atomic(str(args.out) + ".partial", payload)
    payload = build_payload(args, rows, t0)
    payload["partial"] = False
    write_json_atomic(args.out, payload)
    print(json.dumps({"out": args.out, "n": payload["n"], "summary": payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
