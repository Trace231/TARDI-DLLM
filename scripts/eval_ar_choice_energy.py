#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as domain
import eval_subset as base
from eval_llada_adaptive_router import infer_label_space


def apply_prompt_style(sample, style):
    if style == "plain":
        return
    if style == "xml_label":
        sample["prompt"] = domain.DOMAIN_LABEL_EVAL_PROMPT + "\n" + sample["prompt"]
    elif style == "final_label":
        sample["prompt"] = domain.FINAL_LABEL_EVAL_PROMPT + "\n" + sample["prompt"]
    elif style == "final_label_typed":
        if sample.get("metric") == "decision":
            prefix = domain.DECISION_FINAL_LABEL_EVAL_PROMPT
        elif sample.get("metric") == "bool":
            prefix = domain.BOOL_FINAL_LABEL_EVAL_PROMPT
        elif sample.get("metric") == "number":
            prefix = domain.NUMBER_FINAL_LABEL_EVAL_PROMPT
        elif sample.get("metric") == "span_contains":
            prefix = domain.SPAN_FINAL_LABEL_EVAL_PROMPT
        else:
            prefix = domain.MC_FINAL_LABEL_EVAL_PROMPT
        sample["prompt"] = prefix + "\n" + sample["prompt"]
    else:
        raise ValueError(style)


def candidate_labels(sample):
    labels = infer_label_space(sample)
    if sample.get("metric") in {"letter", "bool", "decision"}:
        return labels
    return []


def candidate_suffix(label, answer_prefix):
    if answer_prefix == "final_answer":
        return "\nFinal answer: " + label
    if answer_prefix == "answer":
        return " " + label
    if answer_prefix == "bare":
        return label
    raise ValueError(answer_prefix)


def encode_candidate(tokenizer, prompt, suffix):
    prompt_text = base.chat_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + suffix, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return None
    return prompt_ids, full_ids


@torch.no_grad()
def score_sample(tokenizer, model, sample, labels, args):
    records = []
    for label in labels:
        suffix = candidate_suffix(label, args.answer_prefix)
        encoded = encode_candidate(tokenizer, sample["prompt"], suffix)
        if encoded is None:
            continue
        prompt_ids, full_ids = encoded
        input_ids = torch.tensor([full_ids], device="cuda")
        attention_mask = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        start = max(0, len(prompt_ids) - 1)
        token_losses = F.cross_entropy(
            logits[:, start:, :].reshape(-1, logits.shape[-1]),
            targets[:, start:].reshape(-1),
            reduction="none",
        )
        token_losses = token_losses.detach().float()
        score = -float(token_losses.mean().item() if args.length_norm else token_losses.sum().item())
        records.append(
            {
                "label": label,
                "suffix": suffix,
                "score": score,
                "n_tokens": int(token_losses.numel()),
            }
        )
    if not records:
        return "", {}, []
    max_score = max(r["score"] for r in records)
    z = sum(math.exp((r["score"] - max_score) / max(1e-6, args.temperature)) for r in records)
    probs = {}
    for r in records:
        probs[r["label"]] = math.exp((r["score"] - max_score) / max(1e-6, args.temperature)) / z
    pred = max(records, key=lambda r: r["score"])["label"]
    return pred, probs, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default="winogrande,commonsenseqa")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--prompt-style", choices=["plain", "xml_label", "final_label", "final_label_typed"], default="final_label_typed")
    ap.add_argument("--answer-prefix", choices=["final_answer", "answer", "bare"], default="final_answer")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--length-norm", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = base.load_ar(args.model, args.adapter)
    rows = []
    summary = {}
    for task in [t for t in args.tasks.split(",") if t]:
        samples = domain.build_samples(task, args.limit, args.seed)
        for sample in samples:
            apply_prompt_style(sample, args.prompt_style)
        correct = 0
        total_time = 0.0
        torch.cuda.reset_peak_memory_stats()
        for sample in samples:
            labels = candidate_labels(sample)
            if not labels:
                pred, probs, records = "", {}, []
            else:
                t0 = time.time()
                pred, probs, records = score_sample(tokenizer, model, sample, labels, args)
                total_time += time.time() - t0
            ok = pred == sample["gold"]
            correct += int(ok)
            row = {
                "task": task,
                "id": sample["id"],
                "gold": sample["gold"],
                "pred": pred,
                "correct": ok,
                "metric": sample["metric"],
                "meta": sample.get("meta", {}),
                "prompt": sample["prompt"],
                "choice_probs": probs,
                "choice_scores": records,
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ["task", "id", "gold", "pred", "correct"]}, ensure_ascii=False), flush=True)
        summary[task] = {
            "accuracy": correct / len(samples) if samples else 0.0,
            "n": len(samples),
            "seconds": total_time,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    payload = {"backend": "ar_choice_energy", "model": args.model, "args": vars(args), "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
