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
from eval_llada_sampler_variants import MASK_ID, generate_variant


CALIBRATED_DECISION_PROMPT = """Think briefly, then end with exactly one line:
Final answer: <yes|no|maybe>

Use this decision rule:
- Choose yes only when the snippets support a positive answer to the question.
- Choose no only when the snippets support a negative answer or contradict the proposed relation.
- Choose maybe only when the snippets are genuinely inconclusive or mixed. Do not choose maybe merely because the topic is biomedical.
The final answer must be yes, no, or maybe only. Do not put any text after the final answer line.
"""


def infer_label_space(sample):
    if sample.get("metric") == "bool":
        return ["yes", "no"]
    if sample.get("metric") == "decision":
        return ["yes", "no", "maybe"]
    if sample.get("metric") != "letter":
        return []
    labels = []
    for line in sample["prompt"].splitlines():
        m = re.match(r"\s*([A-J])\s*[\.\)\uff0e\uff09]\s+", line)
        if m and m.group(1) not in labels:
            labels.append(m.group(1))
    if labels:
        return labels
    return list(domain.LETTERS)


def task_profile(sample, tokenizer):
    text = base.chat_prompt(tokenizer, sample["prompt"])
    n_prompt_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    labels = infer_label_space(sample)
    return {
        "metric": sample.get("metric"),
        "label_space": labels,
        "n_labels": len(labels),
        "prompt_tokens": n_prompt_tokens,
    }


def choose_policy(profile):
    # This policy uses observable task shape rather than dataset names.
    if profile["metric"] == "decision":
        return {
            "steps": 32,
            "schedule": "uniform",
            "prompt": "calibrated_decision",
            "fast_candidate": False,
            "reason": "decision labels and long evidence are unsafe for aggressive compression",
        }
    if profile["metric"] not in {"letter", "bool"}:
        return {
            "steps": 32,
            "schedule": "uniform",
            "prompt": "typed",
            "fast_candidate": False,
            "reason": "non-choice task uses conservative decoding",
        }
    if profile["prompt_tokens"] > 512 or profile["n_labels"] >= 8:
        return {
            "steps": 32,
            "schedule": "uniform",
            "prompt": "typed",
            "fast_candidate": False,
            "reason": "long or high-cardinality choice task uses conservative decoding",
        }
    if profile["n_labels"] <= 2:
        return {
            "steps": 8,
            "schedule": "back_loaded",
            "prompt": "typed",
            "fast_candidate": True,
            "reason": "short binary choice task uses delayed filling",
        }
    if profile["n_labels"] >= 5:
        return {
            "steps": 8,
            "schedule": "middle_heavy",
            "prompt": "typed",
            "fast_candidate": True,
            "reason": "moderate-cardinality choice task uses mid-trajectory refinement",
        }
    return {
        "steps": 32,
        "schedule": "uniform",
        "prompt": "typed",
        "fast_candidate": False,
        "reason": "default conservative choice decoding",
    }


def apply_prompt(sample, prompt_kind):
    sample = dict(sample)
    if prompt_kind == "calibrated_decision":
        sample["prompt"] = CALIBRATED_DECISION_PROMPT + "\n" + sample["prompt"]
    elif prompt_kind == "typed":
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
    return sample


def parse_for_metric(sample, output):
    if sample.get("metric") == "letter":
        labels = infer_label_space(sample)
        return domain.parse_letter(output, choices="".join(labels) if labels else domain.LETTERS)
    if sample.get("metric") == "decision":
        return domain.parse_decision(output)
    if sample.get("metric") == "bool":
        return base.parse_bool(output)
    pred, _ = domain.score(sample["metric"], output, sample["gold"])
    return pred


def label_token_ids(tokenizer, labels):
    ids = {}
    for label in labels:
        variants = [label, " " + label]
        for text in variants:
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(toks) == 1:
                ids[label] = toks[0]
                break
    return ids


@torch.no_grad()
def probe_label_distribution(model, tokenizer, sample, labels):
    ids = label_token_ids(tokenizer, labels)
    if len(ids) < 2:
        return {"available": False}
    probe = base.chat_prompt(tokenizer, sample["prompt"].rstrip() + "\nFinal answer: ")
    enc = tokenizer([probe], add_special_tokens=False, return_tensors="pt")
    input_ids = torch.cat(
        [enc["input_ids"], torch.tensor([[MASK_ID]], dtype=torch.long)],
        dim=1,
    ).to(model.device)
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
        "probs": {label: float(prob.item()) for label, prob in zip(ordered_labels, probs)},
    }


def should_accept(pred, labels, probe, min_confidence, min_margin):
    if labels and pred not in labels:
        return False, "invalid_label"
    if not probe.get("available"):
        return True, "no_probe_available"
    # The probe is an auxiliary uncertainty signal, not a second judge.
    # Fall back only when both absolute confidence and relative margin are weak.
    if probe["top_prob"] < min_confidence and probe["margin"] < min_margin:
        return False, "low_confidence_and_margin"
    return True, "accepted"


def run_decode(model, tokenizer, sample, policy, args):
    text = base.chat_prompt(tokenizer, sample["prompt"])
    enc = tokenizer([text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    t0 = time.time()
    out, calls = generate_variant(
        model,
        input_ids,
        attention_mask=attn,
        steps=policy["steps"],
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg=args.cfg,
        remasking=args.remasking,
        schedule=policy["schedule"],
        sampler="standard",
    )
    dt = time.time() - t0
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return output, dt, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="winogrande,commonsenseqa,pubmedqa")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--accept-confidence", type=float, default=0.50)
    ap.add_argument("--accept-margin", type=float, default=0.05)
    args = ap.parse_args()

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
        fallback_count = 0
        constrained_count = 0
        for raw in raw_samples:
            profile = task_profile(raw, tokenizer)
            first_policy = choose_policy(profile)
            labels = profile["label_space"]
            sample = apply_prompt(raw, first_policy["prompt"])
            output, dt, calls = run_decode(model, tokenizer, sample, first_policy, args)
            pred = parse_for_metric(sample, output)
            probe = probe_label_distribution(model, tokenizer, sample, labels)
            calls += 1 if probe.get("available") else 0
            accept, accept_reason = should_accept(pred, labels, probe, args.accept_confidence, args.accept_margin)

            final_policy = dict(first_policy)
            final_output = output
            final_pred = pred
            final_probe = probe
            fallback_used = False
            constraint_used = False
            if not accept and first_policy["fast_candidate"]:
                fallback_used = True
                fallback_count += 1
                final_policy = {
                    "steps": 32,
                    "schedule": "uniform",
                    "prompt": first_policy["prompt"],
                    "fast_candidate": False,
                    "reason": "confidence or validity fallback",
                }
                final_sample = apply_prompt(raw, final_policy["prompt"])
                final_output, dt2, calls2 = run_decode(model, tokenizer, final_sample, final_policy, args)
                final_pred = parse_for_metric(final_sample, final_output)
                final_probe = probe_label_distribution(model, tokenizer, final_sample, labels)
                dt += dt2
                calls += calls2 + (1 if final_probe.get("available") else 0)
                sample = final_sample

            if labels and final_pred not in labels and final_probe.get("available"):
                final_pred = final_probe["top_label"]
                constraint_used = True
                constrained_count += 1

            ok = final_pred == raw["gold"]
            correct += int(ok)
            total_time += dt
            total_calls += calls
            row = {
                "task": task,
                "id": raw["id"],
                "gold": raw["gold"],
                "pred": final_pred,
                "correct": ok,
                "raw_pred": pred,
                "output": final_output,
                "metric": raw["metric"],
                "profile": profile,
                "first_policy": first_policy,
                "final_policy": final_policy,
                "probe": final_probe,
                "first_probe": probe,
                "accept_reason": accept_reason,
                "fallback_used": fallback_used,
                "constraint_used": constraint_used,
                "seconds": dt,
                "forward_calls": calls,
                "meta": raw.get("meta", {}),
            }
            rows.append(row)
            print(
                json.dumps(
                    {k: row[k] for k in ["task", "id", "gold", "pred", "correct", "accept_reason", "fallback_used", "forward_calls"]},
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
            "fallback_rate": fallback_count / n,
            "constraint_rate": constrained_count / n,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    payload = {"model": args.model, "args": vars(args), "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
