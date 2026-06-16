#!/usr/bin/env python3
"""Faithful ReMDM (Remasking Discrete Diffusion Models, Wang et al. 2025, arXiv:2503.00307)
sampler applied to LLaDA-8B. Implements the published remasking reverse process:

  q_sigma(z_s | z_t, x):
    z_t != m (unmasked):  Cat( (1-sigma_t) x + sigma_t m )            # remask w.p. sigma_t
    z_t == m (masked):    Cat( (alpha_s-(1-sigma_t)alpha_t)/(1-alpha_t) x
                                + (1-alpha_s-alpha_t sigma_t)/(1-alpha_t) m )

  sigma schedules (alpha_t = 1 - t, linear masking schedule):
    ReMDM-cap:     sigma_t = min(eta, (1-alpha_s)/alpha_t)
    ReMDM-rescale: sigma_t = eta * min(1, (1-alpha_s)/alpha_t)
  with 0 <= sigma_t <= sigma_t^max = min(1, (1-alpha_s)/alpha_t).

This is a faithful re-implementation of the published method (not a "-like" stand-in):
masked tokens unmask to the model's x0 prediction; already-decoded tokens are probabilistically
remasked per sigma_t, giving ReMDM's iterative refinement. cfg/temperature=0 (greedy x0).
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import eval_subset as base
import eval_domain_shift as domain
from eval_llada_sampler_variants import MASK_ID, model_logits, parse_for_sample


@torch.no_grad()
def generate_remdm(model, prompt, attention_mask, steps, gen_length, eta, kind, seed):
    g = torch.Generator(device=model.device).manual_seed(seed)
    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((1, gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)
    prompt_index = (x != MASK_ID)
    gen_lo, gen_hi = prompt.shape[1], prompt.shape[1] + gen_length
    forward_calls = 0
    eps = 1e-6
    for k in range(steps):
        alpha_t = k / steps          # signal level at time t (alpha increases as we denoise)
        alpha_s = (k + 1) / steps     # next step
        logits = model_logits(model, x, attention_mask, prompt_index, 0.0)
        forward_calls += 1
        x0 = torch.argmax(logits, dim=-1)             # greedy clean prediction
        sigma_max = min(1.0, (1.0 - alpha_s) / max(alpha_t, eps))
        if kind == "cap":
            sigma = min(eta, (1.0 - alpha_s) / max(alpha_t, eps))
        else:  # rescale
            sigma = eta * sigma_max
        sigma = max(0.0, min(sigma, sigma_max))
        p_unmask = (alpha_s - (1.0 - sigma) * alpha_t) / max(1.0 - alpha_t, eps) if (1.0 - alpha_t) > eps else 1.0
        p_unmask = max(0.0, min(1.0, p_unmask))

        is_mask = (x == MASK_ID)
        region = torch.zeros_like(x, dtype=torch.bool)
        region[:, gen_lo:gen_hi] = True
        r1 = torch.rand(x.shape, generator=g, device=model.device)
        r2 = torch.rand(x.shape, generator=g, device=model.device)
        # masked -> unmask to x0 with prob p_unmask
        do_unmask = is_mask & region & (r1 < p_unmask)
        # unmasked (in region) -> remask with prob sigma, else re-predict to x0
        unmasked = (~is_mask) & region
        do_remask = unmasked & (r2 < sigma)
        keep_pred = unmasked & (~do_remask)
        x = torch.where(do_unmask, x0, x)
        x = torch.where(keep_pred, x0, x)
        x = torch.where(do_remask, torch.full_like(x, MASK_ID), x)
    # any leftover masks -> final argmax
    if (x[:, gen_lo:gen_hi] == MASK_ID).any():
        logits = model_logits(model, x, attention_mask, prompt_index, 0.0)
        forward_calls += 1
        x0 = torch.argmax(logits, dim=-1)
        leftover = (x == MASK_ID)
        leftover[:, :gen_lo] = False
        leftover[:, gen_hi:] = False
        x = torch.where(leftover, x0, x)
    return x, forward_calls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--tasks", default="mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--gen-length", type=int, default=32)
    p.add_argument("--eta", type=float, default=0.04)
    p.add_argument("--kind", choices=["cap", "rescale"], default="cap")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok, model = base.load_llada(args.model, args.adapter)
    torch.cuda.reset_peak_memory_stats()
    rows = []; summary = {}
    for task in [t for t in args.tasks.split(",") if t]:
        samples = domain.build_samples(task, args.limit, args.seed)
        for sample in samples:
            m = sample.get("metric")
            prefix = (domain.DECISION_FINAL_LABEL_EVAL_PROMPT if m == "decision" else
                      domain.BOOL_FINAL_LABEL_EVAL_PROMPT if m == "bool" else
                      domain.NUMBER_FINAL_LABEL_EVAL_PROMPT if m == "number" else
                      domain.MC_FINAL_LABEL_EVAL_PROMPT)
            sample["prompt"] = prefix + "\n" + sample["prompt"]
        correct = 0; total_time = 0.0; total_calls = 0
        for i, sample in enumerate(samples):
            text = base.chat_prompt(tok, sample["prompt"])
            enc = tok([text], add_special_tokens=False, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(model.device)
            attn = enc["attention_mask"].to(model.device)
            t0 = time.time()
            out, calls = generate_remdm(model, input_ids, attention_mask=attn, steps=args.steps,
                                        gen_length=args.gen_length, eta=args.eta, kind=args.kind,
                                        seed=args.seed * 100003 + i)
            dt = time.time() - t0
            output = tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            pred = parse_for_sample(sample, output)
            ok = pred == sample["gold"]
            correct += int(ok); total_time += dt; total_calls += calls
            rows.append({"task": task, "id": sample["id"], "gold": sample["gold"], "pred": pred,
                         "correct": ok, "output": output, "seconds": dt, "forward_calls": calls,
                         "metric": sample["metric"]})
            print(json.dumps({"task": task, "id": sample["id"], "gold": sample["gold"], "pred": pred, "correct": ok}, ensure_ascii=False), flush=True)
        summary[task] = {"accuracy": correct / len(samples), "n": len(samples),
                         "avg_forward_calls": total_calls / len(samples)}
    payload = {"method": f"remdm_{args.kind}_eta{args.eta}", "args": vars(args), "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    macro = sum(v["accuracy"] for v in summary.values()) / len(summary)
    print(json.dumps({"macro": macro, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
