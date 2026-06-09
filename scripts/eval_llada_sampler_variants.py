import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as domain
import eval_subset as base

MASK_ID = 126336


def infer_label_space(sample):
    if sample.get("metric") == "bool":
        return ["yes", "no"]
    if sample.get("metric") == "decision":
        return ["yes", "no", "maybe"]
    if sample.get("metric") != "letter":
        return []
    labels = []
    for line in sample["prompt"].splitlines():
        import re

        m = re.match(r"\s*([A-J])\s*[\.\)\uff0e\uff09]\s+", line)
        if m and m.group(1) not in labels:
            labels.append(m.group(1))
    return labels or list(domain.LETTERS)


def parse_for_sample(sample, output):
    if sample.get("metric") == "letter":
        labels = infer_label_space(sample)
        return domain.parse_letter(output, choices="".join(labels) if labels else domain.LETTERS)
    if sample.get("metric") == "decision":
        return domain.parse_decision(output)
    if sample.get("metric") == "bool":
        return base.parse_bool(output)
    pred, _ = domain.score(sample["metric"], output, sample["gold"])
    return pred


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def token_budget(mask_index, steps, schedule):
    bsz = mask_index.shape[0]
    mask_num = mask_index.sum(dim=1)
    if schedule == "uniform":
        base_num = mask_num[:, None] // steps
        rem = mask_num[:, None] % steps
        out = torch.zeros((bsz, steps), device=mask_index.device, dtype=torch.long) + base_num
        for i in range(bsz):
            out[i, :rem[i].item()] += 1
        return out
    xs = torch.arange(1, steps + 1, device=mask_index.device, dtype=torch.float32) / steps
    if schedule == "front_loaded":
        weights = torch.flip(xs, dims=[0])
    elif schedule == "back_loaded":
        weights = xs
    elif schedule == "middle_heavy":
        weights = torch.sin(np.pi * xs).clamp_min(1e-4)
    else:
        raise ValueError(schedule)
    weights = weights / weights.sum()
    raw = mask_num[:, None].float() * weights[None, :]
    out = torch.floor(raw).long()
    missing = mask_num - out.sum(dim=1)
    frac = raw - out.float()
    for i in range(bsz):
        if missing[i] > 0:
            _, idx = torch.topk(frac[i], k=int(missing[i].item()))
            out[i, idx] += 1
    return out


def model_logits(model, x, attention_mask, prompt_index, cfg):
    if cfg > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = MASK_ID
        x_ = torch.cat([x, un_x], dim=0)
        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None
        logits = model(x_, attention_mask=attention_mask_).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        return un_logits + (cfg + 1) * (logits - un_logits)
    return model(x, attention_mask=attention_mask).logits


def x0_and_conf(logits, x, mask_index, prompt_len, block_end, temperature, remasking):
    logits_with_noise = add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    x0_p[:, block_end:] = -np.inf
    x0 = torch.where(mask_index, x0, x)
    conf = torch.where(mask_index, x0_p, -np.inf)
    return x0, conf, logits


@torch.no_grad()
def generate_variant(model, prompt, attention_mask=None, steps=32, gen_length=32, block_length=32, temperature=0.0, cfg=0.0, remasking="low_confidence", schedule="uniform", sampler="standard", corrector_weight=0.5):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)
    prompt_index = x != MASK_ID
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    forward_calls = 0
    for nb in range(num_blocks):
        block_start = prompt.shape[1] + nb * block_length
        block_end = prompt.shape[1] + (nb + 1) * block_length
        block_mask = x[:, block_start:block_end] == MASK_ID
        budgets = token_budget(block_mask, steps_per_block, schedule)
        for i in range(steps_per_block):
            mask_index = x == MASK_ID
            logits1 = model_logits(model, x, attention_mask, prompt_index, cfg)
            forward_calls += 1
            x01, conf1, _ = x0_and_conf(logits1, x, mask_index, prompt.shape[1], block_end, temperature, remasking)
            conf = conf1
            x0 = x01

            if sampler == "predictor_corrector":
                # Provisional half-step/full-budget prediction, then correct logits on the provisional state.
                provisional = x.clone()
                provisional_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                for j in range(conf1.shape[0]):
                    k = max(1, int(budgets[j, i].item()))
                    _, idx = torch.topk(conf1[j], k=k)
                    provisional_index[j, idx] = True
                provisional[provisional_index] = x01[provisional_index]
                logits2 = model_logits(model, provisional, attention_mask, prompt_index, cfg)
                forward_calls += 1
                mixed_logits = (1.0 - corrector_weight) * logits1 + corrector_weight * logits2
                x0, conf, _ = x0_and_conf(mixed_logits, x, mask_index, prompt.shape[1], block_end, temperature, remasking)
            elif sampler != "standard":
                raise ValueError(sampler)

            transfer = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            for j in range(conf.shape[0]):
                k = int(budgets[j, i].item())
                if k <= 0:
                    continue
                _, idx = torch.topk(conf[j], k=k)
                transfer[j, idx] = True
            x[transfer] = x0[transfer]
    return x, forward_calls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--tasks", default="winogrande,commonsenseqa")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--gen-length", type=int, default=32)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cfg", type=float, default=0.0)
    p.add_argument("--remasking", default="low_confidence")
    p.add_argument("--schedule", choices=["uniform", "front_loaded", "back_loaded", "middle_heavy"], default="uniform")
    p.add_argument("--sampler", choices=["standard", "predictor_corrector"], default="standard")
    p.add_argument("--corrector-weight", type=float, default=0.5)
    p.add_argument("--prompt-style", choices=["plain", "final_label_typed"], default="final_label_typed")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok, model = base.load_llada(args.model, args.adapter)
    torch.cuda.reset_peak_memory_stats()
    rows=[]; summary={}
    for task in [t for t in args.tasks.split(',') if t]:
        samples = domain.build_samples(task, args.limit, args.seed)
        if args.prompt_style == "final_label_typed":
            for sample in samples:
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
        correct=0; total_time=0.0; total_calls=0
        for sample in samples:
            text = base.chat_prompt(tok, sample["prompt"])
            enc = tok([text], add_special_tokens=False, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(model.device)
            attn = enc["attention_mask"].to(model.device)
            t0=time.time()
            out, calls = generate_variant(model, input_ids, attention_mask=attn, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=args.temperature, cfg=args.cfg, remasking=args.remasking, schedule=args.schedule, sampler=args.sampler, corrector_weight=args.corrector_weight)
            dt=time.time()-t0
            output = tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            pred = parse_for_sample(sample, output)
            ok = pred == sample["gold"]
            correct += int(ok); total_time += dt; total_calls += calls
            row={"task":task,"id":sample["id"],"gold":sample["gold"],"pred":pred,"correct":ok,"output":output,"prompt":sample["prompt"],"seconds":dt,"forward_calls":calls,"metric":sample["metric"]}
            rows.append(row)
            print(json.dumps({k:row[k] for k in ["task","id","gold","pred","correct","forward_calls"]}, ensure_ascii=False), flush=True)
        summary[task]={"accuracy":correct/len(samples),"n":len(samples),"seconds":total_time,"avg_forward_calls":total_calls/len(samples),"peak_mem_gb":torch.cuda.max_memory_allocated()/1024**3}
    payload={"args":vars(args),"summary":summary,"rows":rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
