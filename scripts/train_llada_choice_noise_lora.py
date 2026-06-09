#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from nara_adapter import install_nara_adapter, save_nara_adapter


MASK_ID = 126336
LETTER_RE = re.compile(r"^\s*([A-J])\s*[\.\)\uff0e\uff09]\s+", re.M)


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def message_text(row):
    prompt = row["prompt"]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def user_content(row):
    prompt = row["prompt"]
    if isinstance(prompt, str):
        return prompt
    return "\n".join(str(m.get("content", "")) for m in prompt)


def label_space(row):
    metric = row.get("metric")
    if metric == "decision":
        return ["yes", "no", "maybe"]
    if metric == "bool":
        return ["yes", "no"]
    labels = []
    for label in LETTER_RE.findall(user_content(row)):
        if label not in labels:
            labels.append(label)
    if labels:
        return labels
    answer = str(row.get("answer", row.get("target", ""))).strip()
    if answer and answer[0].upper() in "ABCDEFGHIJ":
        return list("ABCDEFGHIJ")
    return ["A", "B", "C", "D"]


def apply_chat(tokenizer, row):
    try:
        return tokenizer.apply_chat_template(message_text(row), add_generation_prompt=True, tokenize=False)
    except Exception:
        return user_content(row)


def candidate_token_ids(tokenizer, prefix_text, labels, label_pos):
    ids = []
    for label in labels:
        cand = tokenizer(prefix_text + label, add_special_tokens=False)["input_ids"]
        if len(cand) <= label_pos:
            return None
        ids.append(cand[label_pos])
    return ids


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def encode_one(tokenizer, row, args, rng, forced_ratio=None):
    prompt_text = apply_chat(tokenizer, row)
    target = row.get("target") or f"Final answer: {row['answer']}"
    answer = str(row.get("answer", target.split(":")[-1])).strip()
    prefix = "Final answer: "
    if target.startswith(prefix):
        prefix_text = prompt_text + prefix
        full_text = prefix_text + answer
    else:
        marker = target.rfind(answer)
        if marker < 0:
            prefix_text = prompt_text
            full_text = prompt_text + target
        else:
            prefix_text = prompt_text + target[:marker]
            full_text = prompt_text + target

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    label_pos = common_prefix_len(prefix_ids, full_ids)
    if len(full_ids) > args.max_length or label_pos >= len(full_ids):
        return None

    labels = label_space(row)
    cand_ids = candidate_token_ids(tokenizer, prefix_text, labels, label_pos)
    if not cand_ids:
        return None
    gold_label = answer.split()[0].strip()
    if gold_label not in labels:
        if gold_label.upper() in labels:
            gold_label = gold_label.upper()
        else:
            return None
    gold_idx = labels.index(gold_label)

    original = torch.tensor(full_ids, dtype=torch.long)
    input_ids = original.clone()
    denoise_labels = torch.full_like(original, -100)
    ratio = forced_ratio if forced_ratio is not None else rng.choice(args.noise_ratios)
    completion_positions = list(range(label_pos, len(full_ids)))
    masked = []
    for pos in completion_positions:
        if pos == label_pos or rng.random() < ratio:
            masked.append(pos)
    if args.mask_prompt:
        for pos in range(0, label_pos):
            if rng.random() < ratio * args.prompt_mask_scale:
                masked.append(pos)
    for pos in sorted(set(masked)):
        input_ids[pos] = MASK_ID
        denoise_labels[pos] = original[pos]

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "denoise_labels": denoise_labels,
        "label_pos": label_pos,
        "candidate_ids": torch.tensor(cand_ids, dtype=torch.long),
        "gold_idx": gold_idx,
        "noise_ratio": float(ratio),
    }


def pad_batch(examples, pad_id):
    max_len = max(x["input_ids"].numel() for x in examples)
    batch = {"input_ids": [], "attention_mask": [], "denoise_labels": []}
    label_pos, candidate_ids, gold_idx, ratios = [], [], [], []
    for ex in examples:
        n = ex["input_ids"].numel()
        pad = max_len - n
        batch["input_ids"].append(F.pad(ex["input_ids"], (0, pad), value=pad_id))
        batch["attention_mask"].append(F.pad(ex["attention_mask"], (0, pad), value=0))
        batch["denoise_labels"].append(F.pad(ex["denoise_labels"], (0, pad), value=-100))
        label_pos.append(ex["label_pos"])
        candidate_ids.append(ex["candidate_ids"])
        gold_idx.append(ex["gold_idx"])
        ratios.append(ex["noise_ratio"])
    return {
        "input_ids": torch.stack(batch["input_ids"]),
        "attention_mask": torch.stack(batch["attention_mask"]),
        "denoise_labels": torch.stack(batch["denoise_labels"]),
        "label_pos": label_pos,
        "candidate_ids": candidate_ids,
        "gold_idx": torch.tensor(gold_idx, dtype=torch.long),
        "noise_ratio": ratios,
    }


def choice_loss_from_logits(logits, batch):
    losses = []
    probs = []
    for i, pos in enumerate(batch["label_pos"]):
        cand = batch["candidate_ids"][i].to(logits.device)
        cand_logits = logits[i, pos, cand].float()
        target = torch.tensor([batch["gold_idx"][i].item()], device=logits.device)
        losses.append(F.cross_entropy(cand_logits.unsqueeze(0), target))
        probs.append(F.softmax(cand_logits, dim=-1))
    return torch.stack(losses).mean(), probs


def denoise_loss_from_logits(logits, labels):
    mask = labels != -100
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask].float(), labels[mask].to(logits.device))


def forward_losses(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["denoise_labels"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    choice_loss, choice_probs = choice_loss_from_logits(logits, batch)
    denoise_loss = denoise_loss_from_logits(logits, labels)
    return choice_loss, denoise_loss, choice_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["vanilla", "label", "choice_noise"], default="choice_noise")
    ap.add_argument("--peft-variant", choices=["lora", "rslora", "dora", "nara"], default="lora")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--nara-buckets", type=int, default=4)
    ap.add_argument("--noise-ratios", default="0.15,0.35,0.65,0.85")
    ap.add_argument("--denoise-weight", type=float, default=0.15)
    ap.add_argument("--consistency-weight", type=float, default=0.05)
    ap.add_argument("--mask-prompt", action="store_true")
    ap.add_argument("--prompt-mask-scale", type=float, default=0.15)
    ap.add_argument("--save-every", type=int, default=0)
    args = ap.parse_args()
    args.noise_ratios = [float(x) for x in args.noise_ratios.split(",") if x.strip()]

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.train_jsonl)
    rng.shuffle(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except ValueError as exc:
            print(f"gradient_checkpointing skipped: {exc}", flush=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.peft_variant == "nara":
        model = install_nara_adapter(
            model,
            target_modules=target_modules,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            num_buckets=args.nara_buckets,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.4f}", flush=True)
    else:
        lora_kwargs = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": target_modules,
        }
        if args.peft_variant == "rslora":
            lora_kwargs["use_rslora"] = True
        if args.peft_variant == "dora":
            lora_kwargs["use_dora"] = True
        try:
            config = LoraConfig(**lora_kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Installed PEFT does not support --peft-variant {args.peft_variant}. "
                f"Original error: {exc}"
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    usable = [r for r in rows if encode_one(tokenizer, r, args, rng) is not None]
    if not usable:
        raise RuntimeError("No usable training examples after tokenization.")
    (out_dir / "train_manifest.json").write_text(
        json.dumps(
            {
                "args": {k: (v if k != "noise_ratios" else list(v)) for k, v in vars(args).items()},
                "raw_n": len(rows),
                "usable_n": len(usable),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    step = 0
    accum = 0
    cursor = 0
    running = []
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        batch_rows = [usable[(cursor + j) % len(usable)] for j in range(args.batch_size)]
        cursor += args.batch_size
        examples = []
        for row in batch_rows:
            ex = encode_one(tokenizer, row, args, rng)
            if ex is not None:
                examples.append(ex)
        if not examples:
            continue
        batch = pad_batch(examples, pad_id)
        choice_loss, denoise_loss, probs1 = forward_losses(model, batch, "cuda")
        if args.mode == "vanilla":
            loss = denoise_loss
            consistency_loss = denoise_loss * 0.0
        elif args.mode == "label":
            consistency_loss = denoise_loss * 0.0
            loss = choice_loss + args.denoise_weight * denoise_loss
        else:
            alt_examples = []
            for row in batch_rows:
                ratio = rng.choice(args.noise_ratios)
                alt = encode_one(tokenizer, row, args, rng, forced_ratio=ratio)
                if alt is not None:
                    alt_examples.append(alt)
            alt_batch = pad_batch(alt_examples, pad_id)
            alt_choice_loss, alt_denoise_loss, probs2 = forward_losses(model, alt_batch, "cuda")
            kl_terms = []
            for p, q in zip(probs1, probs2):
                p = p.float()
                q = q.float()
                kl_terms.append(F.kl_div(q.log(), p.detach(), reduction="batchmean"))
            consistency_loss = torch.stack(kl_terms).mean() if kl_terms else choice_loss * 0.0
            choice_loss = 0.5 * (choice_loss + alt_choice_loss)
            denoise_loss = 0.5 * (denoise_loss + alt_denoise_loss)
            loss = choice_loss + args.denoise_weight * denoise_loss + args.consistency_weight * consistency_loss

        (loss / args.grad_accum).backward()
        accum += 1
        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            rec = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "choice_loss": float(choice_loss.detach().cpu()),
                "denoise_loss": float(denoise_loss.detach().cpu()),
                "consistency_loss": float(consistency_loss.detach().cpu()),
            }
            running.append(rec)
            if step % 10 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
            if args.save_every and step % args.save_every == 0:
                if args.peft_variant == "nara":
                    save_nara_adapter(model, out_dir / f"checkpoint-{step}", tokenizer=tokenizer)
                else:
                    model.save_pretrained(str(out_dir / f"checkpoint-{step}"))

    if args.peft_variant == "nara":
        save_nara_adapter(model, out_dir / "final_adapter", tokenizer=tokenizer)
    else:
        model.save_pretrained(str(out_dir / "final_adapter"))
        tokenizer.save_pretrained(str(out_dir / "final_adapter"))
    (out_dir / "train_log.json").write_text(json.dumps(running, indent=2))


if __name__ == "__main__":
    main()
