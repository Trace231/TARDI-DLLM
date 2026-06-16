#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from nara_adapter import install_nara_adapter, save_nara_adapter, set_nara_task_batch


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
    if getattr(args, "noise_jitter", 0.0) > 0:  # CONTINUOUS noise: jitter the (bucketed) ratio into a smooth range
        ratio = min(0.97, max(0.05, ratio + rng.uniform(-args.noise_jitter, args.noise_jitter)))
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
        "task": str(row.get("task", "unknown")),
    }


def pad_batch(examples, pad_id):
    max_len = max(x["input_ids"].numel() for x in examples)
    batch = {"input_ids": [], "attention_mask": [], "denoise_labels": []}
    label_pos, candidate_ids, gold_idx, ratios, tasks = [], [], [], [], []
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
        tasks.append(ex["task"])
    return {
        "input_ids": torch.stack(batch["input_ids"]),
        "attention_mask": torch.stack(batch["attention_mask"]),
        "denoise_labels": torch.stack(batch["denoise_labels"]),
        "label_pos": label_pos,
        "candidate_ids": candidate_ids,
        "gold_idx": torch.tensor(gold_idx, dtype=torch.long),
        "noise_ratio": ratios,
        "task": tasks,
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
    stacked = torch.stack(losses)
    # 3rd return keeps GRAD (loss-weighting backprops through per-example choice loss); EMA-signal
    # callers detach it themselves.
    return stacked.mean(), probs, stacked


def denoise_loss_from_logits(logits, labels):
    mask = labels != -100
    if not bool(mask.any()):
        z = logits.sum() * 0.0
        return z, torch.zeros(logits.shape[0], device=logits.device)
    flat = F.cross_entropy(logits[mask].float(), labels[mask].to(logits.device))
    # per-example reconstruction CE = "how hard is denoising at THIS example's noise ratio rho":
    # the on-target signal for choosing which noise bucket to upsample (vs choice CE = answer difficulty).
    per_ex = []
    for i in range(logits.shape[0]):
        mi = mask[i]
        if bool(mi.any()):
            per_ex.append(F.cross_entropy(logits[i][mi].float(), labels[i][mi].to(logits.device)).detach())
        else:
            per_ex.append((logits[i].sum() * 0.0).detach())
    return flat, torch.stack(per_ex)


def forward_losses(model, batch, device):
    set_nara_task_batch(model, batch.get("task", []))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["denoise_labels"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    choice_loss, choice_probs, choice_per_ex = choice_loss_from_logits(logits, batch)
    denoise_loss, denoise_per_ex = denoise_loss_from_logits(logits, labels)
    return choice_loss, denoise_loss, choice_probs, choice_per_ex, denoise_per_ex


@torch.no_grad()
def val_choice_accuracy(model, tokenizer, val_rows, args, rng, pad_id, device):
    """Cheap validation signal: one forward per held-out row (fixed masking), choice argmax vs gold.
    Used to pick the eval-optimal checkpoint (early stop on VALIDATION, not train loss -> catches overfit)."""
    was_training = model.training
    model.eval()
    correct = total = 0
    for r in val_rows:
        ex = encode_one(tokenizer, r, args, rng, forced_ratio=0.5)  # fixed ratio -> consistent val signal
        if ex is None:
            continue
        batch = pad_batch([ex], pad_id)
        out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        pos = batch["label_pos"][0]
        cand = batch["candidate_ids"][0].to(device)
        pred = int(torch.argmax(out.logits[0, pos, cand]).item())
        correct += int(pred == int(batch["gold_idx"][0].item()))
        total += 1
    if was_training:
        model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["vanilla", "label", "choice_noise"], default="choice_noise")
    ap.add_argument("--peft-variant", choices=["lora", "rslora", "dora", "loraplus", "nara", "tasknara"], default="lora")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loraplus-lr-ratio", type=float, default=16.0)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--nara-buckets", type=int, default=4)
    ap.add_argument("--nara-c-scale", type=float, default=0.1)
    ap.add_argument("--nara-embedding-dim", type=int, default=64)
    ap.add_argument("--nara-hidden1", type=int, default=256)
    ap.add_argument("--nara-hidden2", type=int, default=512)
    ap.add_argument("--task-list", default="")
    ap.add_argument("--task-embedding-dim", type=int, default=32)
    ap.add_argument("--task-conditioning", choices=["concat", "residual"], default="concat")
    ap.add_argument("--task-residual-scale", type=float, default=0.05)
    ap.add_argument("--task-dropout", type=float, default=0.0)
    ap.add_argument("--noise-ratios", default="0.15,0.35,0.65,0.85")
    ap.add_argument("--denoise-weight", type=float, default=0.15)
    ap.add_argument("--consistency-weight", type=float, default=0.05)
    ap.add_argument("--mask-prompt", action="store_true")
    ap.add_argument("--prompt-mask-scale", type=float, default=0.15)
    # --- option C: online loss-aware adaptive noise (per-task x per-bucket loss EMA) ---
    ap.add_argument("--adaptive-noise", choices=["none", "loss_aware", "reducible_loss"], default="none")
    ap.add_argument("--adaptive-eps", type=float, default=0.2, help="exploration floor (uniform mix)")
    ap.add_argument("--adaptive-temp", type=float, default=0.5, help="softmax temperature")
    ap.add_argument("--adaptive-ema", type=float, default=0.9, help="slow EMA decay for per-(task,bucket) loss")
    ap.add_argument("--adaptive-fast-ema", type=float, default=0.6, help="fast EMA decay (reducible_loss: sample ~ slow-fast)")
    ap.add_argument("--adaptive-signal", choices=["choice_ppl", "denoise_ppl", "mix"], default="denoise_ppl",
                    help="which per-example perplexity feeds the EMA / loss weight: denoise_ppl = "
                         "reconstruction difficulty at that noise level (on-target); choice_ppl = answer difficulty")
    ap.add_argument("--adaptive-where", choices=["sampling", "loss_weight", "both"], default="sampling",
                    help="sampling = bandit upsamples hard noise buckets (dynamic noise schedule); "
                         "loss_weight = uniform sampling, focal-weight each example's choice loss by its own "
                         "fresh perplexity; both = dynamic-noise sampling AND loss weighting (PPL on both ends)")
    ap.add_argument("--adaptive-gamma", type=float, default=1.0,
                    help="loss_weight sharpness: w = normalize(PPL^gamma); 0 -> uniform (recovers baseline)")
    ap.add_argument("--adaptive-wclip", type=float, default=4.0, help="loss_weight: clamp normalized weight max")
    ap.add_argument("--save-every", type=int, default=0)
    # --- proper training strategy: LR schedule + convergence-based stopping (vs toy constant-LR/fixed-steps) ---
    ap.add_argument("--lr-scheduler", choices=["none", "cosine", "linear"], default="none",
                    help="cosine/linear LR decay with warmup; 'none' = constant LR (the old toy default)")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--convergence-window", type=int, default=50, help="steps per convergence-check window")
    ap.add_argument("--convergence-patience", type=int, default=0,
                    help=">0 enables EARLY STOP: stop once this many consecutive windows show EMA-loss "
                         "relative improvement < --convergence-tol (i.e. converged). max-steps becomes an upper bound.")
    ap.add_argument("--convergence-tol", type=float, default=0.01)
    # --- proper model selection: hold-out validation -> keep the BEST-val checkpoint (catches overfit) ---
    ap.add_argument("--val-fraction", type=float, default=0.0, help=">0 holds out this fraction of train as val")
    ap.add_argument("--val-every", type=int, default=50, help="evaluate val choice-accuracy every N steps")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (regularization)")
    ap.add_argument("--noise-jitter", type=float, default=0.0,
                    help=">0 turns the discrete noise buckets into a CONTINUOUS schedule by jittering each "
                         "sampled ratio uniformly by +/- this amount (clipped to [0.05, 0.97])")
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
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    if args.peft_variant in {"nara", "tasknara"}:
        task_list = [x.strip() for x in args.task_list.split(",") if x.strip()]
        if args.peft_variant == "tasknara" and not task_list:
            task_list = sorted({str(r.get("task", "unknown")) for r in rows})
        model = install_nara_adapter(
            model,
            target_modules=target_modules,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            num_buckets=args.nara_buckets,
            embedding_dim=args.nara_embedding_dim,
            mapper_hidden1=args.nara_hidden1,
            mapper_hidden2=args.nara_hidden2,
            c_scale=args.nara_c_scale,
            task_list=task_list if args.peft_variant == "tasknara" else None,
            task_embedding_dim=args.task_embedding_dim if args.peft_variant == "tasknara" else 0,
            task_conditioning=args.task_conditioning if args.peft_variant == "tasknara" else "concat",
            task_residual_scale=args.task_residual_scale,
            task_dropout=args.task_dropout if args.peft_variant == "tasknara" else 0.0,
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
    if args.peft_variant == "loraplus":
        group_a, group_b, group_other = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name:
                group_a.append(param)
            elif "lora_B" in name:
                group_b.append(param)
            else:
                group_other.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": group_a, "lr": args.lr},
                {"params": group_b, "lr": args.lr * args.loraplus_lr_ratio},
                {"params": group_other, "lr": args.lr},
            ]
        )
        print(
            f"LoRA+ optimizer: lr_A={args.lr:g}, lr_B={args.lr * args.loraplus_lr_ratio:g}, "
            f"n_A={sum(p.numel() for p in group_a):,}, n_B={sum(p.numel() for p in group_b):,}",
            flush=True,
        )
    else:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                                       weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler != "none":
        from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
        warm = max(1, int(args.warmup_ratio * args.max_steps))
        _build = get_cosine_schedule_with_warmup if args.lr_scheduler == "cosine" else get_linear_schedule_with_warmup
        scheduler = _build(optimizer, num_warmup_steps=warm, num_training_steps=args.max_steps)
        print(f"LR schedule: {args.lr_scheduler}, warmup={warm}/{args.max_steps}", flush=True)

    usable = [r for r in rows if encode_one(tokenizer, r, args, rng) is not None]
    if not usable:
        raise RuntimeError("No usable training examples after tokenization.")
    val_rows = []
    if args.val_fraction > 0:
        n_val = max(1, int(args.val_fraction * len(usable)))
        val_rows = usable[:n_val]          # held out, NOT trained on
        usable = usable[n_val:]
        print(f"held-out val: {len(val_rows)} rows; train: {len(usable)} rows", flush=True)
    best_val = -1.0
    best_val_step = 0
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

    # --- option C: online loss-aware adaptive noise sampler (per-task x per-bucket loss EMA) ---
    noise_buckets = list(args.noise_ratios)
    ema_loss = {}   # task -> [slow EMA loss per bucket]
    ema_fast = {}   # task -> [fast EMA loss per bucket]
    ema_seen = {}   # task -> [visit count per bucket]

    def _bucket_logit(task, i):
        # loss_aware: sample ~ raw loss EMA (high loss -> more samples).
        # reducible_loss: sample ~ (slow - fast) i.e. loss still DROPPING (learnable) -> more samples;
        #   high-but-flat (irreducible, e.g. mmlu knowledge gap) -> ~0 -> falls back to exploration floor.
        slow = ema_loss[task][i]
        if args.adaptive_noise == "reducible_loss":
            fast = ema_fast[task][i]
            return (slow - fast) if (slow is not None and fast is not None) else 0.0
        return slow if slow is not None else 0.0

    def adaptive_sample(task):
        """Return (bucket_idx, ratio): sample bucket ~ softmax(logit/temp) with eps floor; unseen first."""
        B = len(noise_buckets)
        ema_loss.setdefault(task, [None] * B)
        ema_fast.setdefault(task, [None] * B)
        seen = ema_seen.setdefault(task, [0] * B)
        unseen = [i for i in range(B) if seen[i] == 0]
        if unseen:
            j = rng.choice(unseen)
        else:
            logits = [_bucket_logit(task, i) for i in range(B)]
            m = max(logits)
            ws = [math.exp((v - m) / max(1e-6, args.adaptive_temp)) for v in logits]
            s = sum(ws)
            probs = [(1.0 - args.adaptive_eps) * (w / s) + args.adaptive_eps / B for w in ws]
            r = rng.random()
            acc = 0.0
            j = B - 1
            for i, p in enumerate(probs):
                acc += p
                if r <= acc:
                    j = i
                    break
        return j, noise_buckets[j]

    def adaptive_update(task, j, loss_val):
        s = ema_loss[task]
        s[j] = loss_val if s[j] is None else args.adaptive_ema * s[j] + (1.0 - args.adaptive_ema) * loss_val
        f = ema_fast[task]
        f[j] = loss_val if f[j] is None else args.adaptive_fast_ema * f[j] + (1.0 - args.adaptive_fast_ema) * loss_val
        ema_seen[task][j] += 1

    step = 0
    accum = 0
    cursor = 0
    running = []
    loss_ema = None          # smoothed loss for convergence detection (choice_loss is too noisy to use raw)
    conv_marks = []          # EMA-loss snapshot per window
    no_progress = 0
    stopped_converged = False
    optimizer.zero_grad(set_to_none=True)
    use_bandit = (args.adaptive_noise != "none" and args.adaptive_where in ("sampling", "both"))
    use_lossw = (args.adaptive_noise != "none" and args.adaptive_where in ("loss_weight", "both"))
    while step < args.max_steps:
        batch_rows = [usable[(cursor + j) % len(usable)] for j in range(args.batch_size)]
        cursor += args.batch_size
        examples = []
        ex_keys = []  # aligned 1:1 with examples (only non-None rows), for per-example bucket attribution
        for row in batch_rows:
            if use_bandit:
                j, ratio = adaptive_sample(str(row.get("task", "unknown")))
                ex = encode_one(tokenizer, row, args, rng, forced_ratio=ratio)
                key = (str(row.get("task", "unknown")), j)
            else:
                ex = encode_one(tokenizer, row, args, rng)  # uniform draw (baseline / loss_weight)
                key = None
            if ex is not None:
                examples.append(ex)
                ex_keys.append(key)
        if not examples:
            continue
        batch = pad_batch(examples, pad_id)
        choice_loss, denoise_loss, probs1, choice_per_ex, denoise_per_ex = forward_losses(model, batch, "cuda")

        # which per-example perplexity is the signal (used by BOTH sampling EMA and loss weighting)
        if args.adaptive_signal == "denoise_ppl":
            sig_t = denoise_per_ex
        elif args.adaptive_signal == "mix":
            sig_t = 0.5 * choice_per_ex.detach() + 0.5 * denoise_per_ex
        else:
            sig_t = choice_per_ex.detach()

        if use_bandit:
            # PER-EXAMPLE attribution: each bucket's EMA updated with ITS OWN example's perplexity, not the
            # batch mean (the old code gave every bucket the batch mean -> contaminated signal).
            sig = sig_t.detach().cpu().tolist()
            for _idx, _k in enumerate(ex_keys):
                if _k is not None and _idx < len(sig):
                    adaptive_update(_k[0], _k[1], float(sig[_idx]))

        if args.mode == "vanilla":
            loss = denoise_loss
            consistency_loss = denoise_loss * 0.0
        elif args.mode == "label":
            consistency_loss = denoise_loss * 0.0
            if use_lossw:
                # FOCAL-style: weight each example's choice loss by its own FRESH perplexity (no staleness,
                # no chicken-and-egg -- PPL is a byproduct of the forward we already ran). Normalize to mean 1
                # (keeps loss scale, bounds variance) and clamp outliers.
                w = sig_t.detach().clamp(min=0.0) ** args.adaptive_gamma
                w = w / (w.mean() + 1e-6)
                w = w.clamp(max=args.adaptive_wclip)
                weighted_choice = (w * choice_per_ex).mean()
                loss = weighted_choice + args.denoise_weight * denoise_loss
            else:
                loss = choice_loss + args.denoise_weight * denoise_loss
        else:
            alt_examples = []
            for row in batch_rows:
                ratio = rng.choice(args.noise_ratios)
                alt = encode_one(tokenizer, row, args, rng, forced_ratio=ratio)
                if alt is not None:
                    alt_examples.append(alt)
            alt_batch = pad_batch(alt_examples, pad_id)
            alt_choice_loss, alt_denoise_loss, probs2, _alt_per_ex, _alt_dn_per_ex = forward_losses(model, alt_batch, "cuda")
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
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            _lv = float(loss.detach().cpu())
            loss_ema = _lv if loss_ema is None else 0.97 * loss_ema + 0.03 * _lv
            rec = {
                "step": step,
                "loss": _lv,
                "loss_ema": loss_ema,
                "lr": (scheduler.get_last_lr()[0] if scheduler is not None else args.lr),
                "choice_loss": float(choice_loss.detach().cpu()),
                "denoise_loss": float(denoise_loss.detach().cpu()),
                "consistency_loss": float(consistency_loss.detach().cpu()),
            }
            running.append(rec)
            if step % 10 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
            # convergence-based early stop: stop once the smoothed loss plateaus (vs toy fixed-step stop)
            if args.convergence_patience > 0 and step % args.convergence_window == 0:
                conv_marks.append(loss_ema)
                if len(conv_marks) >= 2:
                    rel = (conv_marks[-2] - conv_marks[-1]) / max(abs(conv_marks[-2]), 1e-6)
                    no_progress = no_progress + 1 if rel < args.convergence_tol else 0
                    if no_progress >= args.convergence_patience:
                        print(json.dumps({"converged": True, "step": step, "loss_ema": loss_ema,
                                          "windows_flat": no_progress}), flush=True)
                        stopped_converged = True
                        break
            # VALIDATION-based model selection: keep the BEST-val checkpoint as final_adapter (catches overfit,
            # unlike train-loss stopping). Fixed budget runs to the end; we just select the best point on it.
            if val_rows and step % args.val_every == 0:
                vacc = val_choice_accuracy(model, tokenizer, val_rows, args, rng, pad_id, "cuda")
                print(json.dumps({"step": step, "val_acc": round(vacc, 4),
                                  "best_val": round(max(best_val, vacc), 4)}), flush=True)
                if vacc > best_val:
                    best_val, best_val_step = vacc, step
                    model.save_pretrained(str(out_dir / "final_adapter"))
                    tokenizer.save_pretrained(str(out_dir / "final_adapter"))
            if args.save_every and step % args.save_every == 0:
                if args.peft_variant in {"nara", "tasknara"}:
                    save_nara_adapter(model, out_dir / f"checkpoint-{step}", tokenizer=tokenizer)
                else:
                    model.save_pretrained(str(out_dir / f"checkpoint-{step}"))

    if val_rows and best_val_step > 0:
        # final_adapter already holds the BEST-val checkpoint; do NOT overwrite with the (possibly overfit) last step
        print(json.dumps({"selected_by_val": True, "best_val": round(best_val, 4), "best_val_step": best_val_step}), flush=True)
        tokenizer.save_pretrained(str(out_dir / "final_adapter"))
    elif args.peft_variant in {"nara", "tasknara"}:
        save_nara_adapter(model, out_dir / "final_adapter", tokenizer=tokenizer)
    else:
        model.save_pretrained(str(out_dir / "final_adapter"))
        tokenizer.save_pretrained(str(out_dir / "final_adapter"))
    (out_dir / "train_log.json").write_text(json.dumps(running, indent=2))
    if args.adaptive_noise != "none":
        (out_dir / "adaptive_noise_ema.json").write_text(
            json.dumps({"buckets": noise_buckets, "ema_loss": ema_loss, "ema_fast": ema_fast, "ema_seen": ema_seen},
                       ensure_ascii=False, indent=2)
        )


if __name__ == "__main__":
    main()
