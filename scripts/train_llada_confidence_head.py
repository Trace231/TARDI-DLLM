#!/usr/bin/env python3
"""Prototype: a learnable CONFIDENCE HEAD co-trained with the TARDI-style LoRA.

Idea (attacks the observability ceiling found with frozen features): instead of
fitting a separate GBDT on frozen features, attach a small MLP head to the LLaDA
hidden state at the answer-label position that predicts P(current label == gold),
and co-train it with the LoRA. Gradients flow into BOTH the head and the LoRA, so
the LoRA is pulled toward representations where correctness is *predictable*.

Loss = LoRA(choice + denoise) + conf_weight * BCE(head, 1[pred==gold]).
Targets use the in-batch gold label (no stale offline counterfactuals -> no off-policy issue).

Core hypothesis test: does the co-trained head predict correctness on held-out
samples better than the frozen-feature ceiling (~0.76)? If yes, co-adaptation is
the key and a head-driven gate can exceed the GBDT gate. If not, ~0.76 is a real
task-level observability ceiling.
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_llada_choice_noise_lora as T  # reuse encode_one / pad_batch / loss helpers


class ConfHead(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


def label_correct_and_hidden(model, head, batch, device):
    """Return (choice_loss, denoise_loss, head_bce, head_logits, correct) using hidden at label_pos."""
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["denoise_labels"].to(device)
    out = model(input_ids, attention_mask=attn, output_hidden_states=True)
    logits = out.logits
    hidden = out.hidden_states[-1]  # [B, L, H]
    choice_loss, choice_probs, _choice_per_ex = T.choice_loss_from_logits(logits, batch)
    denoise_loss, _denoise_per_ex = T.denoise_loss_from_logits(logits, labels)

    head_logits, targets = [], []
    for i in range(input_ids.shape[0]):
        pos = batch["label_pos"][i]
        cand = batch["candidate_ids"][i].to(device)
        cand_logits = logits[i, pos, cand]
        pred_idx = int(torch.argmax(cand_logits).item())
        correct = 1.0 if pred_idx == int(batch["gold_idx"][i].item()) else 0.0
        head_logits.append(head(hidden[i, pos, :].float()))
        targets.append(correct)
    head_logits = torch.stack(head_logits)
    targets = torch.tensor(targets, device=device)
    head_bce = F.binary_cross_entropy_with_logits(head_logits, targets)
    return choice_loss, denoise_loss, head_bce, head_logits.detach(), targets.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--noise-ratios", default="0.15,0.35,0.65,0.85")
    ap.add_argument("--denoise-weight", type=float, default=0.15)
    ap.add_argument("--conf-weight", type=float, default=0.3)
    ap.add_argument("--conf-hidden", type=int, default=256)
    ap.add_argument("--mask-prompt", action="store_true")
    ap.add_argument("--prompt-mask-scale", type=float, default=0.15)
    args = ap.parse_args()
    args.noise_ratios = [float(x) for x in args.noise_ratios.split(",") if x.strip()]
    args.mode = "label"
    args.adaptive_noise = "none"
    args.consistency_weight = 0.0

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rows = T.read_jsonl(args.train_jsonl); rng.shuffle(rows)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                     bias="none", task_type="CAUSAL_LM",
                     target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()])
    model = get_peft_model(model, cfg)
    dim = getattr(model.config, "hidden_size", getattr(model.config, "d_model", 4096))
    head = ConfHead(dim, args.conf_hidden).to("cuda")
    model.train(); head.train()
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
         {"params": head.parameters(), "lr": args.head_lr}])

    usable = [r for r in rows if T.encode_one(tok, r, args, rng) is not None]
    step = accum = cursor = 0; opt.zero_grad(set_to_none=True); log = []
    while step < args.max_steps:
        row = usable[cursor % len(usable)]; cursor += 1
        ex = T.encode_one(tok, row, args, rng)
        if ex is None:
            continue
        batch = T.pad_batch([ex], pad_id)
        cl, dl, hb, hlog, tgt = label_correct_and_hidden(model, head, batch, "cuda")
        loss = cl + args.denoise_weight * dl + args.conf_weight * hb
        (loss / args.grad_accum).backward(); accum += 1
        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); accum = 0; step += 1
            rec = {"step": step, "loss": float(loss), "choice": float(cl), "head_bce": float(hb)}
            log.append(rec)
            if step % 10 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
    model.save_pretrained(str(out_dir / "final_adapter")); tok.save_pretrained(str(out_dir / "final_adapter"))
    torch.save(head.state_dict(), out_dir / "conf_head.pt")
    (out_dir / "head_meta.json").write_text(json.dumps({"dim": dim, "hidden": args.conf_hidden, "args": vars(args)}, ensure_ascii=False, indent=2))
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({"done": True, "out": str(out_dir)}))


if __name__ == "__main__":
    main()
