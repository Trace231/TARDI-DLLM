#!/usr/bin/env python3
"""Held-out test of the co-trained confidence head: does it predict per-sample
correctness on the eval-450 (disjoint from train) better than the frozen-feature
ceiling (~0.76)? Loads the LoRA adapter + conf_head, builds the same masked-label
states as training on the eval samples, reads head confidence vs actual correctness.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_llada_choice_noise_lora as T
import build_balanced_lora_train as B
import eval_domain_shift as domain
from train_llada_confidence_head import ConfHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--adapter-dir", required=True, help="dir with final_adapter/ and conf_head.pt")
    ap.add_argument("--tasks", default="mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--noise-ratios", default="0.15,0.35,0.65,0.85")
    ap.add_argument("--mask-prompt", action="store_true", default=True)
    ap.add_argument("--prompt-mask-scale", type=float, default=0.15)
    ap.add_argument("--max-length", type=int, default=1536)
    args = ap.parse_args()
    args.noise_ratios = [float(x) for x in args.noise_ratios.split(",") if x.strip()]
    rng = random.Random(args.seed)

    meta = json.loads((Path(args.adapter_dir) / "head_meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    base = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, str(Path(args.adapter_dir) / "final_adapter")).to("cuda")
    model.eval()
    head = ConfHead(meta["dim"], meta["hidden"]).to("cuda")
    head.load_state_dict(torch.load(Path(args.adapter_dir) / "conf_head.pt")); head.eval()

    confs, corrects = [], []
    per_task = {}
    with torch.no_grad():
        for task in [t for t in args.tasks.split(",") if t]:
            for raw in domain.build_samples(task, args.limit, args.seed):
                row = B.make_row(raw)
                ex = T.encode_one(tok, row, args, rng)
                if ex is None:
                    continue
                batch = T.pad_batch([ex], pad_id)
                input_ids = batch["input_ids"].to("cuda")
                attn = batch["attention_mask"].to("cuda")
                out = model(input_ids, attention_mask=attn, output_hidden_states=True)
                logits = out.logits
                hidden = out.hidden_states[-1]
                pos = batch["label_pos"][0]
                cand = batch["candidate_ids"][0].to("cuda")
                pred_idx = int(torch.argmax(logits[0, pos, cand]).item())
                correct = int(pred_idx == int(batch["gold_idx"][0].item()))
                conf = float(torch.sigmoid(head(hidden[0, pos, :].float())).item())
                confs.append(conf); corrects.append(correct)
                per_task.setdefault(task, []).append((conf, correct))

    confs = np.array(confs); corrects = np.array(corrects)
    pred_correct = (confs >= 0.5).astype(int)
    acc = float((pred_correct == corrects).mean())
    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(corrects, confs)) if len(set(corrects)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    base_rate = float(corrects.mean())
    maj = float(max(base_rate, 1 - base_rate))
    print(json.dumps({
        "n": len(corrects),
        "head_correctness_pred_acc": round(acc, 3),
        "head_auc": round(auc, 3),
        "actual_correct_rate": round(base_rate, 3),
        "majority_baseline_acc": round(maj, 3),
        "frozen_feature_ceiling": 0.76,
        "verdict": "co-adaptation BEATS frozen ceiling" if acc > 0.78 else "~ at/below frozen ceiling",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
