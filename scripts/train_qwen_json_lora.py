#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


class FinalLabelCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        encoded = []
        for ex in features:
            prompt = ex["prompt"]
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            full_text = prompt_text + ex["target"] + self.tokenizer.eos_token
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_length)
            ids = full["input_ids"]
            labels = ids.copy()
            labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
            encoded.append({"input_ids": ids, "labels": labels})

        max_len = max(len(x["input_ids"]) for x in encoded)
        input_ids, labels, attention = [], [], []
        pad_id = self.tokenizer.pad_token_id
        for x in encoded:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            attention.append([1] * len(x["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/hf/models/Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.train_jsonl)
    random.Random(args.seed).shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for row in rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    (out / "train_manifest.json").write_text(json.dumps({"train_jsonl": args.train_jsonl, "n": len(rows), "counts": counts, "max_steps": args.max_steps}, ensure_ascii=False, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, quantization_config=quant, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out),
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=args.max_steps,
            save_total_limit=1,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
        ),
        train_dataset=Dataset.from_list(rows),
        data_collator=FinalLabelCollator(tokenizer, args.max_length),
    )
    trainer.train()
    trainer.save_model(str(out / "final_adapter"))
    tokenizer.save_pretrained(str(out / "final_adapter"))


if __name__ == "__main__":
    main()
