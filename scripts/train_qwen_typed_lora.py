#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


LETTERS = "ABCDEFGHIJ"

MC_FINAL_LABEL_EVAL_PROMPT = """Think briefly, then end with exactly one line:
Final answer: <letter>

The final answer must be one of the listed option letters only. Do not answer yes, no, or maybe for multiple-choice questions. Do not put any other text after the final answer line.
"""

DECISION_FINAL_LABEL_EVAL_PROMPT = """Think briefly, then end with exactly one line:
Final answer: <yes|no|maybe>

The final answer must be yes, no, or maybe only. Do not put any other text after the final answer line.
"""


def format_mc(question, options, instruction="Answer with the letter only."):
    lines = [question]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    return instruction + "\n\n" + "\n".join(lines) + "\nAnswer:"


def load_exclude_ids(paths):
    excluded = {}
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        for row in payload.get("rows", []):
            excluded.setdefault(row.get("task"), set()).add(str(row.get("id")))
    return excluded


def add_final_label_prefix(sample):
    prefix = DECISION_FINAL_LABEL_EVAL_PROMPT if sample["metric"] == "decision" else MC_FINAL_LABEL_EVAL_PROMPT
    sample["prompt"] = prefix + "\n" + sample["prompt"]
    sample["target"] = f"Final answer: {sample['gold']}"
    return sample


def build_train_samples(limit_per_task, seed, excluded):
    rng = random.Random(seed)
    samples = []

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    count = 0
    for i in idxs:
        ex = ds[i]
        sid = str(i)
        if sid in excluded.get("winogrande", set()):
            continue
        question = ex["sentence"].replace("_", "____")
        gold = "A" if str(ex["answer"]) == "1" else "B"
        prompt = format_mc(
            "Fill the blank with the more plausible referent.\nSentence: " + question,
            [ex["option1"], ex["option2"]],
            "Answer with A or B only.",
        )
        samples.append(add_final_label_prefix({"task": "winogrande", "id": sid, "prompt": prompt, "gold": gold, "metric": "letter"}))
        count += 1
        if count >= limit_per_task:
            break

    ds = load_dataset("tau/commonsense_qa", split="train")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    count = 0
    for i in idxs:
        ex = ds[i]
        sid = str(ex["id"])
        if sid in excluded.get("commonsenseqa", set()):
            continue
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        label_to_letter = {lab: LETTERS[j] for j, lab in enumerate(labels)}
        prompt = format_mc(
            "Question: " + ex["question"],
            texts,
            "Choose the best commonsense answer. Reply with the letter only.",
        )
        samples.append(add_final_label_prefix({"task": "commonsenseqa", "id": sid, "prompt": prompt, "gold": label_to_letter[ex["answerKey"]], "metric": "letter"}))
        count += 1
        if count >= limit_per_task:
            break

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    count = 0
    for i in idxs:
        ex = ds[i]
        sid = str(ex["pubid"])
        if sid in excluded.get("pubmedqa", set()):
            continue
        ctx = "\n".join(ex["context"]["contexts"])
        prompt = (
            "Read the biomedical abstract snippets and answer the question with yes, no, or maybe only.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {ex['question']}\nAnswer:"
        )
        samples.append(add_final_label_prefix({"task": "pubmedqa", "id": sid, "prompt": prompt, "gold": ex["final_decision"], "metric": "decision"}))
        count += 1
        if count >= limit_per_task:
            break

    ds = load_dataset("allenai/sciq", split="train")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    count = 0
    for i in idxs:
        ex = ds[i]
        sid = str(i)
        if sid in excluded.get("sciq", set()):
            continue
        opts = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        local = random.Random(seed * 1000003 + i)
        local.shuffle(opts)
        gold = LETTERS[opts.index(ex["correct_answer"])]
        prompt = format_mc(
            "Question: " + ex["question"],
            opts,
            "Answer this science question. Use the support only if useful, and reply with the letter only.",
        )
        samples.append(add_final_label_prefix({"task": "sciq", "id": sid, "prompt": prompt, "gold": gold, "metric": "letter"}))
        count += 1
        if count >= limit_per_task:
            break

    rng.shuffle(samples)
    return samples


class FinalLabelCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        encoded = []
        for ex in features:
            user = [{"role": "user", "content": ex["prompt"]}]
            prompt_text = self.tokenizer.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
            full_text = prompt_text + ex["target"] + self.tokenizer.eos_token
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_length)
            ids = full["input_ids"]
            labels = ids.copy()
            cutoff = min(len(prompt_ids), len(labels))
            labels[:cutoff] = [-100] * cutoff
            encoded.append({"input_ids": ids, "labels": labels})

        max_len = max(len(x["input_ids"]) for x in encoded)
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attention = [], [], []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/hf/models/Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-per-task", type=int, default=800)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--exclude-json", action="append", default=[])
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    excluded = load_exclude_ids(args.exclude_json)
    samples = build_train_samples(args.limit_per_task, args.seed, excluded)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_manifest.json").write_text(json.dumps({"n": len(samples), "seed": args.seed, "limit_per_task": args.limit_per_task}, indent=2))
    (out / "train_samples_preview.json").write_text(json.dumps(samples[:20], ensure_ascii=False, indent=2))

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
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = Dataset.from_list(samples)
    train_args = TrainingArguments(
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
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=FinalLabelCollator(tokenizer, args.max_length),
    )
    trainer.train()
    trainer.save_model(str(out / "final_adapter"))
    tokenizer.save_pretrained(str(out / "final_adapter"))


if __name__ == "__main__":
    main()
