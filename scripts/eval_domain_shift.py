import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_subset as base
try:
    from nara_adapter import set_nara_task_batch
except Exception:
    set_nara_task_batch = None

LETTERS = "ABCDEFGHIJ"

MC_FINAL_LABEL_EVAL_PROMPT = """Think briefly, then end with exactly one line:
Final answer: <letter>

The final answer must be one of the listed option letters only. Do not answer yes, no, or maybe for multiple-choice questions. Do not put any other text after the final answer line.
"""

DECISION_FINAL_LABEL_EVAL_PROMPT = """Think briefly, then end with exactly one line:
Final answer: <yes|no|maybe>

The final answer must be yes, no, or maybe only. Do not put any other text after the final answer line.
"""

FINAL_LABEL_EVAL_PROMPT = """Think briefly, then end with exactly one line:
Final answer: <label>

The label must be one of A-J, yes, no, or maybe. Do not put any other text after the final answer line.
"""

BOOL_FINAL_LABEL_EVAL_PROMPT = """Think briefly, then end with exactly one line:
Final answer: <yes|no>

The final answer must be yes or no only. Do not put any other text after the final answer line.
"""

NUMBER_FINAL_LABEL_EVAL_PROMPT = """Think briefly, then end with exactly one line:
Final answer: <number>

The final answer must be a single number. Do not put any other text after the final answer line.
"""

SPAN_FINAL_LABEL_EVAL_PROMPT = """Think briefly, then end with exactly one line:
Final answer: <short span>

The final answer must be a short span or number. Do not put any other text after the final answer line.
"""

DOMAIN_LABEL_EVAL_PROMPT = """Respond in the following format:
<reasoning>
Your concise reasoning
</reasoning>
<answer>
ONE_LABEL_ONLY
</answer>

Important: the <answer> block must contain exactly one label: A-J, yes, no, or maybe. Do not include words, punctuation, explanations, or extra labels inside <answer>.
"""


def norm_text(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def parse_xml_answer(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", str(text), flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?i)final\s*answer\s*[:：]\s*([A-J]|yes|no|maybe)\b", str(text))
    if m:
        return m.group(1).strip()
    return str(text).strip()


def parse_letter(text, choices=LETTERS):
    text = parse_xml_answer(text)
    patterns = [
        rf"(?i)final\s*answer\s*[:：]?\s*\(?([{choices}])\)?(?:\s|$|[\.。),;:])",
        rf"(?i)answer\s*[:：]?\s*\(?([{choices}])\)?(?:\s|$|[\.。),;:])",
        rf"(?i)^\s*\(?([{choices}])\)?(?:\s*$|[\s\.。),;:])",
        rf"(?i)\b([{choices}])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    return ""


def parse_decision(text):
    low = parse_xml_answer(text).lower()
    hits = []
    for label in ["yes", "no", "maybe"]:
        m = re.search(rf"\b{label}\b", low)
        if m:
            hits.append((m.start(), label))
    if not hits:
        return ""
    return sorted(hits)[0][1]


def score(metric, output, gold):
    if metric == "letter":
        pred = parse_letter(output)
        return pred, pred == gold
    if metric == "decision":
        pred = parse_decision(output)
        return pred, pred == gold
    if metric == "bool":
        pred = base.parse_bool(output)
        return pred, pred == gold
    if metric == "number":
        pred = base.normalize_num(output)
        return pred, base.numeric_equal(pred, gold)
    if metric == "span_contains":
        pred = norm_text(output)
        golds = gold if isinstance(gold, list) else [gold]
        ok = any(norm_text(g) and norm_text(g) in pred for g in golds)
        return output.strip()[:120], ok
    pred = output.strip()
    return pred, pred == gold


def format_mc(question, options, instruction="Answer with the letter only."):
    lines = [question]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    return instruction + "\n\n" + "\n".join(lines) + "\nAnswer:"


def build_samples(task, limit, seed):
    rng = random.Random(seed)
    samples = []
    if task == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            options = list(ex["options"])
            prompt = format_mc(
                "Question: " + ex["question"],
                options,
                "Solve this professional multiple-choice question. Check each option, then answer with the letter only.",
            )
            samples.append({"id": str(ex.get("question_id", i)), "task": task, "prompt": prompt, "gold": ex["answer"], "metric": "letter", "meta": {"category": ex.get("category"), "src": ex.get("src")}})
    elif task == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            ctx = "\n".join(ex["context"]["contexts"])
            prompt = (
                "Read the biomedical abstract snippets and answer the question with yes, no, or maybe only.\n\n"
                f"Context:\n{ctx}\n\nQuestion: {ex['question']}\nAnswer:"
            )
            samples.append({"id": str(ex["pubid"]), "task": task, "prompt": prompt, "gold": ex["final_decision"], "metric": "decision", "meta": {}})
    elif task == "ceval_computer_network":
        ds = load_dataset("ceval/ceval-exam", "computer_network", split="test")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            options = [ex[x] for x in "ABCD"]
            prompt = format_mc(
                "题目：" + ex["question"],
                options,
                "回答下面的中文计算机网络单选题。只输出选项字母。",
            )
            samples.append({"id": str(ex["id"]), "task": task, "prompt": prompt, "gold": ex["answer"], "metric": "letter", "meta": {}})
    elif task == "sciq":
        ds = load_dataset("allenai/sciq", split="test")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            opts = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
            local = random.Random(seed * 1000003 + i)
            local.shuffle(opts)
            gold = LETTERS[opts.index(ex["correct_answer"])]
            prompt = format_mc(
                "Question: " + ex["question"],
                opts,
                "Answer this science question. Use the support only if useful, and reply with the letter only.",
            )
            samples.append({"id": str(i), "task": task, "prompt": prompt, "gold": gold, "metric": "letter", "meta": {"gold_text": ex["correct_answer"]}})
    elif task == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            question = ex["sentence"].replace("_", "____")
            options = [ex["option1"], ex["option2"]]
            gold = "A" if str(ex["answer"]) == "1" else "B"
            prompt = format_mc(
                "Fill the blank with the more plausible referent.\nSentence: " + question,
                options,
                "Answer with A or B only.",
            )
            samples.append({"id": str(i), "task": task, "prompt": prompt, "gold": gold, "metric": "letter", "meta": {}})
    elif task == "commonsenseqa":
        ds = load_dataset("tau/commonsense_qa", split="validation")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            label_to_letter = {lab: LETTERS[j] for j, lab in enumerate(labels)}
            prompt = format_mc(
                "Question: " + ex["question"],
                texts,
                "Choose the best commonsense answer. Reply with the letter only.",
            )
            samples.append({"id": ex["id"], "task": task, "prompt": prompt, "gold": label_to_letter[ex["answerKey"]], "metric": "letter", "meta": {"concept": ex.get("question_concept")}})
    elif task == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            label_to_letter = {lab: LETTERS[j] for j, lab in enumerate(labels)}
            gold = label_to_letter.get(ex["answerKey"], ex["answerKey"])
            prompt = format_mc(
                "Question: " + ex["question"],
                texts,
                "Answer this grade-school science multiple-choice question. Reply with the letter only.",
            )
            samples.append({"id": str(i), "task": task, "prompt": prompt, "gold": gold, "metric": "letter", "meta": {}})
    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            gold = LETTERS[int(ex["label"])]
            prompt = format_mc(
                "Context: " + ex["ctx"],
                list(ex["endings"]),
                "Choose the most plausible continuation. Reply with the letter only.",
            )
            samples.append({"id": str(i), "task": task, "prompt": prompt, "gold": gold, "metric": "letter", "meta": {"activity_label": ex.get("activity_label")}})
    elif task == "boolq":
        ds = load_dataset("google/boolq", split="validation")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            gold = "yes" if ex["answer"] else "no"
            prompt = (
                "Read the passage and answer the question with yes or no only.\n\n"
                f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}\nAnswer:"
            )
            samples.append({"id": str(i), "task": task, "prompt": prompt, "gold": gold, "metric": "bool", "meta": {}})
    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            gold = base.normalize_num(ex["answer"].split("####")[-1])
            prompt = (
                "Solve the math problem. Show brief reasoning, then end with 'Final answer: <number>'.\n\n"
                f"Problem: {ex['question']}"
            )
            samples.append({"id": str(i), "task": task, "prompt": prompt, "gold": gold, "metric": "number", "meta": {}})
    elif task == "drop_span":
        ds = load_dataset("ucinlp/drop", split="validation")
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            golds = ex["answers_spans"]["spans"]
            prompt = (
                "Read the passage and answer the question with a short span or number. End with 'Final answer: ...'.\n\n"
                f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}\nAnswer:"
            )
            samples.append({"id": ex["query_id"], "task": task, "prompt": prompt, "gold": golds, "metric": "span_contains", "meta": {}})
    else:
        raise ValueError(task)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llada", "ar"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default="mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen-length", type=int, default=96)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--logits-eos-inf", action="store_true")
    ap.add_argument("--confidence-eos-eot-inf", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--prompt-style", choices=["plain", "xml_label", "final_label", "final_label_typed"], default="plain")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok, model = base.load_llada(args.model, args.adapter) if args.backend == "llada" else base.load_ar(args.model, args.adapter)
    rows, summary = [], {}
    for task in [t for t in args.tasks.split(",") if t]:
        samples = build_samples(task, args.limit, args.seed)
        if args.prompt_style == "xml_label":
            for sample in samples:
                sample["prompt"] = DOMAIN_LABEL_EVAL_PROMPT + "\n" + sample["prompt"]
        elif args.prompt_style == "final_label":
            for sample in samples:
                sample["prompt"] = FINAL_LABEL_EVAL_PROMPT + "\n" + sample["prompt"]
        elif args.prompt_style == "final_label_typed":
            for sample in samples:
                if sample.get("metric") == "decision":
                    prefix = DECISION_FINAL_LABEL_EVAL_PROMPT
                elif sample.get("metric") == "bool":
                    prefix = BOOL_FINAL_LABEL_EVAL_PROMPT
                elif sample.get("metric") == "number":
                    prefix = NUMBER_FINAL_LABEL_EVAL_PROMPT
                elif sample.get("metric") == "span_contains":
                    prefix = SPAN_FINAL_LABEL_EVAL_PROMPT
                else:
                    prefix = MC_FINAL_LABEL_EVAL_PROMPT
                sample["prompt"] = prefix + "\n" + sample["prompt"]
        correct, total_time, max_mem = 0, 0.0, 0.0
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            prompts = [x["prompt"] for x in batch]
            if args.backend == "llada":
                if set_nara_task_batch is not None:
                    set_nara_task_batch(model, [x["task"] for x in batch])
                outs, dt, mem = base.generate_llada(tok, model, prompts, args)
            else:
                outs, dt, mem = base.generate_ar(tok, model, prompts, args)
            total_time += dt
            max_mem = max(max_mem, mem)
            for ex, out in zip(batch, outs):
                pred, ok = score(ex["metric"], out, ex["gold"])
                correct += int(ok)
                row = dict(task=task, id=ex["id"], gold=ex["gold"], pred=pred, correct=ok, output=out, prompt=ex["prompt"], metric=ex["metric"], meta=ex.get("meta", {}))
                rows.append(row)
                print(json.dumps({k: row[k] for k in ["task", "id", "gold", "pred", "correct"]}, ensure_ascii=False), flush=True)
        summary[task] = {"accuracy": correct / len(samples), "n": len(samples), "seconds": total_time, "peak_mem_gb": max_mem}
    payload = {"backend": args.backend, "model": args.model, "args": vars(args), "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
