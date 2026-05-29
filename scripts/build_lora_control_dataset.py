#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


LETTERS = "ABCDEFGHIJ"

MC_FINAL_LABEL_SYSTEM_PROMPT = """
Think briefly, then end with exactly one line:
Final answer: <letter>

The final answer must be one of the listed option letters only. Do not answer yes, no, or maybe for multiple-choice questions. Do not put any other text after the final answer line.
"""

DECISION_FINAL_LABEL_SYSTEM_PROMPT = """
Think briefly, then end with exactly one line:
Final answer: <yes|no|maybe>

The final answer must be yes, no, or maybe only. Do not put any other text after the final answer line.
"""


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


def format_final_label_mc(question, options, instruction):
    lines = [MC_FINAL_LABEL_SYSTEM_PROMPT, instruction, "", question]
    for i, option in enumerate(options):
        lines.append(f"{LETTERS[i]}. {option}")
    return "\n".join(lines)


def add(rows, task, sample_id, prompt, answer, metric):
    rows.append(
        {
            "prompt": [{"role": "user", "content": prompt}],
            "answer": answer,
            "target": f"Final answer: {answer}",
            "task": task,
            "id": str(sample_id),
            "metric": metric,
        }
    )


def build(limit_per_task, seed, excluded):
    rng = random.Random(seed)
    rows = []

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    for ex in ds.select(range(min(50, len(ds)))):
        add(
            rows,
            "mmlu_pro",
            ex.get("question_id", len(rows)),
            format_final_label_mc(
                "Question: " + ex["question"],
                list(ex["options"]),
                "Solve this professional multiple-choice question.",
            ),
            ex["answer"],
            "letter",
        )

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
            DECISION_FINAL_LABEL_SYSTEM_PROMPT
            + "\nRead the biomedical abstract snippets and answer yes, no, or maybe.\n\nContext:\n"
            + ctx
            + "\n\nQuestion: "
            + ex["question"]
        )
        add(rows, "pubmedqa", sid, prompt, ex["final_decision"], "decision")
        count += 1
        if count >= limit_per_task:
            break

    ds = load_dataset("ceval/ceval-exam", "computer_network", split="val")
    for ex in ds:
        add(
            rows,
            "ceval_computer_network",
            ex["id"],
            format_final_label_mc(
                "Question: " + ex["question"],
                [ex[x] for x in "ABCD"],
                "Answer this Chinese computer-network single-choice question.",
            ),
            ex["answer"],
            "letter",
        )

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
        local = random.Random(17 * 1000003 + i)
        local.shuffle(opts)
        add(
            rows,
            "sciq",
            sid,
            format_final_label_mc("Question: " + ex["question"], opts, "Answer this science question."),
            LETTERS[opts.index(ex["correct_answer"])],
            "letter",
        )
        count += 1
        if count >= limit_per_task:
            break

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    count = 0
    for i in idxs:
        ex = ds[i]
        sid = str(i)
        if sid in excluded.get("winogrande", set()):
            continue
        q = "Fill the blank with the more plausible referent.\nSentence: " + ex["sentence"].replace("_", "____")
        answer = "A" if str(ex["answer"]) == "1" else "B"
        add(
            rows,
            "winogrande",
            sid,
            format_final_label_mc(q, [ex["option1"], ex["option2"]], "Resolve the reference."),
            answer,
            "letter",
        )
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
        label_to_letter = {lab: LETTERS[j] for j, lab in enumerate(ex["choices"]["label"])}
        add(
            rows,
            "commonsenseqa",
            sid,
            format_final_label_mc(
                "Question: " + ex["question"],
                ex["choices"]["text"],
                "Choose the best commonsense answer.",
            ),
            label_to_letter[ex["answerKey"]],
            "letter",
        )
        count += 1
        if count >= limit_per_task:
            break

    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-per-task", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--exclude-json", action="append", default=[])
    args = ap.parse_args()

    excluded = load_exclude_ids(args.exclude_json)
    rows = build(args.limit_per_task, args.seed, excluded)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    manifest = {"path": str(out), "n": len(rows), "counts": counts, "seed": args.seed, "limit_per_task": args.limit_per_task}
    (out.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
