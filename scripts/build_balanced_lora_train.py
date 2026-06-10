#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as ed


DEFAULT_TASKS = (
    "mmlu_pro,pubmedqa,ceval_computer_network,sciq,"
    "winogrande,commonsenseqa,arc_challenge,hellaswag,boolq"
)


def final_label_instruction(metric):
    if metric == "letter":
        return ed.MC_FINAL_LABEL_EVAL_PROMPT
    if metric == "decision":
        return ed.DECISION_FINAL_LABEL_EVAL_PROMPT
    if metric == "bool":
        return ed.BOOL_FINAL_LABEL_EVAL_PROMPT
    return ed.FINAL_LABEL_EVAL_PROMPT


def clean_prompt(prompt):
    text = str(prompt).strip()
    if text.endswith("Answer:"):
        text = text[: -len("Answer:")].rstrip()
    return text


def make_row(sample):
    gold = str(sample["gold"]).strip()
    metric = sample.get("metric", "letter")
    prompt = final_label_instruction(metric).strip() + "\n\n" + clean_prompt(sample["prompt"])
    return {
        "prompt": [{"role": "user", "content": "\n" + prompt}],
        "answer": gold,
        "target": f"Final answer: {gold}",
        "task": sample["task"],
        "id": str(sample["id"]),
        "metric": metric,
    }


def collect_task(task, limit, seed, exclude_ids, pool_limit, max_rounds):
    rows = {}
    for round_idx in range(max_rounds):
        round_seed = seed + round_idx * 9973
        for sample in ed.build_samples(task, pool_limit, round_seed):
            sid = str(sample["id"])
            if sid in exclude_ids or sid in rows:
                continue
            rows[sid] = make_row(sample)
            if len(rows) >= limit:
                return list(rows.values())
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--limit-per-task", type=int, default=100)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--exclude-seed", type=int, default=23)
    ap.add_argument("--exclude-limit", type=int, default=50)
    ap.add_argument("--pool-limit", type=int, default=600)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    all_rows = []
    manifest = {"tasks": {}, "args": vars(args)}
    for task in tasks:
        exclude = {str(x["id"]) for x in ed.build_samples(task, args.exclude_limit, args.exclude_seed)}
        rows = collect_task(task, args.limit_per_task, args.seed, exclude, args.pool_limit, args.max_rounds)
        manifest["tasks"][task] = {
            "n": len(rows),
            "excluded_eval_ids": len(exclude),
        }
        all_rows.extend(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"out": str(out), "n": len(all_rows), "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
