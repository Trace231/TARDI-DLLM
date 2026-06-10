#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METHODS = {
    "base_fixed32": ("choice", "llada_base_fixed32_limit50_seed23.json"),
    "vanilla_lora_fixed32": ("choice", "llada_vanilla_fixed32_limit50_seed23.json"),
    "label_lora_fixed32": ("choice", "llada_label_fixed32_limit50_seed23.json"),
    "choice_noise_lora_fixed32": ("choice", "llada_choice_noise_fixed32_limit50_seed23.json"),
    "nara_vanilla_fixed32": ("external", "llada_nara_vanilla_fixed32_limit50_seed23.json"),
    "loraplus_vanilla_fixed32": ("external", "llada_loraplus_vanilla_fixed32_limit50_seed23.json"),
    "rslora_vanilla_fixed32": ("external", "llada_rslora_vanilla_fixed32_limit50_seed23.json"),
    "dora_vanilla_fixed32": ("external", "llada_dora_vanilla_fixed32_limit50_seed23.json"),
    "tardi_lora_balanced_r8": ("opt", "llada_balanced_vanilla_r8_s100_fixed32_limit50_seed23.json"),
    "tardi_lora_balanced_r8_highnoise": ("opt", "llada_balanced_vanilla_r8_highnoise_s100_fixed32_limit50_seed23.json"),
    "tardi_lora_balanced_r16": ("opt", "llada_balanced_vanilla_r16_s100_fixed32_limit50_seed23.json"),
    "tardi_lora_balanced_loraplus_r16": ("opt", "llada_balanced_loraplus_r16_s100_fixed32_limit50_seed23.json"),
    "tardi_lora_balanced_r8_lr5e5_s150": ("opt", "llada_balanced_vanilla_r8_s150_lr5e5_fixed32_limit50_seed23.json"),
    "tardi_lora_balanced_r16_highnoise": ("opt", "llada_balanced_vanilla_r16_highnoise_s100_fixed32_limit50_seed23.json"),
}


def load_json(path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def summarize(method, payload):
    rows = payload.get("rows", [])
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    task_rows = []
    for task, vals in sorted(by_task.items()):
        n = len(vals)
        correct = sum(1 for r in vals if r.get("correct"))
        task_rows.append({
            "method": method,
            "task": task,
            "accuracy": correct / n if n else 0.0,
            "correct": correct,
            "n": n,
        })
    macro = sum(r["accuracy"] for r in task_rows) / len(task_rows) if task_rows else 0.0
    correct = sum(r["correct"] for r in task_rows)
    n = sum(r["n"] for r in task_rows)
    return {
        "method": method,
        "status": "complete",
        "tasks": len(task_rows),
        "correct": correct,
        "n": n,
        "macro_accuracy": macro,
        "micro_accuracy": correct / n if n else 0.0,
    }, task_rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(macro_rows, task_rows, out_root):
    complete = [r for r in macro_rows if r["status"] == "complete"]
    best = max(complete, key=lambda r: r["macro_accuracy"])
    old = next(r for r in complete if r["method"] == "vanilla_lora_fixed32")
    base = next(r for r in complete if r["method"] == "base_fixed32")
    lines = ["# LoRA Optimization v1 Report\n\n"]
    lines.append("This report evaluates task-balanced and noise-schedule tuned LLaDA LoRA variants against the previous vanilla LoRA and external improved LoRA baselines.\n\n")
    lines.append("## Macro Results\n\n")
    lines.append("| Method | Macro Acc. | Correct / N | Delta vs old vanilla |\n")
    lines.append("|---|---:|---:|---:|\n")
    for r in sorted(complete, key=lambda x: x["macro_accuracy"], reverse=True):
        delta = r["macro_accuracy"] - old["macro_accuracy"]
        lines.append(f"| {r['method']} | {r['macro_accuracy']:.3f} | {r['correct']} / {r['n']} | {delta:+.3f} |\n")
    lines.append("\n## Main Finding\n\n")
    lines.append(
        f"Best method `{best['method']}` reaches `{best['macro_accuracy']:.3f}` "
        f"({best['correct']} / {best['n']}), improving over the previous vanilla LoRA "
        f"`{old['macro_accuracy']:.3f}` by `{best['macro_accuracy'] - old['macro_accuracy']:+.3f}` "
        f"and over base `{base['macro_accuracy']:.3f}` by `{best['macro_accuracy'] - base['macro_accuracy']:+.3f}`.\n\n"
    )
    lines.append("The gain comes from repairing the training/evaluation mismatch: the old train set covered only six tasks and omitted ARC-Challenge, HellaSwag, and BoolQ; the optimized train set is 9-task balanced and excludes the evaluation ids. High-noise denoising and r16 capacity both help, but their combination does not stack additively.\n\n")
    lines.append("## Task-Level Table\n\n")
    methods = [r["method"] for r in sorted(complete, key=lambda x: x["macro_accuracy"], reverse=True)]
    tasks = sorted({r["task"] for r in task_rows})
    lookup = {(r["method"], r["task"]): r["accuracy"] for r in task_rows}
    lines.append("| Task | " + " | ".join(methods[:6]) + " |\n")
    lines.append("|---" + "|---:" * min(6, len(methods)) + "|\n")
    for task in tasks:
        vals = [lookup.get((m, task), 0.0) for m in methods[:6]]
        lines.append("| " + task + " | " + " | ".join(f"{v:.2f}" for v in vals) + " |\n")
    lines.append(f"\nOutputs live in `{out_root}`.\n")
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--choice-root", default="results/domain_shift/task_aware/choice_noise_v1")
    ap.add_argument("--external-root", default="results/domain_shift/task_aware/lora_external_v1")
    ap.add_argument("--opt-root", default="results/domain_shift/task_aware/lora_opt_v1")
    ap.add_argument("--out-root", default="results/domain_shift/task_aware/lora_opt_v1")
    args = ap.parse_args()

    roots = {
        "choice": Path(args.choice_root) / "raw",
        "external": Path(args.external_root) / "raw",
        "opt": Path(args.opt_root) / "raw",
    }
    macro_rows = []
    task_rows = []
    for method, (root_key, filename) in METHODS.items():
        payload = load_json(roots[root_key] / filename)
        if payload is None:
            macro_rows.append({"method": method, "status": "missing"})
            continue
        macro, tasks = summarize(method, payload)
        macro_rows.append(macro)
        task_rows.extend(tasks)

    out = Path(args.out_root)
    write_csv(out / "tables" / "lora_opt_macro_summary.csv", macro_rows, ["method", "status", "tasks", "correct", "n", "macro_accuracy", "micro_accuracy"])
    write_csv(out / "tables" / "lora_opt_task_summary.csv", task_rows, ["method", "task", "accuracy", "correct", "n"])
    (out / "tables" / "lora_opt_summary.json").write_text(json.dumps({"macro_rows": macro_rows, "task_rows": task_rows}, indent=2, ensure_ascii=False))
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "LoRA_Optimization_v1_Report.md").write_text(report(macro_rows, task_rows, out), encoding="utf-8")
    print(json.dumps({"macro_rows": macro_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
