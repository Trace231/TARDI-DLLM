#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


METHOD_FILES = {
    "llada_base_fixed32": "llada_base_fixed32_limit50_seed23.json",
    "llada_vanilla_lora_fixed32": "llada_vanilla_fixed32_limit50_seed23.json",
    "llada_label_lora_fixed32": "llada_label_fixed32_limit50_seed23.json",
    "llada_choice_noise_lora_fixed32": "llada_choice_noise_fixed32_limit50_seed23.json",
    "llada_choice_noise_lora_controller": "llada_choice_noise_controller_limit50_seed23.json",
    "llada_vanilla_lora_controller": "llada_vanilla_controller_limit50_seed23.json",
}


def wilson_ci(correct, n, z=1.96):
    if n <= 0:
        return None, None
    phat = correct / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def load_json(path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def infer_correct_from_summary(task_summary):
    n = int(task_summary.get("n", 0) or 0)
    acc = float(task_summary.get("accuracy", 0.0) or 0.0)
    return int(round(acc * n)), n


def rows_by_task(rows):
    out = {}
    for row in rows or []:
        task = row.get("task", "unknown")
        out.setdefault(task, []).append(row)
    return out


def task_metric_from_rows(rows):
    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    return correct, n


def flatten_extra(summary):
    extras = {}
    for key, value in summary.items():
        if isinstance(value, (int, float)) and key not in {"accuracy", "n"}:
            extras[key] = value
    return extras


def collect(root):
    raw_dir = root / "raw"
    all_rows = []
    task_order = []
    for method, filename in METHOD_FILES.items():
        data = load_json(raw_dir / filename)
        if data is None:
            continue
        summary = data.get("summary", {}) or {}
        row_groups = rows_by_task(data.get("rows", []))
        tasks = list(summary.keys()) or list(row_groups.keys())
        for task in tasks:
            if task not in task_order:
                task_order.append(task)
            if task in row_groups:
                correct, n = task_metric_from_rows(row_groups[task])
            else:
                correct, n = infer_correct_from_summary(summary[task])
            acc = correct / n if n else 0.0
            lo, hi = wilson_ci(correct, n)
            record = {
                "method": method,
                "task": task,
                "correct": correct,
                "n": n,
                "accuracy": acc,
                "wilson_low": lo,
                "wilson_high": hi,
            }
            if isinstance(summary.get(task), dict):
                record.update(flatten_extra(summary[task]))
            all_rows.append(record)
    return all_rows, task_order


def macro_table(rows):
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    out = []
    for method, vals in by_method.items():
        total_correct = sum(v["correct"] for v in vals)
        total_n = sum(v["n"] for v in vals)
        macro_acc = sum(v["accuracy"] for v in vals) / len(vals)
        micro_acc = total_correct / total_n if total_n else 0.0
        seconds = sum(float(v.get("seconds", 0.0) or 0.0) for v in vals)
        avg_calls_vals = [float(v.get("avg_calls", v.get("avg_forward_calls", 0.0)) or 0.0) for v in vals]
        avg_calls_vals = [v for v in avg_calls_vals if v > 0]
        out.append({
            "method": method,
            "tasks": len(vals),
            "total_correct": total_correct,
            "total_n": total_n,
            "macro_accuracy": macro_acc,
            "micro_accuracy": micro_acc,
            "seconds": seconds,
            "avg_calls_macro": sum(avg_calls_vals) / len(avg_calls_vals) if avg_calls_vals else "",
        })
    out.sort(key=lambda r: r["macro_accuracy"], reverse=True)
    return out


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(x):
    if x == "":
        return ""
    return f"{100 * float(x):.1f}%"


def make_report(root, task_rows, macro_rows):
    by_method = {r["method"]: r for r in macro_rows}
    base = by_method.get("llada_base_fixed32")
    best = macro_rows[0] if macro_rows else None
    lines = []
    lines.append("# Choice-aware Noise-aware LoRA 与选择性再掩码控制器实验报告\n")
    lines.append("## 1. 实验目标\n")
    lines.append(
        "本实验检验 masked diffusion language model 在 fixed-label reasoning 上的两个改进方向："
        "训练侧的 choice/noise-aware LoRA，以及推理侧的 selective re-masking controller。"
        "所有方法使用相同 held-out 样本、相同 prompt 格式和相同 seed。"
    )
    lines.append("\n## 2. 主结果\n")
    lines.append("| Method | Macro Acc. | Micro Acc. | Total n | Seconds | Avg calls |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for row in macro_rows:
        avg_calls = row["avg_calls_macro"]
        lines.append(
            f"| {row['method']} | {fmt_pct(row['macro_accuracy'])} | "
            f"{fmt_pct(row['micro_accuracy'])} | {row['total_n']} | "
            f"{row['seconds']:.1f} | {avg_calls if avg_calls == '' else f'{avg_calls:.2f}'} |\n"
        )
    if base and best:
        gain = best["macro_accuracy"] - base["macro_accuracy"]
        lines.append(
            f"\n当前最佳方法为 `{best['method']}`，相对 base fixed-32 的宏平均变化为 {gain*100:+.1f} 个百分点。\n"
        )
    lines.append("\n## 3. 逐任务结果\n")
    methods = [r["method"] for r in macro_rows]
    tasks = []
    for row in task_rows:
        if row["task"] not in tasks:
            tasks.append(row["task"])
    lookup = {(r["method"], r["task"]): r for r in task_rows}
    lines.append("| Task | " + " | ".join(methods) + " |\n")
    lines.append("|---|" + "|".join(["---:"] * len(methods)) + "|\n")
    for task in tasks:
        vals = []
        for method in methods:
            row = lookup.get((method, task))
            vals.append(fmt_pct(row["accuracy"]) if row else "")
        lines.append(f"| {task} | " + " | ".join(vals) + " |\n")
    lines.append("\n## 4. 解释\n")
    lines.append(
        "- `vanilla LoRA` 用于判断普通 DDM LoRA 是否能自然转化为 final-label accuracy gain。\n"
        "- `label-focused LoRA` 直接优化合法标签 posterior，用于验证 final-label objective mismatch。\n"
        "- `choice-noise LoRA` 在 label objective 上加入多噪声阶段监督和一致性约束，用于降低单纯标签对齐带来的负迁移。\n"
        "- `controller` 若保持接近 fixed-32 的准确率并降低 avg calls，则说明适配后的模型仍存在样本级 reverse budget heterogeneity。\n"
    )
    lines.append("\n## 5. 文件\n")
    lines.append(f"原始 JSON、日志、表格均位于 `{root}`。\n")
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/domain_shift/task_aware/choice_noise_v1")
    args = ap.parse_args()
    root = Path(args.root)
    task_rows, _ = collect(root)
    macro_rows = macro_table(task_rows)

    table_dir = root / "tables"
    report_dir = root / "reports"
    write_csv(table_dir / "choice_noise_task_summary.csv", task_rows, sorted({k for r in task_rows for k in r.keys()}))
    write_csv(table_dir / "choice_noise_macro_summary.csv", macro_rows, list(macro_rows[0].keys()) if macro_rows else [])
    (table_dir / "choice_noise_summary.json").write_text(json.dumps({
        "task_rows": task_rows,
        "macro_rows": macro_rows,
    }, ensure_ascii=False, indent=2))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "Choice_Noise_LoRA_Controller_Report_zh.md").write_text(
        make_report(root, task_rows, macro_rows),
        encoding="utf-8",
    )
    print(json.dumps({"macro_rows": macro_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
