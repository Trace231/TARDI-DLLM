#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


FILES = {
    "qwen_base_control": "qwen25_7b_control_base_final_label_typed_domain_shift_limit100.json",
    "qwen_lora_control_v2": "qwen25_7b_control_lora_final_label_typed_domain_shift_limit100.json",
    "llada_base_control": "llada8b_control_base_final_label_typed_domain_shift_limit100.json",
    "llada_lora_control_v2": "llada8b_control_lora_final_label_typed_domain_shift_limit100.json",
    "llada_lora_original": "llada8b_typed_lora_final_label_typed_domain_shift_limit100.json",
}

TASK_LABELS = {
    "mmlu_pro": "MMLU-Pro",
    "pubmedqa": "PubMedQA",
    "ceval_computer_network": "C-Eval CN",
    "sciq": "SciQ",
    "winogrande": "WinoGrande",
    "commonsenseqa": "CommonsenseQA",
}


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/domain_shift/task_aware/solid_v2")
    args = ap.parse_args()
    root = Path(args.root)
    raw = root / "raw"
    tables = root / "tables"
    reports = root / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    payloads = {name: load(raw / fn) for name, fn in FILES.items()}
    rows = []
    for method, payload in payloads.items():
        if not payload:
            continue
        for task, s in payload.get("summary", {}).items():
            rows.append(
                {
                    "method": method,
                    "task": task,
                    "task_label": TASK_LABELS.get(task, task),
                    "accuracy": s.get("accuracy", ""),
                    "n": s.get("n", ""),
                    "seconds": s.get("seconds", ""),
                    "adapter": payload.get("args", {}).get("adapter", ""),
                    "model": payload.get("model", ""),
                    "protocol": "controlled_v2" if "control_v2" in method else ("original_ddm_lora" if method == "llada_lora_original" else "base"),
                }
            )

    def add_gain(name, tuned, base):
        tp = payloads.get(tuned)
        bp = payloads.get(base)
        if not tp or not bp:
            return
        for task in sorted(set(tp.get("summary", {})) & set(bp.get("summary", {}))):
            rows.append(
                {
                    "method": name,
                    "task": task,
                    "task_label": TASK_LABELS.get(task, task),
                    "accuracy": tp["summary"][task]["accuracy"] - bp["summary"][task]["accuracy"],
                    "n": min(tp["summary"][task].get("n", 0), bp["summary"][task].get("n", 0)),
                    "seconds": "",
                    "adapter": "",
                    "model": "",
                    "protocol": "gain",
                }
            )

    add_gain("ar_lora_control_gain", "qwen_lora_control_v2", "qwen_base_control")
    add_gain("ddm_lora_control_gain", "llada_lora_control_v2", "llada_base_control")
    add_gain("ddm_lora_original_gain", "llada_lora_original", "llada_base_control")

    out = tables / "lora_control_v2.csv"
    fields = ["method", "task", "task_label", "accuracy", "n", "seconds", "adapter", "model", "protocol"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines = [
        "# Controlled LoRA-v2: AR vs DDM Adaptation Gain",
        "",
        "Matched protocol: same JSONL training set, LoRA r=8/alpha=16/dropout=0.05, 200 update steps, same 100-sample typed final-label evaluation.",
        "AR uses native causal final-label SFT; DDM uses native LLaDA diffusion GRPO final-label reward. The original DDM LoRA checkpoint is retained as an additional reference, not as the controlled comparison.",
        "",
        "| Method | Task | Value | n | Protocol |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        value = row["accuracy"]
        value = f"{float(value):.3f}" if value != "" else ""
        lines.append(f"| {row['method']} | {row['task_label']} | {value} | {row['n']} | {row['protocol']} |")
    report = "\n".join(lines)
    (reports / "LoRA_Control_v2_Report.md").write_text(report)
    print(json.dumps({"csv": str(out), "report": str(reports / "LoRA_Control_v2_Report.md"), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
