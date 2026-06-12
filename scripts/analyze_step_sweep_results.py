#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


TASK_NAMES = {
    "mmlu_pro": "MMLU-Pro",
    "pubmedqa": "PubMedQA",
    "ceval_computer_network": "C-Eval",
    "sciq": "SciQ",
    "winogrande": "WinoGrande",
    "commonsenseqa": "CommonsenseQA",
    "arc_challenge": "ARC",
    "hellaswag": "HellaSwag",
    "boolq": "BoolQ",
    "gsm8k": "GSM8K",
}


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def step_from_name(path):
    m = re.search(r"steps(\d+)", path.name)
    return int(m.group(1)) if m else None


def rows_from_legacy_payload(payload, path):
    summary = payload.get("summary")
    if not isinstance(summary, list):
        return []
    out = []
    for item in summary:
        if not isinstance(item, dict):
            continue
        task = item.get("task")
        step = item.get("step")
        n = int(item.get("n", 0) or 0)
        acc = float(item.get("accuracy", 0) or 0)
        out.append(
            {
                "task": task,
                "task_name": TASK_NAMES.get(task, task),
                "step": step,
                "n": n,
                "accuracy": f"{acc:.6f}",
                "correct": item.get("correct", int(round(acc * n))),
                "seconds": f"{float(item.get('seconds', 0) or 0):.6f}",
                "peak_mem_gb": f"{float(payload.get('peak_mem_gb', 0) or 0):.6f}",
                "source": str(path),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for path in sorted((root / "raw").glob("*.json")):
        payload = load_json(path)
        if not payload:
            continue
        step = step_from_name(path)
        summary = payload.get("summary", {})
        if isinstance(summary, list):
            rows.extend(rows_from_legacy_payload(payload, path))
            continue
        for task, stats in sorted(summary.items()):
            n = int(stats.get("n", 0) or 0)
            acc = float(stats.get("accuracy", 0) or 0)
            rows.append(
                {
                    "task": task,
                    "task_name": TASK_NAMES.get(task, task),
                    "step": step,
                    "n": n,
                    "accuracy": f"{acc:.6f}",
                    "correct": int(round(acc * n)),
                    "seconds": f"{float(stats.get('seconds', 0) or 0):.6f}",
                    "peak_mem_gb": f"{float(stats.get('peak_mem_gb', 0) or 0):.6f}",
                    "source": str(path),
                }
            )

    rows.sort(key=lambda r: (r["task"], int(r["step"] or 0)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fieldnames = ["task", "task_name", "step", "n", "accuracy", "correct", "seconds", "peak_mem_gb", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
