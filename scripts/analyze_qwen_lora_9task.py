#!/usr/bin/env python3
import argparse
import csv
import json
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
}


def load_task_acc(path):
    payload = json.loads(Path(path).read_text())
    rows = payload.get("rows", [])
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(bool(row["correct"]))
    return {
        task: {
            "n": len(vals),
            "correct": sum(vals),
            "accuracy": sum(vals) / len(vals) if vals else 0.0,
        }
        for task, vals in by_task.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load_task_acc(args.base)
    lora = load_task_acc(args.lora)
    tasks = [t for t in TASK_NAMES if t in base or t in lora]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "task_name",
                "qwen_base_accuracy",
                "qwen_lora_accuracy",
                "qwen_lora_gain",
                "base_correct",
                "lora_correct",
                "n",
            ],
        )
        writer.writeheader()
        gains = []
        for task in tasks:
            b = base.get(task, {"accuracy": 0.0, "correct": 0, "n": 0})
            l = lora.get(task, {"accuracy": 0.0, "correct": 0, "n": 0})
            n = min(b["n"], l["n"])
            gain = l["accuracy"] - b["accuracy"]
            gains.append(gain)
            writer.writerow(
                {
                    "task": task,
                    "task_name": TASK_NAMES.get(task, task),
                    "qwen_base_accuracy": f"{b['accuracy']:.6f}",
                    "qwen_lora_accuracy": f"{l['accuracy']:.6f}",
                    "qwen_lora_gain": f"{gain:+.6f}",
                    "base_correct": b["correct"],
                    "lora_correct": l["correct"],
                    "n": n,
                }
            )
        if tasks:
            writer.writerow(
                {
                    "task": "macro",
                    "task_name": "Macro",
                    "qwen_base_accuracy": f"{sum(base[t]['accuracy'] for t in tasks if t in base) / len(tasks):.6f}",
                    "qwen_lora_accuracy": f"{sum(lora[t]['accuracy'] for t in tasks if t in lora) / len(tasks):.6f}",
                    "qwen_lora_gain": f"{sum(gains) / len(gains):+.6f}",
                    "base_correct": "",
                    "lora_correct": "",
                    "n": "",
                }
            )


if __name__ == "__main__":
    main()
