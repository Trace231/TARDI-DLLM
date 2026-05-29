import argparse
import csv
import json
import math
from pathlib import Path


TASK_LABELS = {
    "winogrande": "WinoGrande",
    "commonsenseqa": "CommonsenseQA",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "boolq": "BoolQ",
}


def wilson(correct, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    return center - half, center + half


def load(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summarize(method, payload):
    rows = []
    if not payload:
        return rows
    all_rows = payload.get("rows", [])
    for task in sorted({r.get("task") for r in all_rows}):
        task_rows = [r for r in all_rows if r.get("task") == task]
        n = len(task_rows)
        correct = sum(1 for r in task_rows if r.get("correct"))
        calls = [r.get("forward_calls") for r in task_rows if isinstance(r.get("forward_calls"), (int, float))]
        seconds = sum(r.get("seconds", 0.0) for r in task_rows if isinstance(r.get("seconds"), (int, float)))
        lo, hi = wilson(correct, n)
        extras = {}
        if method.startswith("prophet"):
            counts = {}
            for row in task_rows:
                key = str(row.get("stop_reason", "unknown"))
                counts[key] = counts.get(key, 0) + 1
            extras["stop_rates"] = json.dumps({k: v / max(1, n) for k, v in sorted(counts.items())}, ensure_ascii=False)
        rows.append(
            {
                "method": method,
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "accuracy": correct / n if n else 0.0,
                "n": n,
                "correct": correct,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "avg_forward_calls": sum(calls) / len(calls) if calls else "",
                "seconds": seconds,
                **extras,
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    raw = root / "raw"
    specs = {
        "jys_like_middle16": raw / "llada8b_jys_like_middle16_wino_cqa_arc_hella_boolq_limit300_seed23.json",
        "jys_like_back16": raw / "llada8b_jys_like_back16_wino_cqa_arc_hella_boolq_limit300_seed23.json",
        "prophet_early_commit": raw / "llada8b_prophet_early_commit_wino_cqa_arc_hella_boolq_limit300_seed23.json",
    }
    rows = []
    for method, path in specs.items():
        rows.extend(summarize(method, load(path)))
    rows.sort(key=lambda r: (r["task"], r["method"]))
    write_csv(root / "tables" / "external_sampler_baselines.csv", rows)
    (root / "tables" / "external_sampler_baselines.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
    print(json.dumps({"rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
