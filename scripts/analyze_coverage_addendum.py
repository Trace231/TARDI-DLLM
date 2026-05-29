import argparse
import csv
import json
import math
from pathlib import Path


TASK_LABELS = {
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "boolq": "BoolQ",
    "gsm8k": "GSM8K",
}


def wilson(correct, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return center - half, center + half


def route_rates(rows):
    counts = {}
    for row in rows:
        route = str(row.get("route", "fixed"))
        counts[route] = counts.get(route, 0) + 1
    total = max(1, len(rows))
    return {k: v / total for k, v in sorted(counts.items())}


def load_payload(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def rows_for(payload, task):
    return [r for r in payload.get("rows", []) if r.get("task") == task]


def summarize_payload(method, payload):
    out = []
    if not payload:
        return out
    tasks = sorted({r.get("task") for r in payload.get("rows", [])})
    for task in tasks:
        rows = rows_for(payload, task)
        n = len(rows)
        correct = sum(1 for r in rows if r.get("correct"))
        lo, hi = wilson(correct, n)
        calls = [r.get("forward_calls") for r in rows if isinstance(r.get("forward_calls"), (int, float))]
        secs = []
        for r in rows:
            if isinstance(r.get("seconds"), (int, float)):
                secs.append(r["seconds"])
        summary = payload.get("summary", {}).get(task, {})
        if not secs and isinstance(summary.get("seconds"), (int, float)):
            secs_total = summary["seconds"]
        else:
            secs_total = sum(secs)
        out.append(
            {
                "method": method,
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "accuracy": correct / n if n else 0.0,
                "n": n,
                "correct": correct,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "avg_forward_calls": (sum(calls) / len(calls)) if calls else "",
                "seconds": secs_total,
                "route_rates": json.dumps(route_rates(rows), ensure_ascii=False) if any("route" in r for r in rows) else json.dumps({"fixed": 1.0}),
            }
        )
    return out


def paired_deltas(rows):
    by_method_task = {}
    for row in rows:
        by_method_task[(row["method"], row["task"])] = row
    pairs = []
    for task in sorted({r["task"] for r in rows}):
        base8 = by_method_task.get(("llada8b_8step", task))
        base32 = by_method_task.get(("llada8b_32step", task))
        ctrl = by_method_task.get(("llada8b_calibrated", task))
        qwen = by_method_task.get(("qwen25_7b", task))
        if base8 and base32:
            pairs.append({"task": task, "comparison": "32step_minus_8step", "delta": base32["accuracy"] - base8["accuracy"]})
        if base32 and ctrl:
            pairs.append({"task": task, "comparison": "calibrated_minus_32step", "delta": ctrl["accuracy"] - base32["accuracy"]})
        if base32 and qwen:
            pairs.append({"task": task, "comparison": "llada32_minus_qwen", "delta": base32["accuracy"] - qwen["accuracy"]})
    return pairs


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x):
    if x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_report(path, rows, deltas):
    closed_tasks = {"arc_challenge", "hellaswag", "boolq"}
    lines = [
        "# Coverage Addendum: Broader Downstream Tasks",
        "",
        "This addendum extends the original closed-label suite with science reasoning, situation continuation, reading yes/no, and answer-only math.",
        "The goal is coverage, not a new controller tuned to every task.",
        "",
        "## Summary Table",
        "",
        "| Method | Task | Acc | n | Avg Calls | Route |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['task_label']} | {fmt(row['accuracy'])} | {row['n']} | {fmt(row['avg_forward_calls'])} | `{row['route_rates']}` |"
        )
    lines.extend(["", "## Deltas", "", "| Task | Comparison | Delta |", "|---|---:|---:|"])
    for row in deltas:
        lines.append(f"| {TASK_LABELS.get(row['task'], row['task'])} | {row['comparison']} | {fmt(row['delta'])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ARC-Challenge, HellaSwag, and BoolQ broaden the closed-label setting beyond the original commonsense/knowledge tasks.",
            "- GSM8K is included as an answer-only long-chain reasoning boundary case; the controller is not claimed as a complete solution for open numeric reasoning.",
            "- If 8-step and 32-step differ on a task, the task is reverse-budget sensitive. If both fail similarly, the bottleneck is more likely knowledge, reasoning format, or answer extraction.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    raw = root / "raw"
    payloads = {
        "llada8b_8step": [
            raw / "llada8b_8step_coverage_closed_limit300_seed23.json",
            raw / "llada8b_8step_gsm8k_limit100_seed23.json",
        ],
        "llada8b_32step": [
            raw / "llada8b_32step_coverage_closed_limit300_seed23.json",
            raw / "llada8b_32step_gsm8k_limit100_seed23.json",
        ],
        "llada8b_calibrated": [
            raw / "llada8b_calibrated_coverage_closed_limit300_seed23.json",
        ],
        "qwen25_7b": [
            raw / "qwen25_7b_coverage_closed_limit300_seed23.json",
            raw / "qwen25_7b_gsm8k_limit100_seed23.json",
        ],
    }
    rows = []
    for method, paths in payloads.items():
        for path in paths:
            payload = load_payload(path)
            rows.extend(summarize_payload(method, payload))
    rows.sort(key=lambda r: (r["task"], r["method"]))
    deltas = paired_deltas(rows)

    write_csv(root / "tables" / "coverage_summary.csv", rows)
    write_csv(root / "tables" / "coverage_deltas.csv", deltas)
    write_report(root / "reports" / "Coverage_Addendum_Report.md", rows, deltas)
    (root / "tables" / "coverage_summary.json").write_text(json.dumps({"rows": rows, "deltas": deltas}, ensure_ascii=False, indent=2))
    print(json.dumps({"n_rows": len(rows), "n_deltas": len(deltas)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
