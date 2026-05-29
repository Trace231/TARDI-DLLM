import argparse
import csv
import json
import math
from pathlib import Path


def wilson(correct, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    return center - half, center + half


def summarize(path):
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    out = []
    for task in sorted({r.get("task") for r in rows if r.get("task")}):
        task_rows = [r for r in rows if r.get("task") == task]
        n = len(task_rows)
        correct = sum(1 for r in task_rows if r.get("correct"))
        calls = [r.get("forward_calls") for r in task_rows if isinstance(r.get("forward_calls"), (int, float))]
        route_counts = {}
        stop_counts = {}
        budget_counts = {}
        for r in task_rows:
            if r.get("route"):
                route_counts[r["route"]] = route_counts.get(r["route"], 0) + 1
            if r.get("stop_reason"):
                stop_counts[r["stop_reason"]] = stop_counts.get(r["stop_reason"], 0) + 1
            if r.get("final_budget") is not None:
                b = str(r["final_budget"])
                budget_counts[b] = budget_counts.get(b, 0) + 1
        lo, hi = wilson(correct, n)
        out.append(
            {
                "method": path.stem,
                "task": task,
                "n": n,
                "accuracy": correct / max(1, n),
                "wilson_lo": lo,
                "wilson_hi": hi,
                "avg_forward_calls": sum(calls) / len(calls) if calls else "",
                "route_rates": json.dumps({k: v / max(1, n) for k, v in sorted(route_counts.items())}, ensure_ascii=False),
                "stop_rates": json.dumps({k: v / max(1, n) for k, v in sorted(stop_counts.items())}, ensure_ascii=False),
                "final_budget_rates": json.dumps({k: v / max(1, n) for k, v in sorted(budget_counts.items())}, ensure_ascii=False),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted((root / "raw").glob("*.json")):
        rows.extend(summarize(path))
    rows.sort(key=lambda r: (r["task"], r["method"]))
    out = root / "tables" / "clean_retest_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "tables" / "clean_retest_summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Clean Retest Summary",
        "",
        "| Method | Task | Acc | n | Avg Calls | Route/Budget |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        calls = r["avg_forward_calls"]
        calls_text = "" if calls == "" else f"{float(calls):.2f}"
        route = r["final_budget_rates"] or r["route_rates"] or r["stop_rates"]
        report.append(f"| {r['method']} | {r['task']} | {r['accuracy']:.3f} | {r['n']} | {calls_text} | `{route}` |")
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "Clean_Retest_Report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
