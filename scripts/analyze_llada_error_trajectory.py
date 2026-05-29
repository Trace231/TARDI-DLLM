import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


TASKS = ["winogrande", "commonsenseqa"]


def load_json(path):
    return json.loads(Path(path).read_text())


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def method_rows(payload, task):
    return [r for r in payload.get("rows", []) if r.get("task") == task]


def row_map(payload, task):
    return {r["id"]: r for r in method_rows(payload, task)}


def get_probe(row):
    return row.get("probe") or row.get("first_probe") or {}


def raw_fast_pred(row):
    return row.get("raw_pred") or row.get("pred")


def confidence_bins(rows, bins):
    stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        subset = []
        for r in rows:
            probe = get_probe(r)
            if not probe.get("available"):
                continue
            p = probe.get("top_prob")
            if p is None:
                continue
            if (p >= lo and p < hi) or (hi == bins[-1] and p <= hi and p >= lo):
                subset.append(r)
        if subset:
            acc = sum(bool(r.get("correct")) for r in subset) / len(subset)
        else:
            acc = None
        stats.append({"bin": f"{lo:.1f}-{hi:.1f}", "lo": lo, "hi": hi, "n": len(subset), "accuracy": acc})
    return stats


def route_rates(rows):
    counts = Counter(r.get("route") or ("fallback" if r.get("fallback_used") else "unknown") for r in rows)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in sorted(counts.items())}


def taxonomy(task, calibrated, old_adaptive, baseline):
    cal = row_map(calibrated, task)
    old = row_map(old_adaptive, task)
    base = row_map(baseline, task)
    ids = sorted(set(cal) & set(old) & set(base))
    counts = Counter()
    examples = defaultdict(list)
    for sid in ids:
        c = cal[sid]
        o = old[sid]
        b = base[sid]
        fast_correct = bool(o.get("correct"))
        cal_correct = bool(c.get("correct"))
        base_correct = bool(b.get("correct"))
        fast_pred = raw_fast_pred(o)
        probe = get_probe(c)
        disagree = probe.get("available") and fast_pred and fast_pred != probe.get("top_label")
        low_conf = probe.get("available") and probe.get("top_prob", 1.0) < 0.7

        if fast_correct and base_correct:
            key = "easy_fast"
        elif (not fast_correct) and base_correct and cal_correct:
            key = "repaired_by_fallback"
        elif fast_correct and (not base_correct):
            key = "harmed_by_fallback"
        elif (not fast_correct) and (not base_correct):
            key = "hard_both_wrong"
        else:
            key = "mixed"
        counts[key] += 1
        if disagree:
            counts["disagreement_risk"] += 1
        if low_conf:
            counts["low_confidence_risk"] += 1
        if len(examples[key]) < 3:
            examples[key].append(
                {
                    "id": sid,
                    "gold": c.get("gold"),
                    "fast_pred": fast_pred,
                    "cal_pred": c.get("pred"),
                    "base_pred": b.get("pred"),
                    "probe_top": probe.get("top_label"),
                    "probe_top_prob": probe.get("top_prob"),
                    "route": c.get("route"),
                }
            )
    return counts, examples, len(ids)


def trace_metrics(row):
    traces = row.get("traces") or []
    if not traces:
        return None
    points = []
    for block in traces:
        for item in block.get("trace", []):
            pred = item.get("pred") or ""
            points.append(
                {
                    "step": int(item.get("step", 0)),
                    "pred": pred,
                    "valid": bool(item.get("valid_label")),
                    "filled": float(item.get("filled_ratio", 0.0)),
                }
            )
    if not points:
        return None
    points.sort(key=lambda x: x["step"])
    final_pred = row.get("pred") or ""
    first_valid = next((p["step"] for p in points if p["valid"]), None)
    first_final = next((p["step"] for p in points if p["pred"] == final_pred and final_pred), None)
    valid_preds = [p["pred"] for p in points if p["valid"] and p["pred"]]
    flips = 0
    for a, b in zip(valid_preds, valid_preds[1:]):
        if a != b:
            flips += 1
    max_step = max(p["step"] for p in points)
    late = False
    late_points = [p for p in points if p["step"] >= 0.75 * max_step and p["valid"] and p["pred"]]
    for a, b in zip(late_points, late_points[1:]):
        if a["pred"] != b["pred"]:
            late = True
            break
    return {
        "first_valid_step": first_valid,
        "first_final_step": first_final,
        "flip_count": flips,
        "late_instability": late,
        "max_step": max_step,
    }


def plot_confidence(conf_stats, out_path):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(10, 4), sharey=True)
    if len(TASKS) == 1:
        axes = [axes]
    for ax, task in zip(axes, TASKS):
        stats = conf_stats.get(task, [])
        xs = [s["bin"] for s in stats]
        ys = [s["accuracy"] if s["accuracy"] is not None else 0 for s in stats]
        ns = [s["n"] for s in stats]
        ax.bar(xs, ys, color="#4C78A8")
        ax.set_title(task)
        ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy")
        ax.tick_params(axis="x", rotation=35)
        for i, (y, n) in enumerate(zip(ys, ns)):
            ax.text(i, min(0.98, y + 0.03), f"n={n}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_routes(route_stats, out_path):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(10, 4), sharey=True)
    if len(TASKS) == 1:
        axes = [axes]
    for ax, task in zip(axes, TASKS):
        rates = route_stats.get(task, {})
        labels = list(rates.keys())
        vals = [rates[k] for k in labels]
        ax.bar(labels, vals, color="#F58518")
        ax.set_title(task)
        ax.set_ylim(0, 1)
        ax.set_ylabel("route rate")
        ax.tick_params(axis="x", rotation=25)
        for i, v in enumerate(vals):
            ax.text(i, min(0.98, v + 0.03), f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pareto(methods, out_path):
    colors = {
        "baseline32": "#4C78A8",
        "old_adaptive": "#E45756",
        "forward": "#72B7B2",
        "calibrated": "#54A24B",
    }
    fig, axes = plt.subplots(1, len(TASKS), figsize=(10, 4), sharey=True)
    if len(TASKS) == 1:
        axes = [axes]
    for ax, task in zip(axes, TASKS):
        for name, payload in methods.items():
            s = payload.get("summary", {}).get(task)
            if not s:
                continue
            calls = s.get("avg_forward_calls", 32.0)
            acc = s.get("accuracy")
            ax.scatter([calls], [acc], s=70, label=name, color=colors.get(name))
            ax.text(calls + 0.3, acc, name, fontsize=8)
        ax.set_title(task)
        ax.set_xlabel("avg forward calls")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trace(trace_payload, out_path):
    metrics_by_task = defaultdict(list)
    for task in TASKS:
        for row in method_rows(trace_payload, task):
            m = trace_metrics(row)
            if m:
                metrics_by_task[task].append(m)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(10, 4), sharey=True)
    if len(TASKS) == 1:
        axes = [axes]
    for ax, task in zip(axes, TASKS):
        vals = [m["first_final_step"] for m in metrics_by_task[task] if m["first_final_step"] is not None]
        if vals:
            ax.hist(vals, bins=range(1, max(vals) + 2), color="#B279A2", alpha=0.85)
        ax.set_title(task)
        ax.set_xlabel("first final-label step")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return metrics_by_task


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline32", required=True)
    ap.add_argument("--old-adaptive", required=True)
    ap.add_argument("--forward", required=True)
    ap.add_argument("--calibrated", required=True)
    ap.add_argument("--trace")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    methods = {
        "baseline32": load_json(args.baseline32),
        "old_adaptive": load_json(args.old_adaptive),
        "forward": load_json(args.forward),
        "calibrated": load_json(args.calibrated),
    }
    calibrated = methods["calibrated"]

    bins = [0.0, 0.4, 0.6, 0.8, 1.0]
    conf_stats = {task: confidence_bins(method_rows(calibrated, task), bins) for task in TASKS}
    route_stats = {task: route_rates(method_rows(calibrated, task)) for task in TASKS}

    conf_rows = []
    for task, stats in conf_stats.items():
        for row in stats:
            conf_rows.append({"task": task, **row})
    write_csv(out_dir / "confidence_accuracy.csv", conf_rows, ["task", "bin", "lo", "hi", "n", "accuracy"])

    route_rows = []
    for task, rates in route_stats.items():
        for route, rate in rates.items():
            route_rows.append({"task": task, "route": route, "rate": rate})
    write_csv(out_dir / "route_distribution.csv", route_rows, ["task", "route", "rate"])

    taxonomy_rows = []
    examples = {}
    for task in TASKS:
        counts, ex, n = taxonomy(task, calibrated, methods["old_adaptive"], methods["baseline32"])
        examples[task] = ex
        for key, count in sorted(counts.items()):
            taxonomy_rows.append({"task": task, "type": key, "count": count, "rate": count / n if n else 0.0})
    write_csv(out_dir / "error_taxonomy.csv", taxonomy_rows, ["task", "type", "count", "rate"])

    plot_confidence(conf_stats, out_dir / "confidence_accuracy.png")
    plot_routes(route_stats, out_dir / "route_distribution.png")
    plot_pareto(methods, out_dir / "accuracy_cost_pareto.png")

    trace_summary = {}
    if args.trace:
        trace_payload = load_json(args.trace)
        metrics_by_task = plot_trace(trace_payload, out_dir / "trajectory_stabilization.png")
        trace_rows = []
        for task, metrics in metrics_by_task.items():
            for m in metrics:
                trace_rows.append({"task": task, **m})
            if metrics:
                first_final = [m["first_final_step"] for m in metrics if m["first_final_step"] is not None]
                flips = [m["flip_count"] for m in metrics]
                trace_summary[task] = {
                    "n": len(metrics),
                    "mean_first_final_step": sum(first_final) / len(first_final) if first_final else None,
                    "mean_flip_count": sum(flips) / len(flips) if flips else None,
                    "late_instability_rate": sum(m["late_instability"] for m in metrics) / len(metrics),
                }
        write_csv(
            out_dir / "trajectory_metrics.csv",
            trace_rows,
            ["task", "first_valid_step", "first_final_step", "flip_count", "late_instability", "max_step"],
        )

    report = {
        "method_summary": {name: payload.get("summary", {}) for name, payload in methods.items()},
        "confidence_stats": conf_stats,
        "route_stats": route_stats,
        "taxonomy_examples": examples,
        "trace_summary": trace_summary,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    lines = ["# LLaDA Error and Trajectory Analysis", ""]
    lines.append("## Method Summary")
    for name, payload in methods.items():
        lines.append(f"- {name}: `{json.dumps(payload.get('summary', {}), ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Generated Artifacts")
    for name in [
        "confidence_accuracy.png",
        "route_distribution.png",
        "accuracy_cost_pareto.png",
        "trajectory_stabilization.png" if args.trace else None,
        "error_taxonomy.csv",
        "confidence_accuracy.csv",
        "route_distribution.csv",
    ]:
        if name:
            lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Taxonomy Highlights")
    for row in taxonomy_rows:
        lines.append(f"- {row['task']} / {row['type']}: {row['count']} ({row['rate']:.3f})")
    (out_dir / "LLaDA_Error_Trajectory_Analysis_Report.md").write_text("\n".join(lines))
    print(json.dumps({"out_dir": str(out_dir), "trace_summary": trace_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
