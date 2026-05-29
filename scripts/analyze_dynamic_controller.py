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


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summarize_payload(method, payload):
    rows = []
    if not payload:
        return rows
    data = payload.get("rows", [])
    for task in sorted({r.get("task") for r in data if r.get("task")}):
        task_rows = [r for r in data if r.get("task") == task]
        n = len(task_rows)
        correct = sum(1 for r in task_rows if r.get("correct"))
        calls = [r.get("forward_calls") for r in task_rows if isinstance(r.get("forward_calls"), (int, float))]
        seconds = sum(r.get("seconds", 0.0) for r in task_rows if isinstance(r.get("seconds"), (int, float)))
        lo, hi = wilson(correct, n)
        route_counts = {}
        budget_counts = {}
        score_vals = []
        for r in task_rows:
            route = str(r.get("route") or r.get("stop_reason") or "unknown")
            route_counts[route] = route_counts.get(route, 0) + 1
            budget = r.get("final_budget") or r.get("committed_step")
            if budget is not None:
                budget_counts[str(budget)] = budget_counts.get(str(budget), 0) + 1
            if isinstance(r.get("risk_score"), (int, float)):
                score_vals.append(float(r["risk_score"]))
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
                "route_rates": json.dumps({k: v / max(1, n) for k, v in sorted(route_counts.items())}, ensure_ascii=False),
                "final_budget_rates": json.dumps({k: v / max(1, n) for k, v in sorted(budget_counts.items())}, ensure_ascii=False),
                "mean_risk_score": sum(score_vals) / len(score_vals) if score_vals else "",
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


def paired_delta(a_payload, b_payload, a_name, b_name):
    out = []
    if not a_payload or not b_payload:
        return out
    a = {(r.get("task"), str(r.get("id"))): r for r in a_payload.get("rows", [])}
    b = {(r.get("task"), str(r.get("id"))): r for r in b_payload.get("rows", [])}
    keys = sorted(set(a) & set(b))
    for task in sorted({k[0] for k in keys}):
        task_keys = [k for k in keys if k[0] == task]
        b_only = sum(1 for k in task_keys if b[k].get("correct") and not a[k].get("correct"))
        a_only = sum(1 for k in task_keys if a[k].get("correct") and not b[k].get("correct"))
        both = sum(1 for k in task_keys if a[k].get("correct") and b[k].get("correct"))
        neither = sum(1 for k in task_keys if not a[k].get("correct") and not b[k].get("correct"))
        out.append(
            {
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "method_a": a_name,
                "method_b": b_name,
                "n_intersection": len(task_keys),
                "a_only": a_only,
                "b_only": b_only,
                "both_correct": both,
                "both_wrong": neither,
                "net_b_minus_a": b_only - a_only,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solid-root", required=True)
    ap.add_argument("--dynamic-root", required=True)
    args = ap.parse_args()
    solid = Path(args.solid_root)
    dyn = Path(args.dynamic_root)
    raw = dyn / "raw"
    coverage = solid / "coverage_addendum" / "raw"
    external = solid / "external_sampler_baselines" / "raw"

    specs = {
        "selective_remask_refinement": raw / "llada8b_refinement_controller_wino_cqa_arc_hella_boolq_limit300_seed23.json",
        "risk_controller": raw / "llada8b_risk_controller_wino_cqa_arc_hella_boolq_limit300_seed23.json",
        "multi_budget_controller": raw / "llada8b_multibudget_controller_wino_cqa_arc_hella_boolq_limit300_seed23.json",
        "llada_8step": coverage / "llada8b_8step_coverage_closed_limit300_seed23.json",
        "llada_32step": coverage / "llada8b_32step_coverage_closed_limit300_seed23.json",
        "binary_calibrated": coverage / "llada8b_calibrated_coverage_closed_limit300_seed23.json",
        "jys_middle16": external / "llada8b_jys_like_middle16_wino_cqa_arc_hella_boolq_limit300_seed23.json",
        "jys_back16": external / "llada8b_jys_like_back16_wino_cqa_arc_hella_boolq_limit300_seed23.json",
        "prophet_early_commit": external / "llada8b_prophet_early_commit_wino_cqa_arc_hella_boolq_limit300_seed23.json",
    }
    payloads = {k: load_json(v) for k, v in specs.items()}
    rows = []
    for method, payload in payloads.items():
        rows.extend(summarize_payload(method, payload))
    rows.sort(key=lambda r: (r["task"], r["method"]))
    write_csv(dyn / "tables" / "dynamic_controller_summary.csv", rows)

    deltas = []
    for target in ["selective_remask_refinement", "risk_controller"]:
        for baseline in ["llada_8step", "llada_32step", "binary_calibrated", "prophet_early_commit", "jys_middle16", "jys_back16"]:
            deltas.extend(paired_delta(payloads.get(baseline), payloads.get(target), baseline, target))
    write_csv(dyn / "tables" / "dynamic_controller_paired_deltas.csv", deltas)

    report_lines = [
        "# Dynamic Controller Addendum",
        "",
        "This addendum evaluates whether the controller is truly dynamic rather than a binary 8/32 fallback.",
        "",
        "## Summary Table",
        "",
        "| Method | Task | Acc | n | Avg Calls | Final Budget Rates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        avg_calls = r["avg_forward_calls"]
        avg_calls_text = "" if avg_calls == "" else f"{float(avg_calls):.3f}"
        report_lines.append(
            f"| {r['method']} | {r['task_label']} | {r['accuracy']:.3f} | {r['n']} | "
            f"{avg_calls_text} | `{r['final_budget_rates']}` |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        "- The selective-remask refinement controller makes dynamic decisions inside the reverse process: after an 8-step scout it re-masks low-confidence generated tokens and spends only the marginal extra budget needed for refinement.",
        "- The risk controller is a stronger but more expensive ablation that restarts at the selected budget; it helps separate routing quality from stateful refinement quality.",
        "- Route distribution is the key evidence: a non-degenerate spread over 8/16/24/32 supports genuine budget allocation rather than binary fallback.",
        "- Paired deltas should be read against the full-budget boundary: an acceptable controller should keep accuracy close to 32-step while reducing average calls.",
    ]
    (dyn / "reports").mkdir(parents=True, exist_ok=True)
    (dyn / "reports" / "Dynamic_Controller_Addendum.md").write_text("\n".join(report_lines), encoding="utf-8")
    (dyn / "tables" / "dynamic_controller_summary.json").write_text(
        json.dumps({"summary_rows": rows, "paired_deltas": deltas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary_rows": len(rows), "paired_delta_rows": len(deltas)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
