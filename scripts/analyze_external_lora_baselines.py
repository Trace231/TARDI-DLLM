#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


EXTERNAL_METHODS = {
    "llada_rslora_vanilla_fixed32": "llada_rslora_vanilla_fixed32_limit{limit}_seed{seed}.json",
    "llada_dora_vanilla_fixed32": "llada_dora_vanilla_fixed32_limit{limit}_seed{seed}.json",
    "llada_loraplus_vanilla_fixed32": "llada_loraplus_vanilla_fixed32_limit{limit}_seed{seed}.json",
    "llada_nara_vanilla_fixed32": "llada_nara_vanilla_fixed32_limit{limit}_seed{seed}.json",
    "llada_nara_choice_noise_fixed32": "llada_nara_choice_noise_fixed32_limit{limit}_seed{seed}.json",
    "llada_nara_r32_vanilla_fixed32": "llada_nara_r32_vanilla_fixed32_limit{limit}_seed{seed}.json",
    "llada_nara_r32_choice_noise_fixed32": "llada_nara_r32_choice_noise_fixed32_limit{limit}_seed{seed}.json",
    "llada_nara_official_targets_vanilla_fixed32": "llada_nara_official_targets_vanilla_fixed32_limit{limit}_seed{seed}.json",
    "llada_nara_official_targets_choice_noise_fixed32": "llada_nara_official_targets_choice_noise_fixed32_limit{limit}_seed{seed}.json",
}

CHOICE_METHODS = {
    "llada_base_fixed32": "llada_base_fixed32_limit50_seed23.json",
    "llada_vanilla_lora_fixed32": "llada_vanilla_fixed32_limit50_seed23.json",
    "llada_label_lora_fixed32": "llada_label_fixed32_limit50_seed23.json",
    "llada_choice_noise_lora_fixed32": "llada_choice_noise_fixed32_limit50_seed23.json",
    "llada_vanilla_lora_controller": "llada_vanilla_controller_limit50_seed23.json",
}


def load(path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def summarize_payload(method, payload):
    rows = payload.get("rows", [])
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    task_rows = []
    for task, vals in by_task.items():
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
    total_correct = sum(r["correct"] for r in task_rows)
    total_n = sum(r["n"] for r in task_rows)
    return {
        "method": method,
        "tasks": len(task_rows),
        "total_correct": total_correct,
        "total_n": total_n,
        "macro_accuracy": macro,
        "micro_accuracy": total_correct / total_n if total_n else 0.0,
        "status": "complete",
    }, task_rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_report(macro_rows, task_rows, out_root):
    complete = [r for r in macro_rows if r.get("status") == "complete"]
    complete.sort(key=lambda r: r["macro_accuracy"], reverse=True)
    by_method = {r["method"]: r for r in complete}
    external_names = {
        "llada_rslora_vanilla_fixed32",
        "llada_dora_vanilla_fixed32",
        "llada_loraplus_vanilla_fixed32",
        "llada_nara_vanilla_fixed32",
        "llada_nara_r32_vanilla_fixed32",
        "llada_nara_official_targets_vanilla_fixed32",
    }
    our_names = {
        "llada_vanilla_lora_controller",
        "llada_choice_noise_lora_fixed32",
        "llada_nara_choice_noise_fixed32",
        "llada_nara_r32_choice_noise_fixed32",
        "llada_nara_official_targets_choice_noise_fixed32",
    }
    completed_external = [r for r in complete if r["method"] in external_names]
    completed_ours = [r for r in complete if r["method"] in our_names]
    best_external = max(completed_external, key=lambda r: r["macro_accuracy"], default=None)
    best_ours = max(completed_ours, key=lambda r: r["macro_accuracy"], default=None)
    lines = ["# External Improved LoRA Baseline Comparison\n\n"]
    lines.append("This report compares external/improved LoRA baselines under the same LLaDA fixed-label protocol when their raw outputs are available.\n\n")
    lines.append("## Macro Results\n\n")
    lines.append("| Method | Status | Macro Acc. | Correct / N |\n")
    lines.append("|---|---|---:|---:|\n")
    for r in macro_rows:
        acc = f"{r['macro_accuracy']:.3f}" if r.get("status") == "complete" else ""
        cn = f"{r.get('total_correct','')} / {r.get('total_n','')}" if r.get("status") == "complete" else ""
        lines.append(f"| {r['method']} | {r.get('status','missing')} | {acc} | {cn} |\n")
    if complete:
        best = complete[0]
        lines.append(f"\nCurrent best completed method: `{best['method']}` with macro accuracy `{best['macro_accuracy']:.3f}`.\n")
    lines.append("\n## Win/Loss Audit\n\n")
    if best_external and best_ours:
        delta = best_ours["macro_accuracy"] - best_external["macro_accuracy"]
        relation = "wins over" if delta > 0 else "ties" if abs(delta) < 1e-9 else "does not beat"
        lines.append(
            f"Best ours `{best_ours['method']}` ({best_ours['macro_accuracy']:.3f}) "
            f"{relation} best completed external baseline `{best_external['method']}` "
            f"({best_external['macro_accuracy']:.3f}); delta={delta:+.3f}.\n"
        )
    elif not best_external:
        lines.append("External improved LoRA baselines are not complete yet, so no superiority claim is allowed.\n")
    else:
        lines.append("Our comparison rows are missing; inspect raw outputs before making a claim.\n")

    nara_cn = by_method.get("llada_nara_choice_noise_fixed32")
    nara_base = by_method.get("llada_nara_vanilla_fixed32")
    if nara_cn and nara_base:
        delta = nara_cn["macro_accuracy"] - nara_base["macro_accuracy"]
        lines.append(
            f"NaRA-style choice-noise vs NaRA-style vanilla delta: {delta:+.3f}. "
            "Positive means the fixed-label objective improves a dLLM-specific adapter.\n"
        )
    missing = [r["method"] for r in macro_rows if r.get("status") != "complete"]
    if missing:
        lines.append("\nMissing methods still need GPU runs:\n\n")
        for m in missing:
            lines.append(f"- `{m}`\n")
    lines.append("\n## Interpretation Guardrail\n\n")
    lines.append("NaRA-style here is a mechanism-level reproduction using `B C(lambda) A x`, where `C(lambda)=I+eta F(GaussianFourier(lambda))` is produced by a shared hypernetwork; it is not claimed to be the official authors' code unless the official implementation is later plugged in.\n")
    lines.append(f"\nOutputs live in `{out_root}`.\n")
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-root", default="results/domain_shift/task_aware/lora_external_v1")
    ap.add_argument("--choice-root", default="results/domain_shift/task_aware/choice_noise_v1")
    ap.add_argument("--out-root", default="results/domain_shift/task_aware/lora_external_v1")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    external_root = Path(args.external_root)
    choice_root = Path(args.choice_root)
    out_root = Path(args.out_root)

    macro_rows = []
    task_rows = []
    for method, filename in CHOICE_METHODS.items():
        payload = load(choice_root / "raw" / filename)
        if payload is None:
            macro_rows.append({"method": method, "status": "missing"})
            continue
        macro, tasks = summarize_payload(method, payload)
        macro_rows.append(macro)
        task_rows.extend(tasks)

    for method, template in EXTERNAL_METHODS.items():
        filename = template.format(limit=args.limit, seed=args.seed)
        payload = load(external_root / "raw" / filename)
        if payload is None:
            macro_rows.append({"method": method, "status": "missing"})
            continue
        macro, tasks = summarize_payload(method, payload)
        macro_rows.append(macro)
        task_rows.extend(tasks)

    macro_rows.sort(key=lambda r: (r.get("status") != "complete", -float(r.get("macro_accuracy", -1))))
    table_dir = out_root / "tables"
    report_dir = out_root / "reports"
    write_csv(table_dir / "external_lora_macro_summary.csv", macro_rows, ["method", "status", "tasks", "total_correct", "total_n", "macro_accuracy", "micro_accuracy"])
    write_csv(table_dir / "external_lora_task_summary.csv", task_rows, ["method", "task", "accuracy", "correct", "n"])
    (table_dir / "external_lora_summary.json").write_text(json.dumps({"macro_rows": macro_rows, "task_rows": task_rows}, indent=2, ensure_ascii=False))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "External_LoRA_Baseline_Report.md").write_text(make_report(macro_rows, task_rows, out_root), encoding="utf-8")
    print(json.dumps({"macro_rows": macro_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
