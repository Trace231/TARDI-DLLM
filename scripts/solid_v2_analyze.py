#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK_LABELS = {
    "winogrande": "WinoGrande",
    "commonsenseqa": "CommonsenseQA",
    "pubmedqa": "PubMedQA",
    "ceval_computer_network": "C-Eval CN",
    "mmlu_pro": "MMLU-Pro",
    "sciq": "SciQ",
}


MAIN_FILES = {
    "32step": "llada8b_32step_wino_cqa_limit1000_seed23.json",
    "old_adaptive": "llada8b_adaptive_router_wino_cqa_limit1000_seed23.json",
    "forward_aware": "llada8b_forward_aware_wino_cqa_limit1000_seed23.json",
    "calibrated": "llada8b_calibrated_controller_wino_cqa_limit1000_seed23.json",
}


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def ensure_dirs(root):
    for name in ["tables", "figures", "reports"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fields})


def wilson(k, n, z=1.96):
    if n <= 0:
        return "", ""
    ph = k / n
    den = 1 + z * z / n
    cen = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt((ph * (1 - ph) + z * z / (4 * n)) / n) / den
    return cen - half, cen + half


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    x = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) * 0.5**n for i in range(x + 1)))


def method_rows(payload, task):
    return [r for r in payload.get("rows", []) if r.get("task") == task]


def route_rates(rows):
    counts = Counter(r.get("route") or ("fallback" if r.get("fallback_used") else "fixed") for r in rows)
    n = sum(counts.values())
    return {k: v / n for k, v in sorted(counts.items())} if n else {}


def flatten_summary(method, payload):
    out = []
    if not payload:
        return out
    for task, s in payload.get("summary", {}).items():
        n = int(s.get("n", 0))
        acc = float(s.get("accuracy", 0.0))
        k = round(acc * n)
        lo, hi = wilson(k, n)
        rows = method_rows(payload, task)
        routes = route_rates(rows)
        out.append(
            {
                "method": method,
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "accuracy": acc,
                "n": n,
                "correct": k,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "avg_forward_calls": s.get("avg_forward_calls", 32 if method == "32step" else ""),
                "seconds": s.get("seconds", ""),
                "fallback_rate": s.get("fallback_rate", ""),
                "route_rates": json.dumps(s.get("route_rates", routes), ensure_ascii=False),
                "peak_mem_gb": s.get("peak_mem_gb", ""),
            }
        )
    return out


def paired_rows(reference_name, reference, candidate_name, candidate, tasks):
    rows = []
    if not reference or not candidate:
        return rows
    for task in tasks:
        rr = {r.get("id"): r for r in method_rows(reference, task)}
        cr = {r.get("id"): r for r in method_rows(candidate, task)}
        ids = sorted(set(rr) & set(cr))
        both = sum(bool(rr[i].get("correct")) and bool(cr[i].get("correct")) for i in ids)
        ref_only = sum(bool(rr[i].get("correct")) and not bool(cr[i].get("correct")) for i in ids)
        cand_only = sum(not bool(rr[i].get("correct")) and bool(cr[i].get("correct")) for i in ids)
        neither = sum(not bool(rr[i].get("correct")) and not bool(cr[i].get("correct")) for i in ids)
        rows.append(
            {
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "reference": reference_name,
                "candidate": candidate_name,
                "n_intersection": len(ids),
                "both_correct": both,
                "reference_only": ref_only,
                "candidate_only": cand_only,
                "neither": neither,
                "net_candidate_minus_reference": cand_only - ref_only,
                "mcnemar_p": mcnemar_p(ref_only, cand_only),
            }
        )
    return rows


def error_taxonomy_rows(main_payloads):
    rows = []
    comparisons = [
        ("32step", "calibrated", "full_budget_vs_controller"),
        ("old_adaptive", "calibrated", "old_router_vs_controller"),
        ("forward_aware", "calibrated", "uncalibrated_vs_calibrated"),
    ]
    tasks = ["winogrande", "commonsenseqa"]
    for left_name, right_name, comparison in comparisons:
        left = main_payloads.get(left_name)
        right = main_payloads.get(right_name)
        if not left or not right:
            continue
        for task in tasks:
            lr = {r.get("id"): r for r in method_rows(left, task)}
            rr = {r.get("id"): r for r in method_rows(right, task)}
            ids = sorted(set(lr) & set(rr))
            if not ids:
                continue
            buckets = Counter()
            route_buckets = defaultdict(Counter)
            for item_id in ids:
                lc = bool(lr[item_id].get("correct"))
                rc = bool(rr[item_id].get("correct"))
                if lc and rc:
                    bucket = "both_correct"
                elif lc and not rc:
                    bucket = f"{left_name}_only"
                elif not lc and rc:
                    bucket = f"{right_name}_only"
                else:
                    bucket = "both_wrong"
                buckets[bucket] += 1
                route = rr[item_id].get("route") or ("fallback" if rr[item_id].get("fallback_used") else "fixed")
                route_buckets[bucket][route] += 1
            for bucket, count in sorted(buckets.items()):
                route_counts = dict(sorted(route_buckets[bucket].items()))
                rows.append(
                    {
                        "comparison": comparison,
                        "left_method": left_name,
                        "right_method": right_name,
                        "task": task,
                        "task_label": TASK_LABELS.get(task, task),
                        "bucket": bucket,
                        "count": count,
                        "rate": count / len(ids),
                        "n_intersection": len(ids),
                        "right_route_counts": json.dumps(route_counts, ensure_ascii=False),
                    }
                )
    return rows


def sweep_rows(raw_dir):
    rows = []
    for threshold in ["060", "065", "070", "075", "080"]:
        p = raw_dir / f"llada8b_calibrated_sweep_t{threshold}_wino_cqa_limit500_seed23.json"
        payload = load_json(p)
        if not payload:
            continue
        tval = int(threshold) / 100
        for row in flatten_summary(f"calibrated_t{threshold}", payload):
            row["threshold"] = tval
            rows.append(row)
    return rows


def boundary_rows(raw_dir):
    rows = []
    mapping = {
        "8step": "llada8b_8step_boundary_pubmed_ceval_limit300_seed23.json",
        "32step": "llada8b_32step_boundary_pubmed_ceval_limit300_seed23.json",
        "calibrated": "llada8b_calibrated_boundary_pubmed_ceval_limit300_seed23.json",
    }
    for method, name in mapping.items():
        payload = load_json(raw_dir / name)
        rows.extend(flatten_summary(method, payload))
    return rows


def lora_rows(raw_dir, legacy_dir):
    base = load_json(raw_dir / "llada8b_base_final_label_typed_domain_shift_limit100.json") or load_json(
        legacy_dir / "llada8b_final_label_typed_domain_shift_limit100.json"
    )
    lora = load_json(raw_dir / "llada8b_typed_lora_final_label_typed_domain_shift_limit100.json") or load_json(
        legacy_dir / "llada8b_domain_mix_final_typed_lora_r8_steps50_final_label_typed_domain_shift_limit100.json"
    )
    ar = load_json(raw_dir / "qwen25_7b_final_label_typed_domain_shift_limit100.json") or load_json(
        legacy_dir / "qwen25_7b_final_label_typed_domain_shift_limit100.json"
    )
    ar_lora = load_json(raw_dir / "qwen25_7b_typed_lora_final_label_typed_domain_shift_limit100.json") or load_json(
        legacy_dir / "qwen25_7b_typed_lora_final_label_typed_domain_shift_limit100.json"
    )
    rows = []
    by_payload = {"llada_base": base, "llada_lora": lora, "qwen_base": ar, "qwen_lora": ar_lora}
    for method, payload in by_payload.items():
        rows.extend(flatten_summary(method, payload))
    if base and lora:
        tasks = sorted(set(base.get("summary", {})) & set(lora.get("summary", {})))
        for task in tasks:
            b = base["summary"][task]["accuracy"]
            l = lora["summary"][task]["accuracy"]
            rows.append(
                {
                    "method": "ddm_lora_gain",
                    "task": task,
                    "task_label": TASK_LABELS.get(task, task),
                    "accuracy": l - b,
                    "n": min(base["summary"][task].get("n", 0), lora["summary"][task].get("n", 0)),
                    "correct": "",
                    "wilson_lo": "",
                    "wilson_hi": "",
                    "avg_forward_calls": "",
                    "seconds": "",
                    "fallback_rate": "",
                    "route_rates": "",
                    "peak_mem_gb": "",
                }
            )
    if ar and ar_lora:
        tasks = sorted(set(ar.get("summary", {})) & set(ar_lora.get("summary", {})))
        for task in tasks:
            b = ar["summary"][task]["accuracy"]
            l = ar_lora["summary"][task]["accuracy"]
            rows.append(
                {
                    "method": "ar_lora_gain",
                    "task": task,
                    "task_label": TASK_LABELS.get(task, task),
                    "accuracy": l - b,
                    "n": min(ar["summary"][task].get("n", 0), ar_lora["summary"][task].get("n", 0)),
                    "correct": "",
                    "wilson_lo": "",
                    "wilson_hi": "",
                    "avg_forward_calls": "",
                    "seconds": "",
                    "fallback_rate": "",
                    "route_rates": "",
                    "peak_mem_gb": "",
                }
            )
    return rows


def confidence_bins(payload):
    rows = []
    if not payload:
        return rows
    bins = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for task in sorted(payload.get("summary", {})):
        task_rows = method_rows(payload, task)
        for lo, hi in bins:
            group = []
            for r in task_rows:
                probe = r.get("probe") or {}
                top = probe.get("top_prob")
                if top is not None and lo <= float(top) < hi:
                    group.append(r)
            if group:
                rows.append(
                    {
                        "task": task,
                        "task_label": TASK_LABELS.get(task, task),
                        "bin": f"[{lo:.1f},{hi:.1f})",
                        "lo": lo,
                        "hi": hi,
                        "n": len(group),
                        "accuracy": sum(bool(r.get("correct")) for r in group) / len(group),
                    }
                )
    return rows


def trace_metrics(trace_payload):
    rows = []
    if not trace_payload:
        return rows
    for task in sorted(trace_payload.get("summary", {})):
        vals = []
        flips = []
        late = []
        for row in method_rows(trace_payload, task):
            labels = []
            first = None
            final_pred = row.get("pred")
            for segment in row.get("traces", []):
                for item in sorted(segment.get("trace", []), key=lambda x: x.get("step", 0)):
                    pred = item.get("pred")
                    if pred:
                        labels.append(pred)
                        if first is None and final_pred and pred == final_pred:
                            first = item.get("step")
            if first is not None:
                vals.append(first)
            flips.append(sum(1 for a, b in zip(labels, labels[1:]) if a != b))
            late.append(1 if len(labels) >= 2 and labels[-1] != labels[-2] else 0)
        if vals or flips:
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS.get(task, task),
                    "n": trace_payload["summary"][task].get("n", ""),
                    "mean_first_final_step": sum(vals) / len(vals) if vals else "",
                    "mean_flip_count": sum(flips) / len(flips) if flips else "",
                    "late_instability_rate": sum(late) / len(late) if late else "",
                }
            )
    return rows


def plot_accuracy_cost(main_rows, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for row in main_rows:
        calls = row.get("avg_forward_calls")
        if calls == "" or calls is None:
            continue
        ax.scatter(float(calls), float(row["accuracy"]), s=60)
        ax.text(float(calls), float(row["accuracy"]), f"{row['method']} {TASK_LABELS.get(row['task'], row['task'])}", fontsize=8)
    ax.set_xlabel("Average forward calls")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy-cost Pareto")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_threshold(sweep, out):
    if not sweep:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    by_task = defaultdict(list)
    for row in sweep:
        by_task[row["task"]].append(row)
    for task, rows in by_task.items():
        rows = sorted(rows, key=lambda r: r["threshold"])
        label = TASK_LABELS.get(task, task)
        axes[0].plot([r["threshold"] for r in rows], [float(r["accuracy"]) for r in rows], marker="o", label=label)
        axes[1].plot([r["threshold"] for r in rows], [float(r["avg_forward_calls"]) for r in rows], marker="o", label=label)
    axes[0].set_title("Accuracy")
    axes[1].set_title("Average calls")
    for ax in axes:
        ax.set_xlabel("Binary threshold")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_routes(main_payload, out):
    if not main_payload:
        return
    rows = []
    for task in sorted(main_payload.get("summary", {})):
        rates = route_rates(method_rows(main_payload, task))
        for route, rate in rates.items():
            rows.append((TASK_LABELS.get(task, task), route, rate))
    if not rows:
        return
    labels = sorted(set(r[0] for r in rows))
    routes = sorted(set(r[1] for r in rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0.0] * len(labels)
    for route in routes:
        vals = [next((rate for label, rt, rate in rows if label == lab and rt == route), 0.0) for lab in labels]
        ax.bar(labels, vals, bottom=bottom, label=route)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylim(0, 1)
    ax.set_ylabel("Route rate")
    ax.set_title("Calibrated route distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_confidence(conf_rows, out):
    if not conf_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for task in sorted(set(r["task"] for r in conf_rows)):
        rows = [r for r in conf_rows if r["task"] == task]
        ax.plot([r["bin"] for r in rows], [float(r["accuracy"]) for r in rows], marker="o", label=TASK_LABELS.get(task, task))
    ax.set_xlabel("Probe top-prob bin")
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe confidence vs accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_trace(trace_rows, out):
    if not trace_rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r["task_label"] for r in trace_rows], [float(r["mean_first_final_step"] or 0) for r in trace_rows])
    ax.set_ylabel("Mean first final-label step")
    ax.set_title("Trajectory hitting time")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def report_lines(summary):
    lines = [
        "# LLaDA Solid Experiment Report",
        "",
        "## 1. Problem and Thesis",
        "",
        "我们不把贡献写成“改 LLaDA 架构”。主张是：LLaDA 的 masked diffusion trajectory 暴露了不同任务、不同样本的反向去噪预算需求，因此可以用 task/sample-aware controller 在 accuracy-cost 之间做风险受控分配。",
        "",
        "核心方法是 **Calibrated Forward-aware Risk Controller**：先用低预算 probe 估计 posterior confidence 与 label disagreement，再按风险路由到 cheap / medium / full reverse budget。LoRA 只作为辅助审计，用来回答普通下游微调是否能解决同一类行为边界。",
        "",
        "## 2. Exploration Process",
        "",
        "1. 先跑 typed final-label benchmark，避免 prompt 格式混淆。",
        "2. 再做 old adaptive router，发现它很省算力，但 WinoGrande 上会越过 full-budget 边界。",
        "3. 接着做 forward-aware probe，把早期 confidence、fallback 与最终正确性关联起来。",
        "4. 最后加入 calibration threshold 与 multi-choice disagreement policy，形成主方法。",
        "5. 用 PubMedQA/C-Eval 作为负例边界：证明 controller 不是万能调参，而是在可由 reverse budget 影响的区域有效。",
        "",
        "## 3. Method",
        "",
        "Controller 的 inference loop 可以抽象为三段：",
        "",
        "- **Forward probe**：用 cheap reverse trajectory 得到初始 label posterior 与 top probability。",
        "- **Scout decision**：检测 binary 任务中的低置信度、近邻混淆，以及 multi-choice disagreement。",
        "- **Calibrated route**：在 cheap / medium / full 预算之间选择；高风险 binary 样本保守 fallback，多选任务默认忽略轻微 disagreement，减少不必要的 full-budget 调用。",
        "",
        "## 4. Mathematical Idea",
        "",
        "离散扩散可以看成 token 状态空间上的连续时间马尔可夫链或其离散化近似。反向生成时，模型给出近似 posterior / score，用有限步数从 masked/corrupted state 回到数据分布。这里的关键量不是单个 token 的局部置信度，而是最终 label 的 hitting time：达到并稳定在最终答案所需的反向步数。",
        "",
        "因此 controller 优化的是一个 cost-sensitive risk：",
        "",
        "`min_pi E[1{y_hat_pi(x) != y} + lambda C(pi, x)]`",
        "",
        "其中策略 `pi` 根据 probe trajectory 的 uncertainty 选择预算。若某任务 hitting time 分布更晚，比如 WinoGrande，策略应更保守；若某任务早期 label posterior 已稳定，比如 CommonsenseQA，则可以更激进地节省预算。",
        "",
        "## 5. Main 1000 Results",
        "",
        "| Method | Task | Acc | 95% CI | Avg Calls | Seconds | Route/Fallback |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in summary["main_1000"]:
        ci = f"[{float(r['wilson_lo']):.3f}, {float(r['wilson_hi']):.3f}]" if r["wilson_lo"] != "" else ""
        calls = f"{float(r['avg_forward_calls']):.3f}" if r["avg_forward_calls"] != "" else ""
        sec = f"{float(r['seconds']):.1f}" if r["seconds"] != "" else ""
        lines.append(f"| {r['method']} | {r['task_label']} | {float(r['accuracy']):.3f} | {ci} | {calls} | {sec} | `{r['route_rates']}` |")
    lines += [
        "",
        "## Paired McNemar",
        "",
        "| Task | Reference | Candidate | n | Ref Only | Cand Only | Net | p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary["paired_mcnemar"]:
        lines.append(
            f"| {r['task_label']} | {r['reference']} | {r['candidate']} | {r['n_intersection']} | {r['reference_only']} | {r['candidate_only']} | {r['net_candidate_minus_reference']} | {float(r['mcnemar_p']):.4f} |"
        )
    lines += [
        "",
        "## 6. Threshold Robustness",
        "",
        "| Threshold | Task | Acc | Avg Calls | Route Rates |",
        "|---:|---|---:|---:|---|",
    ]
    for r in summary["threshold_sweep"]:
        calls = f"{float(r['avg_forward_calls']):.3f}" if r["avg_forward_calls"] != "" else ""
        lines.append(f"| {float(r['threshold']):.2f} | {r['task_label']} | {float(r['accuracy']):.3f} | {calls} | `{r['route_rates']}` |")
    lines += [
        "",
        "## 7. Boundary Negative Cases",
        "",
        "| Method | Task | Acc | Avg Calls | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for r in summary["boundary_negative_cases"]:
        calls = f"{float(r['avg_forward_calls']):.3f}" if r["avg_forward_calls"] != "" else ""
        note = "label-prior / knowledge-boundary probe"
        lines.append(f"| {r['method']} | {r['task_label']} | {float(r['accuracy']):.3f} | {calls} | {note} |")
    lines += [
        "",
        "## 8. LoRA Gain Audit",
        "",
        "| Method | Task | Value | n |",
        "|---|---|---:|---:|",
    ]
    for r in summary["lora_gain_audit"]:
        val = float(r["accuracy"]) if r["accuracy"] != "" else 0.0
        lines.append(f"| {r['method']} | {r['task_label']} | {val:.3f} | {r.get('n','')} |")
    lines += [
        "",
        "## 9. Trajectory and Error Analysis",
        "",
        "| Task | n | Mean First Final Step | Mean Flip Count | Late Instability |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary["trajectory_metrics"]:
        lines.append(
            f"| {r['task_label']} | {r['n']} | {float(r['mean_first_final_step'] or 0):.2f} | {float(r['mean_flip_count'] or 0):.3f} | {float(r['late_instability_rate'] or 0):.3f} |"
        )
    lines += [
        "",
        "### Error Taxonomy",
        "",
        "| Comparison | Task | Bucket | Count | Rate | Controller Route Counts |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in summary.get("error_taxonomy", []):
        lines.append(
            f"| {r['comparison']} | {r['task_label']} | {r['bucket']} | {r['count']} | {float(r['rate']):.3f} | `{r['right_route_counts']}` |"
        )
    lines += [
        "",
        "## 10. Limitations and Final Takeaway",
        "",
        "- 不声称修改 LLaDA 架构；这是 inference-loop/controller 改进。",
        "- WinoGrande 用于展示 high-risk binary semantic binding 需要保守预算控制。",
        "- CommonsenseQA 用于展示 low-risk multi-choice 可减少不必要 fallback。",
        "- PubMedQA/C-Eval 作为边界负例：采样预算不能弥补 label bias 或知识缺口。",
        "- LoRA gain audit 不追求 SOTA，只验证“普通参数高效微调是否能替代 trajectory-aware controller”。",
        "",
        "最终可讲成一句话：**我们没有把 LLaDA 改成另一个模型，而是把 masked diffusion 的反向轨迹从一次性生成过程变成可观测、可校准、可控成本的决策过程。**",
    ]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/domain_shift/task_aware/solid_v2")
    ap.add_argument("--legacy-domain-root", default="results/domain_shift")
    args = ap.parse_args()

    root = Path(args.root)
    raw_dir = root / "raw"
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    reports_dir = root / "reports"
    ensure_dirs(root)

    main_payloads = {method: load_json(raw_dir / filename) for method, filename in MAIN_FILES.items()}
    main_rows = []
    for method, payload in main_payloads.items():
        main_rows.extend(flatten_summary(method, payload))
    paired = []
    tasks = ["winogrande", "commonsenseqa"]
    calibrated = main_payloads.get("calibrated")
    for ref_name in ["32step", "old_adaptive", "forward_aware"]:
        paired.extend(paired_rows(ref_name, main_payloads.get(ref_name), "calibrated", calibrated, tasks))

    sweep = sweep_rows(raw_dir)
    boundary = boundary_rows(raw_dir)
    lora = lora_rows(raw_dir, Path(args.legacy_domain_root))
    errors = error_taxonomy_rows(main_payloads)
    trace_payload = load_json(raw_dir / "llada8b_calibrated_trace_wino_cqa_limit200_seed23.json") or load_json(
        raw_dir / "llada8b_calibrated_controller_trace_wino_cqa_limit200_seed23.json"
    )
    trace = trace_metrics(trace_payload)
    conf = confidence_bins(calibrated)

    write_csv(tables_dir / "main_1000_summary.csv", main_rows, list(main_rows[0].keys()) if main_rows else [])
    write_csv(tables_dir / "paired_mcnemar.csv", paired, list(paired[0].keys()) if paired else [])
    write_csv(tables_dir / "threshold_sweep.csv", sweep, list(sweep[0].keys()) if sweep else [])
    write_csv(tables_dir / "boundary_negative_cases.csv", boundary, list(boundary[0].keys()) if boundary else [])
    write_csv(tables_dir / "lora_gain_audit.csv", lora, list(lora[0].keys()) if lora else [])
    write_csv(tables_dir / "trajectory_metrics.csv", trace, list(trace[0].keys()) if trace else [])
    write_csv(tables_dir / "confidence_accuracy.csv", conf, list(conf[0].keys()) if conf else [])
    write_csv(tables_dir / "error_taxonomy.csv", errors, list(errors[0].keys()) if errors else [])

    plot_accuracy_cost(main_rows, figures_dir / "accuracy_cost_pareto.png")
    plot_threshold(sweep, figures_dir / "threshold_accuracy_cost.png")
    plot_routes(calibrated, figures_dir / "route_distribution.png")
    plot_confidence(conf, figures_dir / "confidence_accuracy.png")
    plot_trace(trace, figures_dir / "trajectory_stabilization.png")

    summary = {
        "main_1000": main_rows,
        "paired_mcnemar": paired,
        "threshold_sweep": sweep,
        "boundary_negative_cases": boundary,
        "lora_gain_audit": lora,
        "error_taxonomy": errors,
        "trajectory_metrics": trace,
        "confidence_accuracy": conf,
    }
    (tables_dir / "solid_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    report = "\n".join(report_lines(summary))
    (reports_dir / "LLaDA_Solid_Experiment_Report.md").write_text(report)
    (reports_dir / "LLaDA_Solid_Experiment_Report_zh.md").write_text(report)
    print(json.dumps({"root": str(root), "tables": str(tables_dir), "figures": str(figures_dir), "reports": str(reports_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
