#!/usr/bin/env python3
import ast
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/domain_shift/task_aware")
OUT = Path("writing")
FIG = OUT / "figures"
TAB = OUT / "tables"


TASK_ZH = {
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

METHOD_ZH = {
    "base_fixed32": "基础模型",
    "vanilla_lora_fixed32": "旧版LoRA",
    "tardi_lora_balanced_r8": "任务均衡LoRA",
    "tardi_lora_balanced_r8_highnoise": "TARDI-LoRA",
    "tardi_lora_balanced_r16": "TARDI-LoRA(r16)",
    "tardi_lora_balanced_r16_highnoise": "r16高噪声",
    "tasknara_r8_highnoise": "任务条件消融",
    "tasknara_residual_vanilla_highnoise": "残差任务条件消融",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "PingFang SC",
            "Hiragino Sans GB",
            "Heiti SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.dpi": 160,
        "savefig.dpi": 260,
        "axes.unicode_minus": False,
    })


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(x, default=0.0):
    if x in (None, ""):
        return default
    return float(x)


def parse_route(s):
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def savefig(name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close()


def copy_core_tables():
    TAB.mkdir(parents=True, exist_ok=True)
    sources = {
        "lora_macro_summary.csv": ROOT / "lora_opt_v1/tables/lora_opt_macro_summary.csv",
        "lora_task_summary.csv": ROOT / "lora_opt_v1/tables/lora_opt_task_summary.csv",
        "controller_main_1000.csv": ROOT / "solid_v2/tables/main_1000_summary.csv",
        "controller_mcnemar.csv": ROOT / "solid_v2/tables/paired_mcnemar.csv",
        "threshold_sweep.csv": ROOT / "solid_v2/tables/threshold_sweep.csv",
        "trajectory_metrics.csv": ROOT / "solid_v2/tables/trajectory_metrics.csv",
        "boundary_cases.csv": ROOT / "solid_v2/tables/boundary_negative_cases.csv",
        "coverage_summary.csv": ROOT / "solid_v2/coverage_addendum/tables/coverage_summary.csv",
        "step_sweep.csv": ROOT / "solid_v2/step_sweep_limit20_4to32/tables/step_sweep_4to32_by_dataset_limit20_seed23.csv",
        "old_lora_gain_audit.csv": ROOT / "solid_v2/tables/lora_gain_audit.csv",
        "qwen_lora_9task_limit50_seed23.csv": ROOT / "solid_v2/tables/qwen_lora_9task_limit50_seed23.csv",
        "trajectory_metrics_limit500_stride1.csv": ROOT / "solid_v2/tables/trajectory_metrics_limit500_stride1.csv",
        "step_sweep_9task_limit50_seed23.csv": ROOT / "solid_v2/step_sweep_limit50_4to32/tables/step_sweep_9task_limit50_seed23.csv",
    }
    for name, src in sources.items():
        if src.exists():
            (TAB / name).write_text(src.read_text(), encoding="utf-8")


def fig_lora_macro():
    rows = read_csv(TAB / "lora_macro_summary.csv")
    wanted = [
        "base_fixed32",
        "vanilla_lora_fixed32",
        "tardi_lora_balanced_r8",
        "tardi_lora_balanced_r8_highnoise",
        "tardi_lora_balanced_r16",
    ]
    data = [r for m in wanted for r in rows if r["method"] == m]
    labels = [METHOD_ZH[r["method"]] for r in data]
    vals = [fnum(r["macro_accuracy"]) for r in data]
    colors = ["#9aa0a6", "#6c8ebf", "#7bb274", "#d55e00", "#e69f00", "#b8a1d9", "#c6b8a8"]
    plt.figure(figsize=(8.4, 4.2))
    bars = plt.bar(range(len(vals)), vals, color=colors, edgecolor="#333333", linewidth=0.5)
    plt.ylim(0.70, 0.79)
    plt.ylabel("宏平均准确率")
    plt.xticks(range(len(vals)), labels, rotation=24, ha="right")
    plt.title("训练侧适配：TARDI-LoRA 相比基础模型与旧版 LoRA 的提升")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    savefig("lora_macro_comparison")


def fig_lora_task_heatmap():
    rows = read_csv(TAB / "lora_task_summary.csv")
    methods = ["base_fixed32", "vanilla_lora_fixed32", "tardi_lora_balanced_r8_highnoise", "tardi_lora_balanced_r16"]
    tasks = ["mmlu_pro", "pubmedqa", "ceval_computer_network", "sciq", "winogrande", "commonsenseqa", "arc_challenge", "hellaswag", "boolq"]
    lookup = {(r["method"], r["task"]): fnum(r["accuracy"]) for r in rows}
    mat = np.array([[lookup.get((m, t), np.nan) for t in tasks] for m in methods])
    plt.figure(figsize=(9.2, 3.8))
    im = plt.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.35, vmax=0.95)
    plt.yticks(range(len(methods)), [METHOD_ZH.get(m, m) for m in methods])
    plt.xticks(range(len(tasks)), [TASK_ZH[t] for t in tasks], rotation=30, ha="right")
    plt.title("逐任务准确率热力图")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = plt.colorbar(im, fraction=0.025, pad=0.02)
    cbar.set_label("准确率")
    savefig("lora_task_heatmap")


def fig_old_lora_gain():
    rows = read_csv(TAB / "old_lora_gain_audit.csv")
    tasks = ["mmlu_pro", "pubmedqa", "ceval_computer_network", "sciq", "winogrande", "commonsenseqa"]
    ar = {r["task"]: fnum(r["accuracy"]) for r in rows if r["method"] == "ar_lora_gain"}
    ddm = {r["task"]: fnum(r["accuracy"]) for r in rows if r["method"] == "ddm_lora_gain"}
    x = np.arange(len(tasks))
    w = 0.36
    plt.figure(figsize=(8.2, 3.8))
    plt.bar(x - w/2, [ar.get(t, 0) for t in tasks], width=w, label="自回归模型 LoRA 增益", color="#4c78a8")
    plt.bar(x + w/2, [ddm.get(t, 0) for t in tasks], width=w, label="旧版扩散模型 LoRA 增益", color="#f58518")
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(x, [TASK_ZH[t] for t in tasks], rotation=25, ha="right")
    plt.ylabel("准确率增量")
    plt.title("旧版 LoRA 审计：自回归模型更容易吸收固定标签监督")
    plt.legend(frameon=False, fontsize=8)
    savefig("old_lora_gain_audit")


def fig_ar_vs_tardi_lora_gain():
    lora_rows = read_csv(TAB / "lora_task_summary.csv")
    qwen_9task = TAB / "qwen_lora_9task_limit50_seed23.csv"
    if qwen_9task.exists():
        qwen_rows = [r for r in read_csv(qwen_9task) if r["task"] != "macro"]
        tasks = [r["task"] for r in qwen_rows]
        qwen = {r["task"]: (fnum(r["qwen_base_accuracy"]), fnum(r["qwen_lora_accuracy"])) for r in qwen_rows}
    else:
        audit_rows = read_csv(TAB / "old_lora_gain_audit.csv")
        tasks = ["mmlu_pro", "pubmedqa", "ceval_computer_network", "sciq", "winogrande", "commonsenseqa"]
        audit = {(r["method"], r["task"]): fnum(r["accuracy"]) for r in audit_rows}
        qwen = {t: (audit[("qwen_base", t)], audit[("qwen_lora", t)]) for t in tasks}
    lora = {(r["method"], r["task"]): fnum(r["accuracy"]) for r in lora_rows}
    old_ddm_gain = [lora[("vanilla_lora_fixed32", t)] - lora[("base_fixed32", t)] for t in tasks]
    tardi_gain = [lora[("tardi_lora_balanced_r8_highnoise", t)] - lora[("base_fixed32", t)] for t in tasks]
    ar_gain = [qwen[t][1] - qwen[t][0] for t in tasks]
    x = np.arange(len(tasks))
    w = 0.25
    plt.figure(figsize=(9.0, 4.0))
    plt.bar(x - w, old_ddm_gain, width=w, label="旧版 DDM LoRA", color="#bab0ac", edgecolor="#333333", linewidth=0.4)
    plt.bar(x, tardi_gain, width=w, label="TARDI-LoRA", color="#d55e00", edgecolor="#333333", linewidth=0.4)
    plt.bar(x + w, ar_gain, width=w, label="Qwen LoRA", color="#4c78a8", edgecolor="#333333", linewidth=0.4)
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(x, [TASK_ZH[t] for t in tasks], rotation=25, ha="right")
    plt.ylabel("相对各自基础模型的准确率增量")
    plt.title("旧版 DDM LoRA、TARDI-LoRA 与自回归 LoRA 的增益对比")
    plt.legend(frameon=False, fontsize=8)
    savefig("ar_vs_tardi_lora_gain")


def fig_controller_accuracy_cost():
    rows = read_csv(TAB / "controller_main_1000.csv")
    colors = {"winogrande": "#4c78a8", "commonsenseqa": "#f58518"}
    markers = {"32step": "o", "old_adaptive": "x", "forward_aware": "^", "calibrated": "s"}
    method_names = {
        "32step": "满预算",
        "old_adaptive": "早期自适应",
        "forward_aware": "前向感知",
        "calibrated": "校准控制器",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharex=True)
    local_offsets = {
        ("winogrande", "forward_aware"): (0.35, 0.0010),
        ("winogrande", "calibrated"): (0.35, -0.0010),
        ("commonsenseqa", "old_adaptive"): (0.35, -0.0005),
        ("commonsenseqa", "calibrated"): (0.35, 0.0005),
    }
    for ax, task in zip(axes, ["winogrande", "commonsenseqa"]):
        task_rows = [r for r in rows if r["task"] == task]
        for r in task_rows:
            method = r["method"]
            calls = fnum(r["avg_forward_calls"])
            acc = fnum(r["accuracy"])
            ax.scatter(calls, acc, s=78, color=colors.get(task, "#777777"), marker=markers.get(method, "o"), edgecolor="#333333", linewidth=0.5)
            dx, dy = local_offsets.get((task, method), (0.35, 0.0))
            ax.text(calls + dx, acc + dy, method_names.get(method, method), fontsize=8, va="center")
        ax.set_title(TASK_ZH[task])
        ax.set_xlabel("平均前向调用次数")
        ax.set_xlim(6, 35)
        if task == "winogrande":
            ax.set_ylim(0.72, 0.765)
        else:
            ax.set_ylim(0.81, 0.824)
    axes[0].set_ylabel("准确率")
    fig.suptitle("主实验：准确率与推理成本", y=1.02)
    savefig("controller_accuracy_cost")


def fig_route_distribution():
    rows = [r for r in read_csv(TAB / "controller_main_1000.csv") if r["method"] == "calibrated"]
    route_keys = ["8", "8->32", "32"]
    labels = [TASK_ZH[r["task"]] for r in rows]
    values = []
    for r in rows:
        route = parse_route(r["route_rates"])
        values.append([route.get(k, 0.0) for k in route_keys])
    values = np.array(values)
    plt.figure(figsize=(6.4, 3.4))
    bottom = np.zeros(len(rows))
    colors = ["#72b7b2", "#eeca3b", "#e45756"]
    names = ["低预算接受", "追加再修", "满预算"]
    for i, key in enumerate(route_keys):
        plt.bar(labels, values[:, i], bottom=bottom, color=colors[i], label=names[i], edgecolor="#333333", linewidth=0.4)
        bottom += values[:, i]
    plt.ylabel("路由比例")
    plt.ylim(0, 1)
    plt.title("校准控制器的路由分布")
    plt.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    savefig("controller_route_distribution")


def fig_threshold_sweep():
    rows = read_csv(TAB / "threshold_sweep.csv")
    tasks = ["winogrande", "commonsenseqa"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), sharex=True)
    for task in tasks:
        rs = sorted([r for r in rows if r["task"] == task], key=lambda r: fnum(r["threshold"]))
        th = [fnum(r["threshold"]) for r in rs]
        acc = [fnum(r["accuracy"]) for r in rs]
        calls = [fnum(r["avg_forward_calls"]) for r in rs]
        axes[0].plot(th, acc, marker="o", label=TASK_ZH[task])
        axes[1].plot(th, calls, marker="o", label=TASK_ZH[task])
    axes[0].set_ylabel("准确率")
    axes[0].set_title("阈值-准确率")
    axes[1].set_ylabel("平均调用次数")
    axes[1].set_title("阈值-成本")
    for ax in axes:
        ax.set_xlabel("二分类阈值")
        ax.legend(frameon=False, fontsize=8)
    savefig("threshold_robustness")


def fig_trajectory_metrics():
    path = TAB / "trajectory_metrics_limit500_stride1.csv"
    rows = read_csv(path if path.exists() else TAB / "trajectory_metrics.csv")
    labels = [TASK_ZH[r["task"]] for r in rows]
    x = np.arange(len(rows))
    w = 0.28
    plt.figure(figsize=(6.4, 3.8))
    plt.bar(x - w, [fnum(r["mean_first_final_step"]) for r in rows], width=w, label="首次到达最终答案步数", color="#4c78a8")
    plt.bar(x, [fnum(r["mean_flip_count"]) for r in rows], width=w, label="标签翻转次数", color="#f58518")
    plt.bar(x + w, [fnum(r["late_instability_rate"]) for r in rows], width=w, label="后期不稳定率", color="#54a24b")
    plt.xticks(x, labels)
    plt.ylabel("统计量")
    plt.title("轨迹统计：不同任务的反向扩散需求不同")
    plt.legend(frameon=False, fontsize=8)
    savefig("trajectory_metrics")


def fig_boundary_cases():
    rows = read_csv(TAB / "boundary_cases.csv")
    methods = ["8step", "32step", "calibrated"]
    tasks = ["pubmedqa", "ceval_computer_network"]
    lookup = {(r["method"], r["task"]): fnum(r["accuracy"]) for r in rows}
    x = np.arange(len(tasks))
    w = 0.24
    plt.figure(figsize=(6.8, 3.7))
    colors = ["#9aa0a6", "#4c78a8", "#e45756"]
    for i, m in enumerate(methods):
        plt.bar(x + (i - 1) * w, [lookup.get((m, t), np.nan) for t in tasks], width=w, label=m, color=colors[i], edgecolor="#333333", linewidth=0.4)
    plt.xticks(x, [TASK_ZH[t] for t in tasks])
    plt.ylim(0.45, 0.72)
    plt.ylabel("准确率")
    plt.title("边界负例：预算控制不能替代知识与校准")
    plt.legend(frameon=False, fontsize=8)
    savefig("boundary_cases")


def fig_coverage():
    rows = read_csv(TAB / "coverage_summary.csv")
    tasks = ["arc_challenge", "hellaswag", "boolq", "gsm8k"]
    methods = ["llada8b_8step", "llada8b_32step", "llada8b_calibrated", "qwen25_7b"]
    names = ["LLaDA 8步", "LLaDA 32步", "校准控制器", "Qwen2.5-7B"]
    lookup = {(r["method"], r["task"]): fnum(r["accuracy"], np.nan) for r in rows}
    x = np.arange(len(tasks))
    w = 0.18
    plt.figure(figsize=(8.6, 4.0))
    colors = ["#9aa0a6", "#4c78a8", "#e45756", "#54a24b"]
    for i, m in enumerate(methods):
        vals = [lookup.get((m, t), np.nan) for t in tasks]
        plt.bar(x + (i - 1.5) * w, vals, width=w, label=names[i], color=colors[i], edgecolor="#333333", linewidth=0.35)
    plt.xticks(x, [TASK_ZH[t] for t in tasks])
    plt.ylabel("准确率")
    plt.ylim(0.25, 0.95)
    plt.title("扩展任务覆盖：预算敏感性与自回归模型对比")
    plt.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    savefig("coverage_comparison")


def fig_step_sweep():
    path = TAB / "step_sweep_9task_limit50_seed23.csv"
    rows = read_csv(path if path.exists() else TAB / "step_sweep.csv")
    tasks = ["mmlu_pro", "pubmedqa", "ceval_computer_network", "sciq", "winogrande", "commonsenseqa", "arc_challenge", "hellaswag", "boolq", "gsm8k"]
    plt.figure(figsize=(8.4, 4.5))
    for task in tasks:
        rs = sorted([r for r in rows if r["task"] == task], key=lambda r: fnum(r["step"]))
        if not rs:
            continue
        plt.plot([fnum(r["step"]) for r in rs], [fnum(r["accuracy"]) for r in rs], marker="o", linewidth=1.6, label=TASK_ZH.get(task, task))
    plt.xlabel("扩散步数")
    plt.ylabel("准确率")
    plt.title("扩散步数扫描：不同任务的预算敏感性")
    plt.legend(frameon=False, ncol=2, fontsize=8)
    savefig("step_sweep_by_task")


def write_markdown_tables():
    lora = read_csv(TAB / "lora_macro_summary.csv")
    keep = [
        "base_fixed32",
        "vanilla_lora_fixed32",
        "tardi_lora_balanced_r8",
        "tardi_lora_balanced_r8_highnoise",
        "tardi_lora_balanced_r16",
        "tasknara_r8_highnoise",
        "tasknara_residual_vanilla_highnoise",
    ]
    rows = []
    for m in keep:
        r = next(x for x in lora if x["method"] == m)
        rows.append({
            "方法": METHOD_ZH[m],
            "正确数": r["correct"],
            "样本数": r["n"],
            "宏平均准确率": f"{fnum(r['macro_accuracy']):.3f}",
            "相对基础模型": f"{fnum(r['macro_accuracy']) - fnum(next(x for x in lora if x['method'] == 'base_fixed32')['macro_accuracy']):+.3f}",
        })
    write_csv(TAB / "paper_lora_main.csv", rows, ["方法", "正确数", "样本数", "宏平均准确率", "相对基础模型"])

    task_rows = read_csv(TAB / "lora_task_summary.csv")
    tasks = ["mmlu_pro", "pubmedqa", "ceval_computer_network", "sciq", "winogrande", "commonsenseqa", "arc_challenge", "hellaswag", "boolq"]
    lookup = {(r["method"], r["task"]): fnum(r["accuracy"]) for r in task_rows}
    rows = []
    for t in tasks:
        base = lookup[("base_fixed32", t)]
        old = lookup[("vanilla_lora_fixed32", t)]
        tardi = lookup[("tardi_lora_balanced_r8_highnoise", t)]
        rows.append({
            "数据集": TASK_ZH[t],
            "LLaDA 基础模型": f"{base:.2f}",
            "旧版 LoRA": f"{old:.2f}",
            "TARDI-LoRA": f"{tardi:.2f}",
            "TARDI 增益": f"{tardi - base:+.2f}",
        })
    rows.append({
        "数据集": "宏平均",
        "LLaDA 基础模型": f"{sum(lookup[('base_fixed32', t)] for t in tasks)/len(tasks):.3f}",
        "旧版 LoRA": f"{sum(lookup[('vanilla_lora_fixed32', t)] for t in tasks)/len(tasks):.3f}",
        "TARDI-LoRA": f"{sum(lookup[('tardi_lora_balanced_r8_highnoise', t)] for t in tasks)/len(tasks):.3f}",
        "TARDI 增益": f"{sum(lookup[('tardi_lora_balanced_r8_highnoise', t)] - lookup[('base_fixed32', t)] for t in tasks)/len(tasks):+.3f}",
    })
    write_csv(TAB / "paper_lora_task_main.csv", rows, ["数据集", "LLaDA 基础模型", "旧版 LoRA", "TARDI-LoRA", "TARDI 增益"])

    qwen_9task = TAB / "qwen_lora_9task_limit50_seed23.csv"
    if qwen_9task.exists():
        qwen_rows = [r for r in read_csv(qwen_9task) if r["task"] != "macro"]
        overlap = [r["task"] for r in qwen_rows]
        qwen_lookup = {
            r["task"]: (fnum(r["qwen_base_accuracy"]), fnum(r["qwen_lora_accuracy"]))
            for r in qwen_rows
        }
    else:
        audit = read_csv(TAB / "old_lora_gain_audit.csv")
        audit_lookup = {(r["method"], r["task"]): fnum(r["accuracy"]) for r in audit}
        overlap = ["mmlu_pro", "pubmedqa", "ceval_computer_network", "sciq", "winogrande", "commonsenseqa"]
        qwen_lookup = {
            t: (audit_lookup[("qwen_base", t)], audit_lookup[("qwen_lora", t)])
            for t in overlap
        }
    rows = []
    for t in overlap:
        llada_base = lookup[("base_fixed32", t)]
        old_lora = lookup[("vanilla_lora_fixed32", t)]
        tardi = lookup[("tardi_lora_balanced_r8_highnoise", t)]
        qwen_base, qwen_lora = qwen_lookup[t]
        rows.append({
            "数据集": TASK_ZH[t],
            "LLaDA 基础模型": f"{llada_base:.2f}",
            "旧版 DDM LoRA": f"{old_lora:.2f}",
            "旧版 DDM LoRA 增益": f"{old_lora - llada_base:+.2f}",
            "TARDI-LoRA": f"{tardi:.2f}",
            "TARDI 增益": f"{tardi - llada_base:+.2f}",
            "Qwen 基础模型": f"{qwen_base:.2f}",
            "Qwen LoRA": f"{qwen_lora:.2f}",
            "Qwen LoRA 增益": f"{qwen_lora - qwen_base:+.2f}",
        })
    rows.append({
        "数据集": "宏平均",
        "LLaDA 基础模型": f"{sum(lookup[('base_fixed32', t)] for t in overlap)/len(overlap):.3f}",
        "旧版 DDM LoRA": f"{sum(lookup[('vanilla_lora_fixed32', t)] for t in overlap)/len(overlap):.3f}",
        "旧版 DDM LoRA 增益": f"{sum(lookup[('vanilla_lora_fixed32', t)] - lookup[('base_fixed32', t)] for t in overlap)/len(overlap):+.3f}",
        "TARDI-LoRA": f"{sum(lookup[('tardi_lora_balanced_r8_highnoise', t)] for t in overlap)/len(overlap):.3f}",
        "TARDI 增益": f"{sum(lookup[('tardi_lora_balanced_r8_highnoise', t)] - lookup[('base_fixed32', t)] for t in overlap)/len(overlap):+.3f}",
        "Qwen 基础模型": f"{sum(qwen_lookup[t][0] for t in overlap)/len(overlap):.3f}",
        "Qwen LoRA": f"{sum(qwen_lookup[t][1] for t in overlap)/len(overlap):.3f}",
        "Qwen LoRA 增益": f"{sum(qwen_lookup[t][1] - qwen_lookup[t][0] for t in overlap)/len(overlap):+.3f}",
    })
    write_csv(TAB / "paper_ar_ddm_lora_comparison.csv", rows, ["数据集", "LLaDA 基础模型", "旧版 DDM LoRA", "旧版 DDM LoRA 增益", "TARDI-LoRA", "TARDI 增益", "Qwen 基础模型", "Qwen LoRA", "Qwen LoRA 增益"])

    main = read_csv(TAB / "controller_main_1000.csv")
    rows = []
    for r in main:
        rows.append({
            "方法": r["method"],
            "任务": TASK_ZH.get(r["task"], r["task"]),
            "准确率": f"{fnum(r['accuracy']):.3f}",
            "正确数": r["correct"],
            "样本数": r["n"],
            "平均调用次数": f"{fnum(r['avg_forward_calls']):.2f}",
            "路由": r["route_rates"],
        })
    write_csv(TAB / "paper_controller_main.csv", rows, ["方法", "任务", "准确率", "正确数", "样本数", "平均调用次数", "路由"])


def main():
    setup_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    copy_core_tables()
    write_markdown_tables()
    fig_lora_macro()
    fig_lora_task_heatmap()
    fig_old_lora_gain()
    fig_ar_vs_tardi_lora_gain()
    fig_controller_accuracy_cost()
    fig_route_distribution()
    fig_threshold_sweep()
    fig_trajectory_metrics()
    fig_boundary_cases()
    fig_coverage()
    fig_step_sweep()
    print(f"wrote {FIG} and {TAB}")


if __name__ == "__main__":
    main()
