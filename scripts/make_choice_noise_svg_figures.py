#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


METHOD_LABELS = {
    "llada_base_fixed32": "Base 32",
    "llada_label_lora_fixed32": "Label LoRA 32",
    "llada_choice_noise_lora_fixed32": "Choice-noise 32",
    "llada_vanilla_lora_fixed32": "Vanilla LoRA 32",
    "llada_choice_noise_lora_controller": "Choice-noise + Ctrl",
    "llada_vanilla_lora_controller": "Vanilla + Ctrl",
}

METHOD_ORDER = [
    "llada_base_fixed32",
    "llada_label_lora_fixed32",
    "llada_choice_noise_lora_fixed32",
    "llada_vanilla_lora_fixed32",
    "llada_choice_noise_lora_controller",
    "llada_vanilla_lora_controller",
]

COLORS = ["#5B6770", "#7E57C2", "#00A6A6", "#E76F51", "#2A9D8F", "#2A9D8F"]


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write(path, body, width, height):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; fill: #1f2933; }}
.title {{ font-size: 20px; font-weight: 700; }}
.axis {{ font-size: 12px; fill: #4b5563; }}
.label {{ font-size: 11px; fill: #374151; }}
.value {{ font-size: 12px; font-weight: 650; }}
.grid {{ stroke: #d9dee5; stroke-width: 1; }}
</style>
{body}
</svg>
"""
    Path(path).write_text(svg)


def macro_bar(root, out_dir):
    rows = {r["method"]: r for r in read_csv(root / "tables" / "choice_noise_macro_summary.csv")}
    methods = [m for m in METHOD_ORDER if m in rows]
    vals = [float(rows[m]["macro_accuracy"]) for m in methods]
    width, height = 940, 460
    left, right, top, bottom = 74, 28, 62, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.68, 0.76
    parts = [f'<text x="{left}" y="34" class="title">LLaDA LoRA objective and controller comparison</text>']
    for tick in [0.68, 0.70, 0.72, 0.74, 0.76]:
        y = top + (y_max - tick) / (y_max - y_min) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>')
    bar_w = plot_w / len(methods) * 0.62
    for i, (m, v) in enumerate(zip(methods, vals)):
        cx = left + (i + 0.5) * plot_w / len(methods)
        y = top + (y_max - v) / (y_max - y_min) * plot_h
        h = top + plot_h - y
        parts.append(f'<rect x="{cx-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{COLORS[i]}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y-7:.1f}" text-anchor="middle" class="value">{v:.3f}</text>')
        label = METHOD_LABELS[m]
        parts.append(f'<text transform="translate({cx-4:.1f},{height-35}) rotate(-28)" text-anchor="end" class="label">{esc(label)}</text>')
    parts.append(f'<text x="{left}" y="{height-12}" class="axis">Macro accuracy, limit=50 × 9 tasks, seed=23</text>')
    write(out_dir / "choice_noise_macro_accuracy.svg", "\n".join(parts), width, height)


def pareto(root, out_dir):
    rows = read_csv(root / "tables" / "choice_noise_macro_summary.csv")
    width, height = 820, 500
    left, right, top, bottom = 74, 230, 60, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = 23.5, 33.5
    y_min, y_max = 0.715, 0.75
    parts = [f'<text x="{left}" y="34" class="title">Accuracy-cost trade-off</text>']
    for tick in [24, 26, 28, 30, 32]:
        x = left + (tick - x_min) / (x_max - x_min) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" class="axis">{tick}</text>')
    for tick in [0.72, 0.73, 0.74, 0.75]:
        y = top + (y_max - tick) / (y_max - y_min) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>')
    for r in rows:
        method = r["method"]
        calls = float(r["avg_calls_macro"]) if r["avg_calls_macro"] else 32.0
        acc = float(r["macro_accuracy"])
        x = left + (calls - x_min) / (x_max - x_min) * plot_w
        y = top + (y_max - acc) / (y_max - y_min) * plot_h
        color = "#2A9D8F" if "controller" in method else "#E76F51" if "vanilla" in method else "#00A6A6" if "choice" in method else "#7E57C2" if "label" in method else "#5B6770"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')
        parts.append(f'<text x="{x+12:.1f}" y="{y+4:.1f}" class="label">{esc(METHOD_LABELS.get(method, method))}</text>')
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-18}" text-anchor="middle" class="axis">Average forward calls</text>')
    parts.append(f'<text transform="translate(18,{top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle" class="axis">Macro accuracy</text>')
    write(out_dir / "choice_noise_accuracy_cost_pareto.svg", "\n".join(parts), width, height)


def heatmap(root, out_dir):
    rows = read_csv(root / "tables" / "choice_noise_task_summary.csv")
    methods = ["llada_base_fixed32", "llada_vanilla_lora_fixed32", "llada_choice_noise_lora_fixed32", "llada_vanilla_lora_controller"]
    tasks = []
    lookup = {}
    for r in rows:
        if r["method"] in methods:
            if r["task"] not in tasks:
                tasks.append(r["task"])
            lookup[(r["method"], r["task"])] = float(r["accuracy"])
    cell_w, cell_h = 96, 46
    left, top = 180, 70
    width = left + cell_w * len(tasks) + 30
    height = top + cell_h * len(methods) + 92
    parts = [f'<text x="{left}" y="34" class="title">Task-level accuracy</text>']
    for j, task in enumerate(tasks):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text transform="translate({x:.1f},{top-10}) rotate(-25)" text-anchor="end" class="label">{esc(task)}</text>')
    for i, method in enumerate(methods):
        y = top + i * cell_h
        parts.append(f'<text x="{left-12}" y="{y+29}" text-anchor="end" class="label">{esc(METHOD_LABELS[method])}</text>')
        for j, task in enumerate(tasks):
            v = lookup.get((method, task), 0.0)
            t = max(0, min(1, (v - 0.35) / 0.60))
            blue = int(245 - 130 * t)
            green = int(248 - 40 * t)
            red = int(255 - 220 * t)
            color = f"rgb({red},{green},{blue})"
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{color}" stroke="white"/>')
            parts.append(f'<text x="{x+cell_w/2:.1f}" y="{y+29}" text-anchor="middle" class="value">{v:.2f}</text>')
    write(out_dir / "choice_noise_task_heatmap.svg", "\n".join(parts), width, height)


def ar_gain(solid_root, out_dir):
    path = solid_root / "tables" / "lora_control_v2.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    gains = {}
    labels = {}
    for r in rows:
        if r["method"] in {"ar_lora_control_gain", "ddm_lora_control_gain", "ddm_lora_original_gain"}:
            gains.setdefault(r["method"], {})[r["task"]] = float(r["accuracy"])
            labels[r["task"]] = r["task_label"]
    tasks = [t for t in labels if t in gains.get("ar_lora_control_gain", {})]
    width, height = 900, 440
    left, right, top, bottom = 72, 34, 54, 112
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = -0.08, 0.30
    specs = [
        ("ar_lora_control_gain", "AR", "#457B9D"),
        ("ddm_lora_control_gain", "Old DDM control", "#5B6770"),
        ("ddm_lora_original_gain", "Old DDM original", "#00A6A6"),
    ]
    parts = [f'<text x="{left}" y="32" class="title">Prior AR vs DDM LoRA gain audit</text>']
    zero_y = top + (y_max - 0) / (y_max - y_min) * plot_h
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left+plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="1"/>')
    for tick in [-0.05, 0.0, 0.1, 0.2, 0.3]:
        y = top + (y_max - tick) / (y_max - y_min) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>')
    group_w = plot_w / len(tasks)
    bar_w = group_w / 4
    for j, task in enumerate(tasks):
        cx = left + j * group_w + group_w / 2
        parts.append(f'<text transform="translate({cx:.1f},{height-32}) rotate(-22)" text-anchor="end" class="label">{esc(labels[task])}</text>')
        for k, (method, _, color) in enumerate(specs):
            v = gains.get(method, {}).get(task, 0.0)
            x = cx + (k - 1) * bar_w - bar_w / 2
            y = top + (y_max - max(v, 0)) / (y_max - y_min) * plot_h
            y0 = top + (y_max - min(v, 0)) / (y_max - y_min) * plot_h
            parts.append(f'<rect x="{x:.1f}" y="{min(y,y0):.1f}" width="{bar_w*.8:.1f}" height="{abs(y0-y):.1f}" fill="{color}"/>')
    lx = left + plot_w - 210
    for i, (_, label, color) in enumerate(specs):
        parts.append(f'<rect x="{lx}" y="{top+i*20}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx+18}" y="{top+11+i*20}" class="label">{esc(label)}</text>')
    write(out_dir / "prior_ar_ddm_lora_gain.svg", "\n".join(parts), width, height)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/domain_shift/task_aware/choice_noise_v1")
    ap.add_argument("--solid-root", default="results/domain_shift/task_aware/solid_v2")
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    macro_bar(root, out_dir)
    pareto(root, out_dir)
    heatmap(root, out_dir)
    ar_gain(Path(args.solid_root), out_dir)
    print(f"wrote SVG figures to {out_dir}")


if __name__ == "__main__":
    main()
