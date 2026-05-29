#!/usr/bin/env python3
"""Draw a clean Chinese publication-style mechanism figure for the LLaDA report."""

from __future__ import annotations

import json
import re
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Matplotlib is required. On this machine, run:\n"
        "/opt/miniconda3/bin/python3 "
        "'/Users/thomaswang/Documents/New project/scripts/plot_llada_chinese_flow_figure.py'"
    ) from exc

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path("/Users/thomaswang/Documents/New project")
REPORT_PATH = PROJECT_ROOT / "LLaDA_Complete_Updated_Result_Report.md"
FIGURE_DIR = PROJECT_ROOT / "figures"
OUTPUT_BASENAME = "llada_calibrated_controller_cn"
LOCAL_FONT_DIR = PROJECT_ROOT / "fonts"

YAHEI_FONT_FILES = [
    LOCAL_FONT_DIR / "msyh.ttc",
    LOCAL_FONT_DIR / "msyh.ttf",
    LOCAL_FONT_DIR / "Microsoft YaHei.ttf",
    LOCAL_FONT_DIR / "Microsoft YaHei.ttc",
    Path("/Library/Fonts/Microsoft YaHei.ttf"),
    Path("/Library/Fonts/Microsoft YaHei.ttc"),
    Path("/System/Library/Fonts/Microsoft YaHei.ttf"),
    Path("/System/Library/Fonts/Microsoft YaHei.ttc"),
    Path.home() / "Library/Fonts/Microsoft YaHei.ttf",
    Path.home() / "Library/Fonts/Microsoft YaHei.ttc",
]

TYPE = {
    "title": 21.0,
    "subtitle": 11.2,
    "equation": 10.2,
    "section": 12.2,
    "section_note": 8.8,
    "node_title": 10.2,
    "node_body": 8.4,
    "route_title": 8.4,
    "route_body": 7.8,
    "panel_title": 10.8,
    "panel_note": 8.3,
    "task": 9.3,
    "small": 7.5,
    "tiny": 6.7,
}

PALETTE = {
    "blue": "#263A4D",
    "blue_2": "#567A9A",
    "blue_pale": "#F1F5F8",
    "green": "#7D956E",
    "green_pale": "#F3F6EF",
    "gold": "#B08A45",
    "gold_pale": "#F8F3E8",
    "red": "#A6665F",
    "red_pale": "#F7EFEE",
    "ink": "#252729",
    "gray_1": "#FBFBFA",
    "gray_2": "#E4E7EA",
    "gray_3": "#B7BEC6",
    "gray_4": "#68717B",
    "white": "#FFFFFF",
}

FALLBACK_RESULTS = {
    "WinoGrande": {
        "acc": 0.756,
        "calls": 17.56,
        "routes": {"8": 0.648, "8->32": 0.014, "32": 0.338},
    },
    "CommonsenseQA": {
        "acc": 0.817,
        "calls": 9.064,
        "routes": {"8": 0.998, "8->32": 0.002, "32": 0.0},
    },
}

FALLBACK_MCNEMAR = {"WinoGrande": 1.0, "CommonsenseQA": 0.7905}


def choose_chinese_font() -> str:
    for font_path in YAHEI_FONT_FILES:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            return fm.FontProperties(fname=str(font_path)).get_name()

    installed = {font.name for font in fm.fontManager.ttflist}
    for name in ["Microsoft YaHei", "Microsoft YaHei UI", "微软雅黑"]:
        if name in installed:
            return name

    fallback_order = ["Arial Unicode MS", "Hiragino Sans GB", "STHeiti", "Heiti TC", "PingFang HK"]
    for name in fallback_order:
        if name in installed:
            print(
                "WARNING: Microsoft YaHei / 微软雅黑 is not installed; "
                f"using fallback font '{name}' for this preview."
            )
            return name

    return "DejaVu Sans"


def setup_style() -> str:
    font_name = choose_chinese_font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TYPE["node_body"],
            "axes.linewidth": 2.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": PALETTE["ink"],
        }
    )
    return font_name


def read_report() -> str:
    if REPORT_PATH.exists():
        return REPORT_PATH.read_text(encoding="utf-8")
    return ""


def parse_results(report_text: str) -> dict[str, dict[str, float | dict[str, float]]]:
    results = {task: values.copy() for task, values in FALLBACK_RESULTS.items()}
    match = re.search(r"## 5\. Main 1000 Results(?P<body>.*?)(?:\n## |\Z)", report_text, re.S)
    if not match:
        return results

    for line in match.group("body").splitlines():
        if not line.startswith("| calibrated |"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        _, task, acc, _ci, avg_calls, _seconds, route_text = cells[:7]
        try:
            routes = json.loads(route_text)
            results[task] = {
                "acc": float(acc),
                "calls": float(avg_calls),
                "routes": {str(key): float(value) for key, value in routes.items()},
            }
        except (ValueError, json.JSONDecodeError):
            continue
    return results


def parse_mcnemar(report_text: str) -> dict[str, float]:
    p_values = FALLBACK_MCNEMAR.copy()
    match = re.search(r"## Paired McNemar(?P<body>.*?)(?:\n## |\Z)", report_text, re.S)
    if not match:
        return p_values

    for line in match.group("body").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 8 or cells[1] != "32step" or cells[2] != "calibrated":
            continue
        try:
            p_values[cells[0]] = float(cells[-1])
        except ValueError:
            continue
    return p_values


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    *,
    fc: str = PALETTE["white"],
    ec: str = PALETTE["gray_3"],
    lw: float = 1.2,
    color: str = PALETTE["ink"],
    fontsize: float = 10.0,
    weight: str = "normal",
    radius: float = 0.014,
    z: float = 2,
    alpha: float = 1.0,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            linespacing=1.18,
            zorder=z + 0.2,
        )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = PALETTE["blue"],
    lw: float = 2.0,
    scale: float = 14,
    rad: float = 0.0,
    alpha: float = 1.0,
    z: float = 4,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def header(ax):
    ax.text(
        0.045,
        0.940,
        "校准式前向感知风险控制器",
        ha="left",
        va="center",
        fontsize=TYPE["title"],
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.047,
        0.895,
        "用低预算 probe 观测 LLaDA 反向扩散轨迹，并按风险选择 cheap / fallback / full reverse budget",
        ha="left",
        va="center",
        fontsize=TYPE["subtitle"],
        color=PALETTE["gray_4"],
    )
    box(
        ax,
        0.750,
        0.900,
        0.205,
        0.062,
        r"$\min_{\pi}\ \mathbb{E}[\mathrm{err}+\lambda C(\pi,x)]$",
        fc=PALETTE["blue"],
        ec=PALETTE["blue"],
        color=PALETTE["white"],
        fontsize=TYPE["equation"],
        weight="bold",
        radius=0.016,
    )


def node(ax, x: float, y: float, w: float, h: float, title: str, body: str, *, fc: str, ec: str, color: str):
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.7, radius=0.018)
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=TYPE["node_title"], fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=TYPE["node_body"], color=color, linespacing=1.12)


def draw_mechanism(ax):
    ax.text(0.055, 0.805, "机制主链", ha="left", va="center", fontsize=TYPE["section"], fontweight="bold")
    ax.text(0.055, 0.775, "probe 提供可观测信号，scout 层判断风险，再由校准路由分配预算", ha="left", va="center", fontsize=TYPE["section_note"], color=PALETTE["gray_4"])

    node(
        ax,
        0.055,
        0.610,
        0.115,
        0.115,
        "输入样本",
        "prompt + [MASK]\n反向去噪开始",
        fc=PALETTE["white"],
        ec=PALETTE["gray_3"],
        color=PALETTE["ink"],
    )
    node(
        ax,
        0.205,
        0.610,
        0.120,
        0.115,
        "8-step probe",
        "cheap trajectory\n低预算探测",
        fc=PALETTE["blue"],
        ec=PALETTE["blue"],
        color=PALETTE["white"],
    )
    node(
        ax,
        0.360,
        0.610,
        0.125,
        0.115,
        "不确定性信号",
        "标签后验 / 置信度\nlabel disagreement",
        fc=PALETTE["white"],
        ec=PALETTE["blue_2"],
        color=PALETTE["blue"],
    )
    node(
        ax,
        0.670,
        0.595,
        0.130,
        0.145,
        "Calibrated route",
        "threshold τ\nrisk-cost objective\n任务级策略",
        fc=PALETTE["blue"],
        ec=PALETTE["blue"],
        color=PALETTE["white"],
    )

    box(ax, 0.530, 0.570, 0.105, 0.165, fc=PALETTE["white"], ec=PALETTE["gray_3"], lw=1.0, radius=0.014)
    ax.text(0.5825, 0.710, "Scout decision", ha="center", va="center", fontsize=TYPE["node_title"], fontweight="bold", color=PALETTE["ink"])
    scout_lines = [
        ("低置信度", PALETTE["red"]),
        ("二元近邻混淆", PALETTE["gold"]),
        ("多选分歧策略", PALETTE["green"]),
    ]
    for yy, (label, c) in zip([0.675, 0.645, 0.615], scout_lines):
        ax.plot([0.548, 0.562], [yy, yy], color=c, lw=2.2, solid_capstyle="round", zorder=4)
        ax.text(0.568, yy, label, ha="left", va="center", fontsize=TYPE["route_body"], color=PALETTE["ink"])

    arrow(ax, (0.173, 0.668), (0.202, 0.668), lw=2.3, scale=15)
    arrow(ax, (0.328, 0.668), (0.357, 0.668), lw=2.3, scale=15)
    arrow(ax, (0.488, 0.668), (0.527, 0.668), lw=2.3, scale=15)
    arrow(ax, (0.638, 0.668), (0.667, 0.668), lw=2.3, scale=15)

    routes = [
        (0.855, 0.720, PALETTE["white"], PALETTE["green"], "低风险", "8步接受"),
        (0.855, 0.625, PALETTE["white"], PALETTE["gold"], "中风险", "8->32回退"),
        (0.855, 0.530, PALETTE["white"], PALETTE["red"], "高风险", "32步全预算"),
    ]
    for y0, line_color, rad in [(0.755, PALETTE["green"], 0.16), (0.660, PALETTE["gold"], 0.00), (0.565, PALETTE["red"], -0.16)]:
        arrow(ax, (0.803, 0.668), (0.852, y0), color=line_color, lw=2.1, scale=14, rad=rad)
    for x, y, fc, ec, title, body in routes:
        box(ax, x, y, 0.095, 0.070, fc=fc, ec=PALETTE["gray_3"], lw=1.0, radius=0.012)
        ax.plot([x + 0.010, x + 0.010], [y + 0.014, y + 0.056], color=ec, lw=3.0, solid_capstyle="round", zorder=4)
        ax.text(x + 0.022, y + 0.045, title, ha="left", va="center", fontsize=TYPE["route_title"], fontweight="bold", color=ec)
        ax.text(x + 0.022, y + 0.023, body, ha="left", va="center", fontsize=TYPE["route_body"], color=PALETTE["ink"])

    box(
        ax,
        0.852,
        0.405,
        0.100,
        0.064,
        r"输出 $\hat{y}$",
        fc=PALETTE["blue"],
        ec=PALETTE["blue"],
        color=PALETTE["white"],
        fontsize=TYPE["node_title"],
        weight="bold",
        radius=0.014,
    )
    for start, line_color, rad in [((0.950, 0.755), PALETTE["green"], 0.33), ((0.950, 0.660), PALETTE["gold"], 0.10), ((0.950, 0.565), PALETTE["red"], -0.14)]:
        arrow(ax, start, (0.902, 0.472), color=line_color, lw=1.25, scale=9, rad=rad, alpha=0.60, z=1)


def trajectory_row(ax, y: float, task: str, first_step: float, flips: float, color: str):
    x0, x1 = 0.125, 0.445
    ax.text(0.075, y + 0.030, task, ha="left", va="center", fontsize=TYPE["task"], fontweight="bold", color=color)
    ax.text(0.075, y - 0.005, f"首次稳定步 {first_step:.2f}  |  flip {flips:.2f}", ha="left", va="center", fontsize=TYPE["small"], color=PALETTE["gray_4"])
    ax.plot([x0, x1], [y, y], color=PALETTE["gray_2"], lw=7.0, solid_capstyle="round", zorder=1)
    ax.plot([x0, x0 + (x1 - x0) * min(first_step / 32, 1)], [y, y], color=PALETTE["blue_pale"], lw=7.0, solid_capstyle="round", zorder=2)
    ax.plot([x0 + (x1 - x0) * min(first_step / 32, 1), x1], [y, y], color=color, lw=7.0, solid_capstyle="round", zorder=3, alpha=0.82)
    for step, label in [(8, "8"), (32, "32")]:
        xx = x0 + (x1 - x0) * step / 32
        ax.plot([xx, xx], [y - 0.027, y + 0.027], color=PALETTE["gray_3"], lw=1.2, zorder=4)
        ax.text(xx, y - 0.047, label, ha="center", va="top", fontsize=TYPE["small"], color=PALETTE["gray_4"])


def route_bar(ax, x: float, y: float, w: float, routes: dict[str, float]):
    h = 0.026
    cursor = x
    for key, color in [("8", PALETTE["green"]), ("8->32", PALETTE["gold"]), ("32", PALETTE["red"])]:
        frac = routes.get(key, 0.0)
        seg = w * frac
        if seg > 0.002:
            ax.add_patch(
                FancyBboxPatch(
                    (cursor, y),
                    seg,
                    h,
                    boxstyle="round,pad=0.0,rounding_size=0.006",
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0,
                    zorder=3,
                )
            )
        cursor += seg
    box(ax, x, y, w, h, fc="none", ec=PALETTE["gray_3"], lw=0.8, radius=0.006, z=4)


def draw_bottom_panels(ax, results: dict[str, dict[str, float | dict[str, float]]], mcnemar: dict[str, float]):
    box(ax, 0.055, 0.120, 0.435, 0.255, fc=PALETTE["white"], ec=PALETTE["gray_2"], lw=1.0, radius=0.018, z=0)
    ax.text(0.075, 0.335, "轨迹统计：预算需求不是常数", ha="left", va="center", fontsize=TYPE["panel_title"], fontweight="bold")
    ax.text(0.075, 0.305, "早期 label posterior 的稳定时间决定 cheap path 是否安全", ha="left", va="center", fontsize=TYPE["panel_note"], color=PALETTE["gray_4"])
    trajectory_row(ax, 0.245, "CommonsenseQA", 7.44, 0.72, PALETTE["green"])
    trajectory_row(ax, 0.165, "WinoGrande", 15.89, 0.35, PALETTE["blue_2"])

    box(ax, 0.535, 0.120, 0.410, 0.255, fc=PALETTE["white"], ec=PALETTE["gray_2"], lw=1.0, radius=0.018, z=0)
    ax.text(0.555, 0.335, "主实验锚点", ha="left", va="center", fontsize=TYPE["panel_title"], fontweight="bold")
    ax.text(0.555, 0.305, "相对 32-step full budget，精度边界基本守住，同时显著减少调用", ha="left", va="center", fontsize=TYPE["panel_note"], color=PALETTE["gray_4"])

    cards = [
        (0.555, "WinoGrande", f"Acc {results['WinoGrande']['acc']:.3f}", f"Calls {results['WinoGrande']['calls']:.2f}/32"),
        (0.683, "CommonsenseQA", f"Acc {results['CommonsenseQA']['acc']:.3f}", f"Calls {results['CommonsenseQA']['calls']:.2f}/32"),
        (0.811, "McNemar", f"p={mcnemar['WinoGrande']:.3f}", f"p={mcnemar['CommonsenseQA']:.3f}"),
    ]
    for x, title, line1, line2 in cards:
        box(ax, x, 0.203, 0.110, 0.072, fc=PALETTE["white"], ec=PALETTE["gray_3"], lw=1.0, radius=0.012)
        ax.text(x + 0.012, 0.257, title, ha="left", va="center", fontsize=TYPE["route_title"], fontweight="bold", color=PALETTE["blue"])
        ax.text(x + 0.012, 0.232, line1, ha="left", va="center", fontsize=TYPE["small"], color=PALETTE["ink"])
        ax.text(x + 0.012, 0.211, line2, ha="left", va="center", fontsize=TYPE["small"], color=PALETTE["gray_4"])

    ax.text(0.555, 0.175, "实际路由分布", ha="left", va="center", fontsize=TYPE["section_note"], fontweight="bold", color=PALETTE["ink"])
    for y, task in [(0.150, "WinoGrande"), (0.122, "CommonsenseQA")]:
        ax.text(0.660, y + 0.012, task, ha="right", va="center", fontsize=TYPE["small"], color=PALETTE["gray_4"])
        route_bar(ax, 0.668, y, 0.200, results[task]["routes"])  # type: ignore[index]
    ax.text(0.875, 0.136, "颜色同上方\n预算通道", ha="left", va="center", fontsize=TYPE["tiny"], color=PALETTE["gray_4"], linespacing=1.15)


def build_figure(results: dict[str, dict[str, float | dict[str, float]]], mcnemar: dict[str, float]):
    font_name = setup_style()
    fig, ax = plt.subplots(figsize=(17.2, 8.4), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    header(ax)
    draw_mechanism(ax)
    draw_bottom_panels(ax, results, mcnemar)
    ax.text(0.955, 0.055, f"Font: {font_name}", ha="right", va="center", fontsize=TYPE["tiny"], color=PALETTE["gray_3"])
    return fig


def save_outputs(fig) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ["pdf", "svg", "png"]:
        out_path = FIGURE_DIR / f"{OUTPUT_BASENAME}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.06, "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(out_path, **kwargs)
        saved.append(out_path)
    return saved


def main() -> None:
    report_text = read_report()
    results = parse_results(report_text)
    mcnemar = parse_mcnemar(report_text)
    fig = build_figure(results, mcnemar)
    saved = save_outputs(fig)
    plt.close(fig)
    for task, values in results.items():
        print(f"{task}: acc={values['acc']:.3f}, avg_calls={values['calls']:.3f}, routes={values['routes']}")
    print(
        "McNemar vs 32step: "
        f"WinoGrande p={mcnemar['WinoGrande']:.4f}, "
        f"CommonsenseQA p={mcnemar['CommonsenseQA']:.4f}"
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
