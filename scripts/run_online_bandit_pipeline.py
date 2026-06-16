#!/usr/bin/env python3
"""Run the online bandit evaluation bundle for one decision table.

This is the reproducibility entry point for the online router experiment. It
does not collect GPU samples; it consumes an `online_decision_table_v1` JSON and
produces leakage checks, selective calibration, several LinUCB configurations,
and a compact comparison CSV.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-table", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--alpha", type=float, default=1.25)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--lambda-ar-calls", default="0.05,0.10,0.20")
    ap.add_argument("--min-ar-checkpoints", default="0,4,8")
    ap.add_argument("--selective-epsilon", type=float, default=0.12)
    ap.add_argument("--seconds-per-llda-call", type=float, default=1.0)
    ap.add_argument("--seconds-per-ar-call", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    tables = out / "tables"
    figures = out / "figures"
    routers = out / "routers"
    for directory in [tables, figures, routers]:
        directory.mkdir(parents=True, exist_ok=True)

    base = Path(__file__).resolve().parent
    run([sys.executable, base / "test_online_bandit_artifacts.py", "--decision-table", args.decision_table])
    run(
        [
            sys.executable,
            base / "calibrate_selective_acceptance.py",
            "--decision-table",
            args.decision_table,
            "--epsilon",
            args.selective_epsilon,
            "--max-checkpoint",
            8,
            "--out-summary",
            tables / "selective_acceptance_summary.csv",
            "--out-rows",
            tables / "selective_acceptance_rows.csv",
            "--seed",
            args.seed,
        ]
    )

    all_rows = []
    for min_ckpt in [int(x) for x in args.min_ar_checkpoints.split(",") if x.strip()]:
        for lam in [float(x) for x in args.lambda_ar_calls.split(",") if x.strip()]:
            tag = f"m{min_ckpt}_c{int(round(lam * 100)):03d}"
            summary_path = tables / f"online_bandit_{tag}_summary.csv"
            task_path = tables / f"online_bandit_{tag}_by_task.csv"
            run(
                [
                    sys.executable,
                    base / "eval_online_bandit_router.py",
                    "--decision-table",
                    args.decision_table,
                    "--out-summary",
                    summary_path,
                    "--out-task",
                    task_path,
                    "--out-rows",
                    tables / f"online_bandit_{tag}_rows.json",
                    "--out-router",
                    routers / f"online_bandit_{tag}_router.json",
                    "--out-figure",
                    figures / f"online_bandit_{tag}_pareto.png",
                    "--seed",
                    args.seed,
                    "--alpha",
                    args.alpha,
                    "--epsilon",
                    args.epsilon,
                    "--lambda-ar-call",
                    lam,
                    "--min-ar-checkpoint",
                    min_ckpt,
                    "--selective-epsilon",
                    args.selective_epsilon,
                    "--seconds-per-llda-call",
                    args.seconds_per_llda_call,
                    "--seconds-per-ar-call",
                    args.seconds_per_ar_call,
                ]
            )
            for row in read_csv(summary_path):
                row["config"] = tag
                row["min_ar_checkpoint"] = min_ckpt
                row["lambda_ar_call"] = lam
                all_rows.append(row)

    write_csv(tables / "online_bandit_config_sweep.csv", all_rows)
    payload = {
        "decision_table": args.decision_table,
        "out_dir": str(out),
        "n_configs": len(set(r["config"] for r in all_rows)),
        "summary_csv": str(tables / "online_bandit_config_sweep.csv"),
    }
    (tables / "pipeline_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
