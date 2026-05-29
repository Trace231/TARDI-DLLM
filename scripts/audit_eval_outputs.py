import argparse
import csv
import json
from pathlib import Path


LABEL_SPACES = {
    "winogrande": set("AB"),
    "commonsenseqa": set("ABCDE"),
    "arc_challenge": set("ABCDE"),
    "hellaswag": set("ABCD"),
    "boolq": {"yes", "no"},
    "pubmedqa": {"yes", "no", "maybe"},
}


def audit_file(path):
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    out = []
    for task in sorted({r.get("task") for r in rows if r.get("task")}):
        task_rows = [r for r in rows if r.get("task") == task]
        labels = LABEL_SPACES.get(task, set())
        invalid = [r for r in task_rows if labels and r.get("pred") not in labels]
        blank = [r for r in task_rows if not r.get("pred")]
        correct = sum(1 for r in task_rows if r.get("correct"))
        calls = [r.get("forward_calls") for r in task_rows if isinstance(r.get("forward_calls"), (int, float))]
        examples = []
        for r in invalid[:3]:
            examples.append(
                {
                    "id": r.get("id"),
                    "gold": r.get("gold"),
                    "pred": r.get("pred"),
                    "output": str(r.get("output", "")).replace("\n", " ")[:120],
                }
            )
        out.append(
            {
                "file": path.name,
                "task": task,
                "n": len(task_rows),
                "accuracy": correct / max(1, len(task_rows)),
                "invalid": len(invalid),
                "invalid_rate": len(invalid) / max(1, len(task_rows)),
                "blank": len(blank),
                "blank_rate": len(blank) / max(1, len(task_rows)),
                "avg_forward_calls": sum(calls) / len(calls) if calls else "",
                "invalid_examples": json.dumps(examples, ensure_ascii=False),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--max-invalid-rate", type=float, default=0.02)
    args = ap.parse_args()

    rows = []
    for item in args.paths:
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.glob("*.json")):
                rows.extend(audit_file(child))
        else:
            rows.extend(audit_file(path))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys()) if rows else []
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))

    bad = [r for r in rows if r["invalid_rate"] > args.max_invalid_rate]
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
