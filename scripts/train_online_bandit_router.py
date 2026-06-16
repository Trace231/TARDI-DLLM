#!/usr/bin/env python3
"""Train/init an online LinUCB router artifact from a decision table.

This is a calibration-stage trainer. Final stream evaluation is handled by
`eval_online_bandit_router.py`, which can continue updating the same type of
router online.
"""

import argparse
import json
import random
from pathlib import Path

from eval_online_bandit_router import LinUCBRouter, feature_schema, run_episode, split_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-table", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--calibration-fraction", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--lambda-llda-call", type=float, default=1 / 32)
    ap.add_argument("--lambda-ar-call", type=float, default=0.20)
    ap.add_argument("--min-ar-checkpoint", type=int, default=8)
    ap.add_argument("--max-checkpoint", type=int, default=32)
    ap.add_argument("--policy", choices=["linucb", "epsilon_greedy_sgd"], default="linucb")
    args = ap.parse_args()
    if args.policy != "linucb":
        raise SystemExit("epsilon_greedy_sgd is reserved for a later variant; use --policy linucb")

    payload = json.loads(Path(args.decision_table).read_text())
    samples = payload["samples"]
    names = feature_schema(samples)
    calibration = [s for s in samples if split_name(s, args.seed, args.calibration_fraction) == "calibration"]
    router = LinUCBRouter(len(names), alpha=args.alpha)
    rng = random.Random(args.seed)
    for sample in calibration:
        run_episode(sample, router, names, rng, args, learn=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_schema.json").write_text(json.dumps({"feature_schema": names}, ensure_ascii=False, indent=2))
    (out / "router_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    (out / "linucb_state.json").write_text(
        json.dumps(
            {
                "router": router.state_dict(),
                "n_calibration": len(calibration),
                "decision_table": args.decision_table,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"out_dir": str(out), "n_calibration": len(calibration), "n_features": len(names)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
