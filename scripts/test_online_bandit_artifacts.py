#!/usr/bin/env python3
"""Lightweight correctness checks for online-bandit artifacts."""

import argparse
import json
from pathlib import Path


BANNED_FEATURE_KEYS = {
    "gold",
    "correct",
    "route",
    "final_budget",
    "forward_calls",
    "full_forward_calls",
    "oracle",
}


def check_features(payload):
    offenders = []
    for sample in payload["samples"]:
        for state in sample["states"]:
            feats = state.get("router_features") or {}
            for key in feats:
                if key in BANNED_FEATURE_KEYS:
                    offenders.append((sample["task"], sample["id"], state["checkpoint"], key))
    if offenders:
        raise AssertionError(f"feature leakage: {offenders[:10]}")


def check_action_validity(payload):
    for sample in payload["samples"]:
        for state in sample["states"]:
            if not state["llda_valid"]:
                assert state["llda_pred"] == "", (sample["task"], sample["id"], state["checkpoint"])
            if state["checkpoint"] >= 32:
                assert not state["has_next_state"], (sample["task"], sample["id"], state["checkpoint"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-table", required=True)
    args = ap.parse_args()
    payload = json.loads(Path(args.decision_table).read_text())
    assert payload["schema"] == "online_decision_table_v1"
    assert payload["samples"], "empty decision table"
    check_features(payload)
    check_action_validity(payload)
    print(json.dumps({"ok": True, "n": len(payload["samples"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
