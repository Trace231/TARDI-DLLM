#!/usr/bin/env python3
"""Collect LLaDA-only counterfactual refinement actions.

For each sample, this script creates early LLaDA states at configurable
checkpoints, then executes several *actual* LLaDA-only actions from the same
state: accept, restart with a larger budget, and selective re-masking variants.

The output is a counterfactual action table for training/evaluating a utility
policy. It intentionally does not use AR models.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as domain
import eval_subset as base
from eval_llada_adaptive_router import apply_prompt, infer_label_space, parse_for_metric, probe_label_distribution, task_profile
from eval_llada_refinement_controller import (
    fill_masks,
    label_token_ids,
    make_state,
    online_risk_features,
    remask_answer_consistency,
    remask_low_confidence,
    remask_structured,
    risk_score,
    scout_stats,
    token_text_flags,
)
from eval_llada_sampler_variants import MASK_ID


def parse_ints(text):
    return sorted({int(x.strip()) for x in text.split(",") if x.strip()})


def stable_seed(seed, *parts):
    key = ":".join([str(seed), *map(str, parts)]).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def safe_float(value, default=0.0):
    try:
        out = float(value)
        return default if math.isnan(out) or math.isinf(out) else out
    except Exception:
        return default


def clone_tensor(x):
    return None if x is None else x.clone()


def clone_state(x, attention_mask, prompt_index, last_conf):
    return x.clone(), clone_tensor(attention_mask), prompt_index.clone(), last_conf.clone()


def random_remask(x, prompt_len, fraction, min_tokens, seed):
    gen_tokens = x[:, prompt_len:]
    valid = gen_tokens != MASK_ID
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return 0, {"policy": "random", "reason": "no_valid_tokens"}
    k = max(min_tokens, int(round(valid_count * fraction)))
    k = min(k, valid_count)
    idx_all = torch.nonzero(valid[0], as_tuple=False).flatten()
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed) % (2**31 - 1))
    perm = torch.randperm(idx_all.numel(), generator=gen, device=x.device)[:k]
    idx = idx_all[perm]
    gen_tokens[0, idx] = MASK_ID
    return k, {"policy": "random", "fraction": fraction, "selected": idx.detach().cpu().tolist(), "valid_count": valid_count}


def remask_label_only(x, tokenizer, prompt_len, last_conf, labels, args):
    """Re-mask only the fixed-label decision boundary.

    Low-confidence remasking can disturb useful generated context. For fixed
    label tasks, the decision lives in a tiny legal-label subspace, so this
    action targets label tokens and answer markers first, then falls back to the
    single least-confident generated token if no such token is visible.
    """
    gen_tokens = x[:, prompt_len:]
    valid = gen_tokens != MASK_ID
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return 0, {"policy": "label_only", "reason": "no_valid_tokens"}
    token_ids = gen_tokens[0].detach().cpu().tolist()
    label_ids = label_token_ids(tokenizer, labels)
    flags = token_text_flags(tokenizer, token_ids)
    selected = []
    for idx, token_id in enumerate(token_ids):
        if not bool(valid[0, idx].item()):
            continue
        is_label = int(token_id) in label_ids
        is_marker = flags[idx]["final_marker"] or flags[idx]["separator"]
        if is_label or is_marker:
            selected.append(idx)
    if selected:
        # Prefer the last decision-bearing positions; in final-label prompts the
        # answer usually appears near the end of the generated block.
        selected = selected[-args.label_only_max_tokens :]
    else:
        gen_conf = last_conf[:, prompt_len:].clone()
        gen_conf = torch.where(valid, gen_conf, torch.full_like(gen_conf, float("inf")))
        _, idx = torch.topk(-gen_conf[0], k=1)
        selected = idx.detach().cpu().tolist()
    if not selected:
        return 0, {"policy": "label_only", "reason": "empty_selection"}
    idx_t = torch.tensor(selected, dtype=torch.long, device=x.device)
    gen_tokens[0, idx_t] = MASK_ID
    return len(selected), {
        "policy": "label_only",
        "selected": selected,
        "label_token_hits": int(sum(1 for i in selected if int(token_ids[i]) in label_ids)),
        "marker_hits": int(sum(1 for i in selected if flags[i]["final_marker"] or flags[i]["separator"])),
        "valid_count": valid_count,
    }


def label_id_map(tokenizer, labels):
    ids = {}
    for label in labels:
        for text in (label, " " + label, "\n" + label):
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(toks) == 1:
                ids[label] = toks[0]
                break
    return ids


@torch.no_grad()
def project_label_from_output(model, tokenizer, sample, current_output, labels):
    """Project the scout state onto the legal final-label posterior.

    The action keeps the generated reasoning/context intact and spends one
    forward call to score only the legal labels at the final answer position.
    """
    ids = label_id_map(tokenizer, labels)
    if len(ids) < 2:
        return None, {"policy": "label_projection", "available": False, "reason": "label_ids_unavailable"}

    text = current_output or ""
    marker = "Final answer:"
    marker_pos = text.lower().rfind(marker.lower())
    if marker_pos >= 0:
        prefix = text[: marker_pos + len(marker)].rstrip() + " "
    else:
        prefix = text.rstrip()
        if prefix:
            prefix += "\n"
        prefix += marker + " "

    prompt_text = base.chat_prompt(tokenizer, sample["prompt"]) + prefix
    enc = tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt")
    input_ids = torch.cat([enc["input_ids"], torch.tensor([[MASK_ID]], dtype=torch.long)], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    logits = model(input_ids, attention_mask=attention_mask).logits[0, -1]
    ordered = list(ids.keys())
    label_logits = torch.stack([logits[ids[label]] for label in ordered])
    probs = F.softmax(label_logits.float(), dim=0)
    order = torch.argsort(probs, descending=True)
    pred = ordered[int(order[0])]
    top_prob = float(probs[order[0]].item())
    second_prob = float(probs[order[1]].item()) if len(order) > 1 else 0.0
    return pred, {
        "policy": "label_projection",
        "available": True,
        "top_prob": top_prob,
        "margin": top_prob - second_prob,
        "probs": {label: float(prob.item()) for label, prob in zip(ordered, probs)},
    }


def remask_fraction(state_features, args):
    score = safe_float(state_features.get("risk_score"))
    base = args.remask_min_fraction + (args.remask_max_fraction - args.remask_min_fraction) * max(0.0, score)
    return max(args.remask_min_fraction, min(args.remask_max_fraction, base))


def summarize_output(tokenizer, sample, x, prompt_len):
    output = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
    pred = parse_for_metric(sample, output)
    return output, pred, bool(pred == sample["gold"])


def feature_pack(profile, probe, pred, labels, stats, args, checkpoint, calls_so_far):
    feats = online_risk_features(profile, probe, pred, labels, stats, args, checkpoint)
    score = risk_score(feats, profile)
    feats = dict(feats)
    feats.update(
        {
            "risk_score": score,
            "checkpoint": float(checkpoint),
            "checkpoint_norm": checkpoint / 32.0,
            "calls_so_far": float(calls_so_far),
            "calls_so_far_norm": calls_so_far / 32.0,
            "n_labels": float(profile.get("n_labels", 0)),
            "is_binary": 1.0 if profile.get("n_labels", 0) == 2 else 0.0,
            "is_multichoice": 1.0 if profile.get("n_labels", 0) >= 3 else 0.0,
            "probe_top_prob": safe_float(probe.get("top_prob")),
            "probe_margin": safe_float(probe.get("margin")),
            "probe_available": 1.0 if probe.get("available") else 0.0,
            "pred_valid": 1.0 if (pred and ((not labels) or pred in labels)) else 0.0,
        }
    )
    return feats


@torch.no_grad()
def run_fresh(model, tokenizer, sample, steps, schedule, args, labels):
    x, attention_mask, prompt_len, prompt_index = make_state(tokenizer, sample, args, model.device)
    x, output, pred, calls, last_conf, history = fill_masks(
        model,
        tokenizer,
        sample,
        x,
        attention_mask,
        prompt_len,
        prompt_index,
        steps,
        schedule,
        args,
        labels,
        {steps},
    )
    return x, attention_mask, prompt_len, prompt_index, output, pred, calls, last_conf, history


@torch.no_grad()
def run_from_state(model, tokenizer, sample, state, extra_steps, schedule, args, labels):
    x, attention_mask, prompt_len, prompt_index, last_conf = state
    x, output, pred, calls, last_conf2, history = fill_masks(
        model,
        tokenizer,
        sample,
        x,
        attention_mask,
        prompt_len,
        prompt_index,
        max(1, int(extra_steps)),
        schedule,
        args,
        labels,
        {max(1, int(extra_steps))},
    )
    return x, output, pred, calls, last_conf2, history


def make_action_result(action, target, checkpoint, calls_so_far, extra_calls, remasked, output, pred, correct, plan):
    total_calls = calls_so_far + extra_calls
    return {
        "action": action,
        "target_budget": target,
        "extra_calls": extra_calls,
        "total_calls": total_calls,
        "remasked_tokens": remasked,
        "pred": pred,
        "correct": bool(correct),
        "output": output,
        "remask_plan": plan,
        "cost_from_checkpoint": extra_calls,
        "budget_delta": max(0, int(target) - int(checkpoint)),
    }


def _apply_remask_family(family, x2, tokenizer, prompt_len, conf2, labels, pred, probe, feats, args, seed):
    """Apply one remask family in-place to x2; return (remasked_count, plan) uniformly."""
    frac = remask_fraction(feats, args)
    if family == "lowconf":
        return remask_low_confidence(x2, prompt_len, conf2, frac, args.remask_min_tokens), {"policy": "lowconf", "fraction": frac}
    if family == "answer":
        return remask_answer_consistency(x2, tokenizer, prompt_len, conf2, labels, pred, probe, frac, args.remask_min_tokens, args)
    if family == "structured":
        return remask_structured(x2, tokenizer, prompt_len, conf2, labels, frac, args.remask_min_tokens, args)
    if family == "random":
        return random_remask(x2, prompt_len, frac, args.remask_min_tokens, seed)
    if family == "label_only":
        return remask_label_only(x2, tokenizer, prompt_len, conf2, labels, args)
    raise ValueError(family)


@torch.no_grad()
def expand_remask(model, tokenizer, sample, raw, task, args, labels, profile, probe, probe_calls,
                  x, attention_mask, prompt_len, prompt_index, last_conf, pred,
                  eff_ckpt, calls_so_far, feats, families, targets, depth):
    """Remask actions for a decoded state at budget eff_ckpt. With depth>1, each (non-skipped) remask
    carries 'next_state' = the POST-REMASK state record (accept + further remask actions), built
    recursively. This makes the remask axis a genuine multi-step decision, not a one-shot terminal."""
    out_actions = []
    for target in targets:
        if target <= eff_ckpt:
            continue
        for family in families:
            if family == "projection":
                continue
            x2, attention2, prompt_index2, conf2 = clone_state(x, attention_mask, prompt_index, last_conf)
            seed = stable_seed(args.seed, task, raw["id"], eff_ckpt, target, family, depth)  # depth -> no root/child seed collision
            remasked, plan = _apply_remask_family(family, x2, tokenizer, prompt_len, conf2, labels, pred, probe, feats, args, seed)
            if remasked <= 0:
                out_a, pred_a, ok_a = summarize_output(tokenizer, sample, x2, prompt_len)
                out_actions.append(make_action_result(
                    f"remask_{family}_to_{target}", target, eff_ckpt, calls_so_far, 0, 0,
                    out_a, pred_a, ok_a, {**plan, "skipped": True}))
                continue
            x3, out_a, pred_a, c_a, conf_after, hist_after = run_from_state(
                model, tokenizer, sample, (x2, attention2, prompt_len, prompt_index2, conf2),
                target - eff_ckpt, args.refine_schedule, args, labels)
            act = make_action_result(
                f"remask_{family}_to_{target}", target, eff_ckpt, calls_so_far, c_a, remasked,
                out_a, pred_a, pred_a == raw["gold"], plan)
            if depth > 1:
                stats2 = scout_stats(hist_after, args, conf_after, prompt_len, target)
                feats2 = feature_pack(profile, probe, pred_a, labels, stats2, args, target, calls_so_far + c_a)
                # ORIGIN markers: child features come from a PARTIAL-refine history (vs root's full decode),
                # so the learned tree-solver must be able to tell root vs child states apart. depth level:
                # 1 at first remask child, 2 at grandchild, ...
                feats2["is_child"] = 1.0
                feats2["remask_depth"] = float(args.remask_depth - depth + 1)
                child = {
                    "checkpoint": target, "state_pred": pred_a, "state_correct": bool(pred_a == raw["gold"]),
                    "state_output": out_a, "calls_so_far": calls_so_far + c_a, "features": feats2,
                    "from_remask": f"remask_{family}_to_{target}@{eff_ckpt}",
                    "actions": [make_action_result("accept_current", target, target, calls_so_far + c_a,
                                                   0, 0, out_a, pred_a, pred_a == raw["gold"], {"policy": "accept_current"})],
                }
                child["actions"].extend(expand_remask(
                    model, tokenizer, sample, raw, task, args, labels, profile, probe, probe_calls,
                    x3, attention2, prompt_len, prompt_index2, conf_after, pred_a,
                    target, calls_so_far + c_a, feats2, args.child_families, args.child_targets, depth - 1))
                act["next_state"] = child
            out_actions.append(act)
    return out_actions


@torch.no_grad()
def collect_one(model, tokenizer, raw, task, args):
    sample = apply_prompt(dict(raw), "typed")
    profile = task_profile(sample, tokenizer)
    labels = infer_label_space(sample)
    probe = probe_label_distribution(model, tokenizer, sample, labels)
    probe_calls = 1 if probe.get("available") else 0
    sample_out = {
        "task": task,
        "id": str(raw["id"]),
        "metric": raw["metric"],
        "gold": raw["gold"],
        "label_space": labels,
        "profile": profile,
        "probe": probe,
        "states": [],
    }

    for checkpoint in args.checkpoints:
        if checkpoint <= 0:
            continue
        x, attention_mask, prompt_len, prompt_index, output, pred, calls, last_conf, history = run_fresh(
            model, tokenizer, sample, checkpoint, args.base_schedule, args, labels
        )
        calls_so_far = probe_calls + calls
        stats = scout_stats(history, args, last_conf, prompt_len, checkpoint)
        feats = feature_pack(profile, probe, pred, labels, stats, args, checkpoint, calls_so_far)
        feats["is_child"] = 0.0  # root checkpoint state (vs post-remask child); see expand_remask origin markers
        feats["remask_depth"] = 0.0
        state_record = {
            "checkpoint": checkpoint,
            "state_pred": pred,
            "state_correct": bool(pred == raw["gold"]),
            "state_output": output,
            "calls_so_far": calls_so_far,
            "features": feats,
            "scout_stats": stats,
            "actions": [],
        }
        state_record["actions"].append(
            make_action_result(
                "accept_current",
                checkpoint,
                checkpoint,
                calls_so_far,
                0,
                0,
                output,
                pred,
                pred == raw["gold"],
                {"policy": "accept_current"},
            )
        )
        if "projection" in args.remask_families:
            projected_pred, projection_plan = project_label_from_output(model, tokenizer, sample, output, labels)
            if projected_pred is not None:
                state_record["actions"].append(
                    make_action_result(
                        "label_projection",
                        checkpoint,
                        checkpoint,
                        calls_so_far,
                        1,
                        0,
                        f"Final answer: {projected_pred}",
                        projected_pred,
                        projected_pred == raw["gold"],
                        projection_plan,
                    )
                )
        # restart actions (root-only, fresh decode to target)
        if args.include_restart:
            for target in args.targets:
                if target <= checkpoint:
                    continue
                _, _, _, _, out_r, pred_r, c_r, _, _ = run_fresh(model, tokenizer, sample, target, args.base_schedule, args, labels)
                state_record["actions"].append(
                    make_action_result(
                        f"restart_to_{target}", target, checkpoint, probe_calls, c_r, 0,
                        out_r, pred_r, pred_r == raw["gold"], {"policy": "restart", "target": target},
                    )
                )
        # remask actions (iterative-capable: depth>1 attaches post-remask 'next_state' recursively)
        state_record["actions"].extend(
            expand_remask(
                model, tokenizer, sample, raw, task, args, labels, profile, probe, probe_calls,
                x, attention_mask, prompt_len, prompt_index, last_conf, pred,
                checkpoint, calls_so_far, feats, args.remask_families, args.targets, args.remask_depth,
            )
        )
        sample_out["states"].append(state_record)
    return sample_out


def summarize(samples):
    by_task = {}
    for sample in samples:
        task = sample["task"]
        by_task.setdefault(task, {"n": 0, "accept_correct": 0, "action_rows": 0})
        by_task[task]["n"] += 1
        if sample["states"]:
            by_task[task]["accept_correct"] += int(sample["states"][0]["actions"][0]["correct"])
        by_task[task]["action_rows"] += sum(len(s["actions"]) for s in sample["states"])
    for vals in by_task.values():
        vals["first_checkpoint_accept_accuracy"] = vals["accept_correct"] / max(1, vals["n"])
    return by_task


def build_payload(args, samples_out, t0):
    return {
        "schema": "llada_counterfactual_actions_v1",
        "model": args.model,
        "args": vars(args),
        "summary": summarize(samples_out),
        "n": len(samples_out),
        "seconds": time.time() - t0,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "samples": samples_out,
    }


def write_payload(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default="winogrande,commonsenseqa,arc_challenge,sciq,hellaswag")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="4,8")
    ap.add_argument("--targets", default="8,16,32")
    ap.add_argument("--remask-families", default="lowconf,answer,structured,random")
    ap.add_argument("--include-restart", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--remask-depth", type=int, default=1,
                    help="1 = remask is terminal (current); >1 = ITERATIVE remask: after a remask, capture the "
                         "post-remask state + its own actions recursively (true multi-step decision on the remask axis)")
    ap.add_argument("--child-families", default="",
                    help="remask families allowed at depth>1 child states (default = same as --remask-families); "
                         "restrict (e.g. 'lowconf') to bound branching")
    ap.add_argument("--child-targets", default="",
                    help="remask targets allowed at depth>1 child states (default = same as --targets)")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--base-schedule", choices=["uniform", "front_loaded", "back_loaded", "middle_heavy"], default="uniform")
    ap.add_argument("--refine-schedule", choices=["uniform", "front_loaded", "back_loaded", "middle_heavy"], default="uniform")
    ap.add_argument("--margin-target", type=float, default=0.18)
    ap.add_argument("--fill-confidence-target", type=float, default=0.70)
    ap.add_argument("--remask-min-fraction", type=float, default=0.10)
    ap.add_argument("--remask-max-fraction", type=float, default=0.45)
    ap.add_argument("--remask-min-tokens", type=int, default=3)
    ap.add_argument("--label-remask-bonus", type=float, default=0.45)
    ap.add_argument("--label-context-bonus", type=float, default=0.20)
    ap.add_argument("--answer-marker-bonus", type=float, default=0.12)
    ap.add_argument("--label-remask-window", type=int, default=2)
    ap.add_argument("--answer-protect-confidence", type=float, default=0.58)
    ap.add_argument("--answer-protect-margin", type=float, default=0.04)
    ap.add_argument("--answer-protect-penalty", type=float, default=0.55)
    ap.add_argument("--answer-repair-confidence", type=float, default=0.62)
    ap.add_argument("--answer-repair-bonus", type=float, default=0.35)
    ap.add_argument("--answer-label-bonus", type=float, default=0.10)
    ap.add_argument("--label-only-max-tokens", type=int, default=2)
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args()
    args.checkpoints = parse_ints(args.checkpoints)
    args.targets = parse_ints(args.targets)
    args.remask_families = [x.strip() for x in args.remask_families.split(",") if x.strip()]
    args.child_families = [x.strip() for x in args.child_families.split(",") if x.strip()] or list(args.remask_families)
    args.child_targets = parse_ints(args.child_targets) if args.child_targets else list(args.targets)
    args.scout_steps = max(args.checkpoints or [8])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = base.load_llada(args.model, args.adapter)
    torch.cuda.reset_peak_memory_stats()
    samples_out = []
    t0 = time.time()
    for task in [t for t in args.tasks.split(",") if t]:
        for raw in domain.build_samples(task, args.limit, args.seed):
            row = collect_one(model, tokenizer, raw, task, args)
            samples_out.append(row)
            first_state = row["states"][0] if row["states"] else {}
            print(
                json.dumps(
                    {
                        "task": row["task"],
                        "id": row["id"],
                        "gold": row["gold"],
                        "state_pred": first_state.get("state_pred"),
                        "actions": len(first_state.get("actions", [])),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
	            )
            if args.flush_every > 0 and len(samples_out) % args.flush_every == 0:
                payload = build_payload(args, samples_out, t0)
                payload["partial"] = True
                write_payload(args.out, payload)
                write_payload(str(args.out) + ".partial", payload)
    payload = build_payload(args, samples_out, t0)
    payload["partial"] = False
    write_payload(args.out, payload)
    print(json.dumps({"out": args.out, "n": payload["n"], "summary": payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
