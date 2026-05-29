import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_domain_shift as domain
import eval_subset as base
from eval_llada_adaptive_router import apply_prompt, infer_label_space, parse_for_metric
from eval_llada_sampler_variants import MASK_ID, model_logits, token_budget, x0_and_conf


def choose_prompt_kind(sample):
    return "calibrated_decision" if sample.get("metric") == "decision" else "typed"


@torch.no_grad()
def generate_early_commit(model, tokenizer, sample, args, labels):
    text = base.chat_prompt(tokenizer, sample["prompt"])
    enc = tokenizer([text], add_special_tokens=False, padding=True, return_tensors="pt")
    prompt = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    x = torch.full((prompt.shape[0], prompt.shape[1] + args.gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((prompt.shape[0], args.gen_length), dtype=attention_mask.dtype, device=model.device),
        ],
        dim=-1,
    )
    prompt_index = x != MASK_ID
    assert args.gen_length % args.block_length == 0
    num_blocks = args.gen_length // args.block_length
    assert args.max_steps % num_blocks == 0
    steps_per_block = args.max_steps // num_blocks

    forward_calls = 0
    history = []
    last_valid = ""
    stable_count = 0
    stop_reason = "max_steps"
    committed_step = args.max_steps

    for nb in range(num_blocks):
        block_start = prompt.shape[1] + nb * args.block_length
        block_end = prompt.shape[1] + (nb + 1) * args.block_length
        block_mask = x[:, block_start:block_end] == MASK_ID
        budgets = token_budget(block_mask, steps_per_block, args.schedule)
        for i in range(steps_per_block):
            mask_index = x == MASK_ID
            logits = model_logits(model, x, attention_mask, prompt_index, args.cfg)
            forward_calls += 1
            x0, conf, _ = x0_and_conf(logits, x, mask_index, prompt.shape[1], block_end, args.temperature, args.remasking)
            transfer = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            for j in range(conf.shape[0]):
                k = int(budgets[j, i].item())
                if k <= 0:
                    continue
                _, idx = torch.topk(conf[j], k=k)
                transfer[j, idx] = True
            x[transfer] = x0[transfer]

            global_step = nb * steps_per_block + i + 1
            should_check = global_step == args.max_steps or (
                global_step >= args.min_steps and global_step % args.check_interval == 0
            )
            if not should_check:
                continue
            gen_ids = x[:, prompt.shape[1] :]
            output_now = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            pred_now = parse_for_metric(sample, output_now)
            valid = (not labels) or pred_now in labels
            filled = float((gen_ids != MASK_ID).float().mean().item())
            history.append({"step": global_step, "pred": pred_now, "valid": valid, "filled_ratio": filled})
            if valid and pred_now:
                stable_count = stable_count + 1 if pred_now == last_valid else 1
                last_valid = pred_now
                if stable_count >= args.patience:
                    stop_reason = "stable_label"
                    committed_step = global_step
                    output = output_now
                    return output, forward_calls, committed_step, stop_reason, history
            elif args.stop_on_invalid and global_step >= args.max_steps:
                stop_reason = "invalid_at_max"

    output = tokenizer.batch_decode(x[:, prompt.shape[1] :], skip_special_tokens=True)[0]
    return output, forward_calls, committed_step, stop_reason, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="winogrande,commonsenseqa")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--min-steps", type=int, default=8)
    ap.add_argument("--check-interval", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--schedule", choices=["uniform", "front_loaded", "back_loaded", "middle_heavy"], default="uniform")
    ap.add_argument("--stop-on-invalid", action="store_true")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model = base.load_llada(args.model)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    summary = {}
    for task in [t for t in args.tasks.split(",") if t]:
        raw_samples = domain.build_samples(task, args.limit, args.seed)
        correct = 0
        total_time = 0.0
        total_calls = 0
        stop_counts = {}
        for raw in raw_samples:
            labels = infer_label_space(raw)
            sample = apply_prompt(raw, choose_prompt_kind(raw))
            t0 = time.time()
            output, calls, committed_step, stop_reason, history = generate_early_commit(
                model, tokenizer, sample, args, labels
            )
            dt = time.time() - t0
            pred = parse_for_metric(sample, output)
            ok = pred == sample["gold"]
            correct += int(ok)
            total_time += dt
            total_calls += calls
            stop_counts[stop_reason] = stop_counts.get(stop_reason, 0) + 1
            row = {
                "task": task,
                "id": raw["id"],
                "gold": raw["gold"],
                "pred": pred,
                "correct": ok,
                "output": output,
                "metric": raw["metric"],
                "seconds": dt,
                "forward_calls": calls,
                "committed_step": committed_step,
                "stop_reason": stop_reason,
                "history": history,
                "meta": raw.get("meta", {}),
            }
            rows.append(row)
            print(
                json.dumps(
                    {k: row[k] for k in ["task", "id", "gold", "pred", "correct", "forward_calls", "committed_step", "stop_reason"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        n = len(raw_samples)
        summary[task] = {
            "accuracy": correct / n,
            "n": n,
            "seconds": total_time,
            "avg_forward_calls": total_calls / n,
            "stop_rates": {k: v / n for k, v in sorted(stop_counts.items())},
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    payload = {"method": "prophet_style_early_commit", "model": args.model, "args": vars(args), "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
