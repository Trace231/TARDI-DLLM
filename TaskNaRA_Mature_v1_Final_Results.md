# Mature TaskNaRA v1 Final Results

## 1. Why The First TaskNaRA Was Not Enough

The first TaskNaRA implementation was real, but still too shallow to be the main method:

```text
C(task, lambda) = I + c_scale * MLP([Fourier(lambda); Emb(task)])
```

It concatenated the task embedding into the same NaRA mapper and the formal run used vanilla denoising. That makes it task-aware, but not mature enough: task identity directly perturbs the shared dynamic core, and there is no mechanism to prevent small-task overfitting or negative transfer.

## 2. Mature Adapter Design

The revised adapter separates shared diffusion dynamics from task-specific residual correction:

```text
C(task, lambda)
= I
  + c_scale * F(lambda)
  + sigmoid(g_task) * rho * G(lambda, Emb(task))
```

where:

- `F(lambda)` is the shared noise-aware core.
- `G(lambda, Emb(task))` is a task residual mapper.
- `sigmoid(g_task)` is a learned per-task gate initialized small.
- `rho = 0.04` constrains the residual magnitude.
- task dropout randomly removes task embedding during training with probability `0.20`.

This is more mature than simple concatenation because the model must first learn a shared diffusion adapter and can only apply bounded task-specific deviations.

Implemented in:

```text
scripts/nara_adapter.py
scripts/train_llada_choice_noise_lora.py
scripts/run_tasknara_mature_v1.sh
```

## 3. Experimental Setup

Evaluation protocol:

```text
model = /data/hf/models/GSAI-ML/LLaDA-8B-Instruct
tasks = 9 fixed-label tasks
limit = 50 per task
seed = 23
decoding = fixed 32-step final-label evaluation
train = 9-task balanced eval-disjoint set
```

Mature variants:

| Method | Objective | Adapter |
|---|---|---|
| residual vanilla high-noise | denoising only, high noise | shared noise core + gated task residual |
| residual label high-noise | label posterior + denoising | shared noise core + gated task residual |

## 4. Macro Results

| Method | Macro Acc. | Correct / N | Comment |
|---|---:|---:|---|
| TARDI-LoRA r8 high-noise | 0.776 | 349 / 450 | best overall |
| TARDI-LoRA r16 | 0.776 | 349 / 450 | best overall |
| TaskNaRA concat r8 high-noise | 0.773 | 348 / 450 | first task-aware adapter |
| Mature TaskNaRA residual vanilla | 0.769 | 346 / 450 | more controlled, but not best |
| Mature TaskNaRA residual label | 0.762 | 343 / 450 | label loss adds negative transfer |
| Old vanilla LoRA | 0.744 | 335 / 450 | old baseline |
| Base LLaDA | 0.722 | 325 / 450 | base |

## 5. Task-Level Results

| Task | TARDI r8 high-noise | TaskNaRA concat | Mature residual vanilla | Mature residual label |
|---|---:|---:|---:|---:|
| MMLU-Pro | 0.44 | 0.42 | 0.44 | 0.42 |
| PubMedQA | 0.74 | 0.74 | 0.76 | 0.76 |
| C-Eval CN | 0.64 | 0.68 | 0.60 | 0.60 |
| SciQ | 0.92 | 0.92 | 0.92 | 0.92 |
| WinoGrande | 0.76 | 0.76 | 0.80 | 0.80 |
| CommonsenseQA | 0.92 | 0.90 | 0.92 | 0.88 |
| ARC-Challenge | 0.86 | 0.84 | 0.82 | 0.82 |
| HellaSwag | 0.78 | 0.78 | 0.76 | 0.78 |
| BoolQ | 0.92 | 0.92 | 0.90 | 0.88 |

## 6. What This Means

The mature version is not toy: it has a bounded residual architecture, task dropout, trained task gates, high-noise training, full save/load support, and 9-task evaluation. But it still does not beat the simpler TARDI-LoRA protocol.

The useful scientific conclusion is:

1. Task-aware capacity helps some tasks: WinoGrande improves from `0.76` to `0.80`, PubMedQA from `0.74` to `0.76`.
2. The same capacity hurts other tasks: ARC-Challenge drops from `0.86` to `0.82`, HellaSwag from `0.78` to `0.76`, BoolQ from `0.92` to `0.90`.
3. Adding explicit label posterior loss is not automatically better. It preserves WinoGrande/PubMedQA gains, but hurts CommonsenseQA and BoolQ.
4. Current evidence supports TARDI-LoRA as the main method and Mature TaskNaRA as a serious negative result and boundary analysis.

## 7. Recommended Paper Wording

Do not claim:

```text
Task-aware noise-conditioned LoRA is the final best method.
```

Claim:

```text
We further test a more expressive task-aware noise-conditioned adapter with bounded residual task gates and task dropout. Although it improves WinoGrande and PubMedQA, it does not surpass the simpler task-balanced high-noise TARDI-LoRA protocol. This suggests that, under limited fixed-label supervision, task-aware capacity must be carefully regularized and is not a substitute for distribution/noise-stage alignment.
```

## 8. Files

```text
results/domain_shift/task_aware/lora_tasknara_mature_v1/raw/
results/domain_shift/task_aware/lora_tasknara_mature_v1/logs/
results/domain_shift/task_aware/lora_opt_v1/tables/lora_opt_macro_summary.csv
results/domain_shift/task_aware/lora_opt_v1/tables/lora_opt_task_summary.csv
```
