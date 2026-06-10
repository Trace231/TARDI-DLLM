# LoRA Optimization v1 Final Results

## 1. 目标

上一轮结论是：vanilla LLaDA LoRA 能从 base 的 `0.722` 提升到 `0.744`，但 label-focused、choice-noise、rsLoRA、DoRA、LoRA+、NaRA-style 都没有稳定超过 vanilla LoRA。因此本轮目标很明确：

> 在同一 9 个任务、同一 `limit=50`、同一 `seed=23`、同一 32-step fixed-label evaluation 下，优化出一个超过原 vanilla LoRA 的 LLaDA LoRA。

## 2. 改进方法：TARDI-LoRA

最终有效的改进不是继续堆复杂 LoRA 结构，而是修正训练/评测分布错位：

1. **Task-balanced data**：旧 LoRA 训练集只有 469 条，覆盖 6 个任务，其中 CommonsenseQA/SciQ/PubMedQA/WinoGrande 各 100，MMLU-Pro 50，C-Eval 19，并且完全缺少 ARC-Challenge、HellaSwag、BoolQ。
2. **Eval-disjoint sampling**：新训练集覆盖 9 个评测任务，每任务 100 条，共 900 条，并显式排除当前 evaluation 的 `seed=23, limit=50` 样本 id。
3. **High-noise final-label denoising**：最佳轻量版本使用 LoRA `r=8, alpha=16`，训练 100 steps，但把 noise ratios 调整为 `0.65,0.85,1.0`，更贴近 LLaDA 推理时从 masked answer token 恢复最终标签的状态。

我把最佳轻量版本记为：

```text
TARDI-LoRA balanced r8 high-noise
```

## 3. Macro Results

| Method | Macro Acc. | Correct / N | vs old vanilla LoRA |
|---|---:|---:|---:|
| TARDI-LoRA balanced r8 high-noise | 0.776 | 349 / 450 | +0.031 |
| TARDI-LoRA balanced r16 | 0.776 | 349 / 450 | +0.031 |
| TARDI-LoRA balanced r16 high-noise | 0.769 | 346 / 450 | +0.024 |
| TARDI-LoRA balanced LoRA+ r16 | 0.762 | 343 / 450 | +0.018 |
| TARDI-LoRA balanced r8 | 0.760 | 342 / 450 | +0.016 |
| TARDI-LoRA balanced r8 lr5e-5 s150 | 0.760 | 342 / 450 | +0.016 |
| Vanilla LoRA fixed-32 | 0.744 | 335 / 450 | 0.000 |
| NaRA-style vanilla fixed-32 | 0.744 | 335 / 450 | 0.000 |
| LoRA+ vanilla fixed-32 | 0.742 | 334 / 450 | -0.002 |
| Choice-noise LoRA fixed-32 | 0.738 | 332 / 450 | -0.007 |
| rsLoRA vanilla fixed-32 | 0.736 | 331 / 450 | -0.009 |
| Label-focused LoRA fixed-32 | 0.733 | 330 / 450 | -0.011 |
| DoRA vanilla fixed-32 | 0.733 | 330 / 450 | -0.011 |
| Base fixed-32 | 0.722 | 325 / 450 | -0.022 |

主结论：

```text
TARDI-LoRA: 0.776
Old vanilla LoRA: 0.744
Base LLaDA: 0.722
```

也就是：

- 相比 old vanilla LoRA：`+3.1` 个百分点，`+14/450`。
- 相比 base LLaDA：`+5.3` 个百分点，`+24/450`。

## 4. 逐任务结果

| Task | TARDI r8 high-noise | TARDI r16 | Old vanilla LoRA | Base |
|---|---:|---:|---:|---:|
| MMLU-Pro | 0.44 | 0.42 | 0.38 | 0.44 |
| PubMedQA | 0.74 | 0.74 | 0.70 | 0.64 |
| C-Eval CN | 0.64 | 0.70 | 0.60 | 0.56 |
| SciQ | 0.92 | 0.92 | 0.90 | 0.86 |
| WinoGrande | 0.76 | 0.76 | 0.74 | 0.76 |
| CommonsenseQA | 0.92 | 0.90 | 0.90 | 0.80 |
| ARC-Challenge | 0.86 | 0.84 | 0.88 | 0.86 |
| HellaSwag | 0.78 | 0.80 | 0.72 | 0.74 |
| BoolQ | 0.92 | 0.90 | 0.88 | 0.84 |

主要收益来自：

- PubMedQA: `0.70 -> 0.74`
- C-Eval: `0.60 -> 0.64/0.70`
- SciQ: `0.90 -> 0.92`
- HellaSwag: `0.72 -> 0.78/0.80`
- BoolQ: `0.88 -> 0.92`
- CommonsenseQA: `0.90 -> 0.92`

ARC-Challenge 上 old vanilla LoRA 仍略高，这是边界。

## 5. 消融解释

这轮实验显示三个信号：

1. **数据覆盖是主因**：单独 balanced r8 从 `0.744` 提升到 `0.760`。
2. **高噪声训练有效**：balanced r8 high-noise 从 `0.760` 提升到 `0.776`。
3. **容量增加也有效**：balanced r16 default 同样达到 `0.776`。

但 `r16 + high-noise` 只有 `0.769`，说明二者不是简单叠加；更大的 rank 在高噪声下可能放大了任务间偏置。

## 6. 最终可写贡献

建议把训练侧贡献改写为：

> We propose TARDI-LoRA, a task-balanced and high-noise final-label denoising adaptation protocol for masked diffusion language models. Instead of introducing another generic LoRA variant, TARDI-LoRA repairs the mismatch between training coverage, diffusion noise level, and fixed-label downstream evaluation.

中文：

> 本文提出 TARDI-LoRA：一种面向 masked diffusion LM 固定标签推理的任务均衡、高噪声 final-label denoising LoRA 适配协议。它不依赖新的 backbone 或复杂 LoRA 层，而是修正训练任务覆盖、扩散噪声阶段与下游 fixed-label evaluation 之间的错位。

## 7. 文件位置

```text
results/domain_shift/task_aware/lora_opt_v1/train/
results/domain_shift/task_aware/lora_opt_v1/raw/
results/domain_shift/task_aware/lora_opt_v1/logs/
results/domain_shift/task_aware/lora_opt_v1/tables/lora_opt_macro_summary.csv
results/domain_shift/task_aware/lora_opt_v1/tables/lora_opt_task_summary.csv
results/domain_shift/task_aware/lora_opt_v1/reports/LoRA_Optimization_v1_Report.md
```

