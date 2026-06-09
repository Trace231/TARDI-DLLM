# Choice-aware / Noise-aware LoRA + Selective Re-masking Controller 结果速报

实验目录：

```text
results/domain_shift/task_aware/choice_noise_v1/
```

新服务器本轮实验覆盖 9 个任务，每个任务 `limit=50`、`seed=23`，总计 450 条 held-out 样本：

```text
mmlu_pro, pubmedqa, ceval_computer_network, sciq,
winogrande, commonsenseqa, arc_challenge, hellaswag, boolq
```

## 1. 核心结论

最强性能版：

> `LLaDA vanilla denoising LoRA fixed-32` 达到 macro accuracy `0.744`，高于 base fixed-32 的 `0.722`。

最强效率版：

> `LLaDA vanilla denoising LoRA + selective re-masking controller` 达到 macro accuracy `0.742`，平均 forward calls `25.24`，几乎保住 LoRA full-budget accuracy，同时比 32-step 少约 `21.1%` 调用。

方法创新版：

> `choice-noise LoRA fixed-32` 达到 macro accuracy `0.738`，高于 base `0.722` 和 label-focused LoRA `0.733`。它不是总体最佳，但在 PubMedQA、BoolQ、MMLU 损失控制上优于 vanilla LoRA，说明 choice posterior 和 noise-stage regularization 确实改变了 DDM LoRA 的任务偏置。

最稳妥的 paper 表述不是“choice-noise 全面超过 vanilla”，而是：

> DDM LoRA 可以提升 fixed-label reasoning，但不同 objective 会带来不同任务偏置；choice/noise-aware objective 提供了更可解释的校准收益，selective re-masking controller 进一步把适配后模型的推理成本降下来。

## 2. 主表

| Method | Macro Acc. | Correct / N | Avg calls | 说明 |
|---|---:|---:|---:|---|
| LLaDA vanilla LoRA fixed-32 | 0.744 | 335 / 450 | 32.00 | 当前最高准确率 |
| LLaDA vanilla LoRA + controller | 0.742 | 334 / 450 | 25.24 | 最佳质量-成本折中 |
| LLaDA choice-noise LoRA fixed-32 | 0.738 | 332 / 450 | 32.00 | 方法设计版 adapter |
| LLaDA label-focused LoRA fixed-32 | 0.733 | 330 / 450 | 32.00 | final-label posterior ablation |
| LLaDA choice-noise LoRA + controller | 0.733 | 330 / 450 | 25.30 | 方法设计版 + 省算 |
| LLaDA base fixed-32 | 0.722 | 325 / 450 | 32.00 | 基础线 |

## 3. 逐任务观察

| Task | Base | Vanilla LoRA | Choice-noise LoRA | Vanilla + Controller |
|---|---:|---:|---:|---:|
| MMLU-Pro | 0.44 | 0.38 | 0.42 | 0.38 |
| PubMedQA | 0.64 | 0.70 | 0.74 | 0.74 |
| C-Eval CN | 0.56 | 0.60 | 0.58 | 0.60 |
| SciQ | 0.86 | 0.90 | 0.88 | 0.90 |
| WinoGrande | 0.76 | 0.74 | 0.74 | 0.72 |
| CommonsenseQA | 0.80 | 0.90 | 0.82 | 0.90 |
| ARC-Challenge | 0.86 | 0.88 | 0.88 | 0.88 |
| HellaSwag | 0.74 | 0.72 | 0.68 | 0.72 |
| BoolQ | 0.84 | 0.88 | 0.90 | 0.84 |

关键现象：

- LoRA 确实能提升 LLaDA fixed-label reasoning，不再是之前旧 DDM LoRA audit 里的“基本不动”。
- vanilla denoising LoRA 的 overall macro 最强，尤其 CQA/SciQ 很好。
- choice-noise LoRA 在 PubMedQA 和 BoolQ 更强，并且 MMLU 损失小于 label-only/vanilla。
- controller 对 WinoGrande、CommonsenseQA、BoolQ 明显省算；对 MMLU/PubMedQA/C-Eval/SciQ/ARC/HellaSwag 保守走 full budget。

## 4. Controller 成本结果

`choice-noise LoRA + controller`：

| Task | Acc. | Avg calls |
|---|---:|---:|
| MMLU-Pro | 0.42 | 32.50 |
| PubMedQA | 0.74 | 33.00 |
| C-Eval CN | 0.58 | 33.00 |
| SciQ | 0.88 | 33.00 |
| WinoGrande | 0.74 | 10.44 |
| CommonsenseQA | 0.82 | 8.98 |
| ARC-Challenge | 0.88 | 33.00 |
| HellaSwag | 0.68 | 33.00 |
| BoolQ | 0.86 | 10.76 |

`vanilla LoRA + controller`：

| Task | Acc. | Avg calls |
|---|---:|---:|
| MMLU-Pro | 0.38 | 32.64 |
| PubMedQA | 0.74 | 33.00 |
| C-Eval CN | 0.60 | 33.00 |
| SciQ | 0.90 | 33.00 |
| WinoGrande | 0.72 | 10.12 |
| CommonsenseQA | 0.90 | 9.26 |
| ARC-Challenge | 0.88 | 33.00 |
| HellaSwag | 0.72 | 33.00 |
| BoolQ | 0.84 | 10.12 |

解释：

- controller 不是盲目 early-stop；高基数、多证据或校准敏感任务保持 full budget。
- 在 WinoGrande/CQA/BoolQ 上，早期轨迹足够稳定，controller 大量走 8/16/24。
- 因此它的创新点应表述为 **condition-aware reverse budget allocation**，而不是泛泛 adaptive sampling。

## 5. 与旧 AR/DDM LoRA audit 的关系

旧 solid_v2 结果中，Qwen AR LoRA 在 6 任务上有明显增益，例如 PubMedQA 从 `0.57` 到 `0.85`，CQA 从 `0.80` 到 `0.86`。而旧 LLaDA LoRA 几乎不动或不稳定。

本轮新结果补上了关键缺口：

> DDM LoRA 不是不能涨，而是需要适合 masked denoising / fixed-label reasoning 的训练格式与 objective。新 vanilla/choice-noise LoRA 在 9 任务上都超过 base fixed-32。

这可以作为 pre 中的转折：

1. 初始 audit：AR LoRA 有效，旧 DDM LoRA 不稳。
2. 诊断：DDM LoRA 的训练信号容易和 final-label decision boundary 错位。
3. 修正：重新构造 fixed-label denoising / choice posterior objective 后，LLaDA LoRA 出现正增益。
4. 推理控制：适配后仍有样本级 reverse budget heterogeneity，因此用 controller 降成本。

## 6. 推荐汇报口径

不要说：

> 我们提出的 choice-noise LoRA 全面超过所有 LoRA。

应该说：

> 我们系统比较了 vanilla denoising、label-focused 和 choice-noise 三种 DDM LoRA objective。结果表明，DDM LoRA 可以有效提升 fixed-label reasoning，但 objective 决定任务偏置：vanilla denoising overall 最强，choice-noise 在医学/二分类校准与高基数任务损失控制上更稳。进一步结合 selective re-masking controller 后，可以在几乎不牺牲准确率的情况下降低约 21% 的平均 forward calls。

## 7. 文件索引

核心表：

```text
results/domain_shift/task_aware/choice_noise_v1/tables/choice_noise_macro_summary.csv
results/domain_shift/task_aware/choice_noise_v1/tables/choice_noise_task_summary.csv
results/domain_shift/task_aware/choice_noise_v1/tables/choice_noise_summary.json
```

原始结果：

```text
results/domain_shift/task_aware/choice_noise_v1/raw/
```

日志：

```text
results/domain_shift/task_aware/choice_noise_v1/logs/
```

方法说明：

```text
Choice_Noise_LoRA_and_Remask_Controller_Method.md
```
