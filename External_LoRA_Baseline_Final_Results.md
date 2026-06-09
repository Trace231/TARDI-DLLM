# External LoRA Baseline Final Results

## 1. 实验目的

本实验补齐 LoRA 相关工作的公平对比：在同一 LLaDA-8B-Instruct、同一 fixed-label prompt、同一 9 个任务、同一 `limit=50` 与 `seed=23` 下，对比 vanilla LoRA、label-focused LoRA、choice-noise LoRA、rsLoRA、DoRA、LoRA+ 和 NaRA-style adapter。

相关方法覆盖：

- LoRA: 标准低秩适配。
- rsLoRA: 使用 rank-stabilized scaling。
- DoRA: 将权重分解为 magnitude 与 direction 后适配。
- LoRA+: 对 LoRA A/B 矩阵使用不同学习率。
- NaRA-style: 使用噪声水平驱动的动态低秩核心矩阵 `B C(lambda) A x`。

## 2. 实验设置

远端目录：

```text
/data/llada_eval/results/domain_shift/task_aware/lora_external_v1/
```

本地同步目录：

```text
results/domain_shift/task_aware/lora_external_v1/
```

任务：

```text
mmlu_pro, pubmedqa, ceval_computer_network, sciq,
winogrande, commonsenseqa, arc_challenge, hellaswag, boolq
```

每个任务 50 条，共 450 条。所有方法使用 32-step fixed-label evaluation。

## 3. Macro Results

| Method | Macro Acc. | Correct / N | 结论 |
|---|---:|---:|---|
| vanilla LoRA fixed-32 | 0.744 | 335 / 450 | 并列第一 |
| NaRA-style vanilla fixed-32 | 0.744 | 335 / 450 | 并列第一 |
| vanilla LoRA + controller | 0.742 | 334 / 450 | 几乎保分但省算 |
| LoRA+ vanilla fixed-32 | 0.742 | 334 / 450 | 接近第一 |
| NaRA-style choice-noise fixed-32 | 0.740 | 333 / 450 | 未超过 NaRA vanilla |
| choice-noise LoRA fixed-32 | 0.738 | 332 / 450 | 未超过 vanilla |
| rsLoRA vanilla fixed-32 | 0.736 | 331 / 450 | 低于 vanilla |
| label-focused LoRA fixed-32 | 0.733 | 330 / 450 | 负结果 |
| DoRA vanilla fixed-32 | 0.733 | 330 / 450 | 低于 vanilla |
| base fixed-32 | 0.722 | 325 / 450 | 基础线 |

## 4. 逐任务结果

| Method | MMLU-Pro | PubMedQA | C-Eval | SciQ | WinoGrande | CQA | ARC-C | HellaSwag | BoolQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla LoRA | 0.38 | 0.70 | 0.60 | 0.90 | 0.74 | 0.90 | 0.88 | 0.72 | 0.88 |
| NaRA vanilla | 0.38 | 0.72 | 0.58 | 0.90 | 0.72 | 0.88 | 0.90 | 0.72 | 0.90 |
| LoRA+ vanilla | 0.44 | 0.74 | 0.56 | 0.86 | 0.80 | 0.82 | 0.84 | 0.74 | 0.88 |
| NaRA choice-noise | 0.38 | 0.74 | 0.58 | 0.88 | 0.76 | 0.84 | 0.90 | 0.68 | 0.90 |
| rsLoRA vanilla | 0.38 | 0.74 | 0.60 | 0.88 | 0.78 | 0.84 | 0.86 | 0.70 | 0.84 |
| DoRA vanilla | 0.38 | 0.70 | 0.58 | 0.88 | 0.72 | 0.88 | 0.86 | 0.72 | 0.88 |

## 5. 核心结论

第一，**通用 improved LoRA 并没有稳定超过 vanilla LoRA**。rsLoRA、DoRA、LoRA+ 都是合理相关工作 baseline，但在 LLaDA fixed-label reasoning 上，最强的是 vanilla LoRA 与 NaRA-style vanilla 并列 0.744。

第二，**NaRA-style 的噪声感知结构是有信号的，但不是压倒性提升**。NaRA vanilla 达到 0.744，说明 noise-level-conditioned adapter 至少能持平最强 vanilla LoRA；但 NaRA choice-noise 下降到 0.740，说明额外 choice/noise objective 在当前样本规模和训练步数下没有带来稳定收益。

第三，**choice-aware / label-focused objective 不能直接声称成功**。label-focused LoRA 为 0.733，choice-noise LoRA 为 0.738，均低于 vanilla LoRA。报告中应把它写成探索性负结果：final-label posterior supervision 会改变任务偏置，但短训下并不保证 overall accuracy 提升。

第四，**真正稳定的正结果仍然是 inference controller**。vanilla LoRA + controller 达到 0.742，只比并列第一低 0.002，但此前结果显示平均 forward calls 从 32 降到约 25.24。因此 controller 可以作为“接近最强准确率下的推理预算控制”来讲，而不是声称改良 LoRA 全面胜出。

## 6. 推荐写法

不建议写：

```text
我们提出的 choice-noise LoRA 超过现有 LoRA 改进方法。
```

建议写：

```text
我们复现并比较了 rsLoRA、DoRA、LoRA+ 与 NaRA-style 等改良 LoRA。
实验显示，在 masked diffusion LM 的 fixed-label reasoning 中，通用 LoRA 改良并不会自动带来更高准确率；
vanilla denoising LoRA 与 NaRA-style adapter 并列最优，而 choice/label-focused 目标呈现任务相关的负迁移。
这一结果说明训练侧适配仍然存在 objective mismatch，因而本文将稳定贡献收束到推理侧：
在不改变 LLaDA backbone 的前提下，通过 selective re-masking controller 以极小准确率代价降低反向扩散预算。
```

## 7. 文件索引

```text
results/domain_shift/task_aware/lora_external_v1/raw/
results/domain_shift/task_aware/lora_external_v1/logs/
results/domain_shift/task_aware/lora_external_v1/tables/external_lora_macro_summary.csv
results/domain_shift/task_aware/lora_external_v1/tables/external_lora_task_summary.csv
results/domain_shift/task_aware/lora_external_v1/reports/External_LoRA_Baseline_Report.md
```

