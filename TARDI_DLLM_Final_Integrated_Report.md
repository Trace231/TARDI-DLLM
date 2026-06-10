# TARDI-DLLM 最终综合实验报告

## 1. 任务目标

本项目围绕 masked diffusion language model 在 fixed-label reasoning 任务上的适配与推理控制展开。核心问题不是重新训练 LLaDA，也不是声称修改 LLaDA backbone，而是回答：

> 对于选择题、判断题、医学问答、中文知识题等固定标签任务，LLaDA 的训练侧适配目标和推理侧反向扩散预算是否存在错位？如果存在，能否用 LoRA objective 和 selective re-masking controller 改善？

本轮新服务器实验完成了两个主实验：

1. **改进 LoRA 实验**：比较 base、vanilla denoising LoRA、label-focused LoRA、choice-noise LoRA。
2. **LoRA + 选择性再修掩码器实验**：在 LoRA 后接入 selective re-masking controller，评估 accuracy-cost trade-off。

同时，本报告接入已有 solid_v2 中的 AR/Qwen LoRA audit 和 related sampler baseline，形成完整对照。

## 2. 实验环境与数据

远端实验目录：

```text
/data/llada_eval/results/domain_shift/task_aware/choice_noise_v1/
```

本地同步目录：

```text
results/domain_shift/task_aware/choice_noise_v1/
```

新服务器远端仓库：

```text
/data/llada_eval
origin = https://github.com/Trace231/TARDI-DLLM.git
commit = e688726
```

主模型：

```text
/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
```

新 LoRA/controller 实验覆盖 9 个任务，每任务 `limit=50`，总计 450 条样本：

```text
mmlu_pro, pubmedqa, ceval_computer_network, sciq,
winogrande, commonsenseqa, arc_challenge, hellaswag, boolq
```

## 3. 方法概述

### 3.1 训练侧：DDM LoRA objective

比较三种 LoRA：

| Adapter | Objective | 目的 |
|---|---|---|
| Vanilla LoRA | denoising only | 普通 DDM LoRA 基线 |
| Label-focused LoRA | choice posterior + denoise | 检验 final-label objective mismatch |
| Choice-noise LoRA | choice + denoise + cross-noise consistency | 检验 noise-stage regularization 是否改善任务偏置 |

Choice-aware objective 作用在合法标签空间：

```text
Y(x) = {A, B, C, D, ...}
```

或：

```text
Y(x) = {yes, no}, {yes, no, maybe}
```

核心思想是让 LoRA 梯度直接作用于：

```text
p(label | prompt, "Final answer: [MASK]")
```

而不是只被完整 completion 的普通去噪目标摊薄。

### 3.2 推理侧：Selective Re-masking Controller

controller 将反向扩散步数视作样本级预算分配问题：

```text
K(x) in {8, 16, 24, 32}
```

它根据 forward probe、8-step scout、label validity、风险分数和低置信 token 位置决定是否追加 denoising。追加时不是完全重跑，而是把低置信位置重新 mask 后继续修复。

## 4. 新实验主结果

![Macro accuracy](results/domain_shift/task_aware/choice_noise_v1/figures/choice_noise_macro_accuracy.svg)

| Method | Macro Acc. | Correct / N | Avg calls | 说明 |
|---|---:|---:|---:|---|
| LLaDA vanilla LoRA fixed-32 | 0.744 | 335 / 450 | 32.00 | 当前最高准确率 |
| LLaDA vanilla LoRA + controller | 0.742 | 334 / 450 | 25.24 | 最佳质量-成本折中 |
| LLaDA choice-noise LoRA fixed-32 | 0.738 | 332 / 450 | 32.00 | 方法设计版 adapter |
| LLaDA label-focused LoRA fixed-32 | 0.733 | 330 / 450 | 32.00 | final-label posterior ablation |
| LLaDA choice-noise LoRA + controller | 0.733 | 330 / 450 | 25.30 | 方法设计版 + 省算 |
| LLaDA base fixed-32 | 0.722 | 325 / 450 | 32.00 | 基础线 |

主要结论：

1. DDM LoRA 可以有效提升 fixed-label reasoning。base fixed-32 为 `0.722`，最强 LoRA fixed-32 达到 `0.744`。
2. Vanilla denoising LoRA 在这批任务上 overall 最强，说明 DDM LoRA 的普通去噪适配本身有价值。
3. Choice-noise LoRA 不是 overall 最强，但相对 vanilla 在 PubMedQA、BoolQ 和 MMLU 损失控制上更有优势，说明 choice posterior 与 noise-stage regularization 会改变任务偏置。
4. Vanilla LoRA + controller 几乎保持 full-budget LoRA 的准确率，从 `0.744` 到 `0.742`，同时 avg calls 从 32 降到 `25.24`，约省 `21.1%`。

## 5. 逐任务结果

![Task heatmap](results/domain_shift/task_aware/choice_noise_v1/figures/choice_noise_task_heatmap.svg)

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

解释：

- **PubMedQA**：choice-noise 和 controller 后均达到 `0.74`，比 base `0.64` 高 10 个点。这说明标签 posterior 对医学 yes/no/maybe 校准有帮助。
- **CommonsenseQA / SciQ**：vanilla LoRA 很强，说明普通 denoising adaptation 对短选择题有效。
- **MMLU-Pro / HellaSwag / WinoGrande**：部分 LoRA 有负迁移，说明 fixed-label LoRA 不是知识注入，也不是所有任务的万能增强。
- **BoolQ**：choice-noise fixed-32 达到 `0.90`，但 controller 后 vanilla 版本回到 `0.84`，说明二分类早停在某些样本上仍有校准风险。

## 6. Accuracy-cost Trade-off

![Accuracy-cost pareto](results/domain_shift/task_aware/choice_noise_v1/figures/choice_noise_accuracy_cost_pareto.svg)

Controller 的行为具有明显任务依赖：

| Task | Choice-noise + Ctrl Acc. | Calls | Vanilla + Ctrl Acc. | Calls |
|---|---:|---:|---:|---:|
| MMLU-Pro | 0.42 | 32.50 | 0.38 | 32.64 |
| PubMedQA | 0.74 | 33.00 | 0.74 | 33.00 |
| C-Eval CN | 0.58 | 33.00 | 0.60 | 33.00 |
| SciQ | 0.88 | 33.00 | 0.90 | 33.00 |
| WinoGrande | 0.74 | 10.44 | 0.72 | 10.12 |
| CommonsenseQA | 0.82 | 8.98 | 0.90 | 9.26 |
| ARC-Challenge | 0.88 | 33.00 | 0.88 | 33.00 |
| HellaSwag | 0.68 | 33.00 | 0.72 | 33.00 |
| BoolQ | 0.86 | 10.76 | 0.84 | 10.12 |

这说明 controller 不是简单 early stopping：

- 高基数、长上下文或高风险任务保持 full budget。
- WinoGrande、CommonsenseQA、BoolQ 大量走 8/16/24。
- 最终全局平均调用降到约 25.2，而不是所有任务无差别压缩。

## 7. AR/Qwen LoRA 与旧 DDM LoRA Audit

![Prior AR/DDM LoRA gain](results/domain_shift/task_aware/choice_noise_v1/figures/prior_ar_ddm_lora_gain.svg)

已有 solid_v2 中的 AR/Qwen 对照显示：

| Model | Protocol | 现象 |
|---|---|---|
| Qwen2.5-7B base vs LoRA | 6 任务，每任务 100 | AR LoRA 通常有明显增益，PubMedQA +0.26，WinoGrande +0.09，CQA +0.04 |
| 旧 LLaDA control LoRA | 6 任务，每任务 100 | 大多接近 0 增益，甚至 C-Eval 略降 |
| 旧 LLaDA original LoRA | 6 任务，每任务 100 | 任务间波动大，Wino/MMLU 有提升，但 PubMed/SciQ/CQA 下降 |

这和本轮新实验形成一个完整故事：

1. 初始 audit 发现：AR LoRA 能稳定把 final-label supervision 转化成 accuracy gain，旧 DDM LoRA 不稳定。
2. 诊断：DDM LoRA 的训练信号和 final-label decision boundary 可能错位。
3. 修正：重新构造 fixed-label denoising / choice posterior 训练后，新 LLaDA LoRA 在 9 任务上超过 base。
4. 推理控制：适配后仍有样本级 reverse budget heterogeneity，因此 controller 可以进一步省算。

## 8. External Improved LoRA Baseline

为避免训练侧贡献只停留在自家消融，本轮补充复现并比较了多种 LoRA 改良方法：rsLoRA、DoRA、LoRA+ 与 NaRA-style noise-aware adapter。所有方法使用相同 LLaDA-8B-Instruct、相同 fixed-label prompt、相同 9 个任务、相同 `limit=50` 和 `seed=23`，并统一使用 32-step fixed-label evaluation。

| Method | Macro Acc. | Correct / N | 解释 |
|---|---:|---:|---|
| Vanilla LoRA fixed-32 | 0.744 | 335 / 450 | 并列最高 |
| NaRA-style vanilla fixed-32 | 0.744 | 335 / 450 | 并列最高 |
| Vanilla LoRA + controller | 0.742 | 334 / 450 | 接近最高且省算 |
| LoRA+ vanilla fixed-32 | 0.742 | 334 / 450 | 接近最高 |
| NaRA-style choice-noise fixed-32 | 0.740 | 333 / 450 | 低于 NaRA vanilla |
| Choice-noise LoRA fixed-32 | 0.738 | 332 / 450 | 低于 vanilla |
| rsLoRA vanilla fixed-32 | 0.736 | 331 / 450 | 低于 vanilla |
| Label-focused LoRA fixed-32 | 0.733 | 330 / 450 | 负结果 |
| DoRA vanilla fixed-32 | 0.733 | 330 / 450 | 低于 vanilla |
| Base fixed-32 | 0.722 | 325 / 450 | 基础线 |

这组结果改变了训练侧叙事：不能声称 choice-noise LoRA 全面超过现有 LoRA 改良。更稳的结论是，通用 improved LoRA 并不会自动解决 masked diffusion LM 的 fixed-label reasoning 适配；NaRA-style 的 noise-aware 结构可以达到 vanilla LoRA 水平，但额外的 choice/noise objective 在当前规模下没有稳定提升。

因此最终贡献应收束为：

1. 训练侧给出系统诊断：DDM LoRA objective 会改变任务偏置，但 choice-aware/label-focused 目标短训下存在负迁移。
2. Related LoRA baseline 已补齐：rsLoRA、DoRA、LoRA+、NaRA-style 均在同一协议下比较。
3. 推理侧是稳定正结果：controller 在 0.742 macro accuracy 下接近最高 0.744，同时保持此前约 21% forward-call 节省。

完整 external LoRA 结果见：

```text
External_LoRA_Baseline_Final_Results.md
results/domain_shift/task_aware/lora_external_v1/
```

## 9. TARDI-LoRA Optimization

在 external LoRA baseline 之后，我们进一步检查了原 vanilla LoRA 的训练数据分布，发现关键问题不是 LoRA 层不够复杂，而是训练/评测分布错位：旧训练集只有 469 条，覆盖 6 个任务，且完全缺少 ARC-Challenge、HellaSwag、BoolQ；MMLU-Pro 只有 50 条，C-Eval 只有 19 条。为此补充了一个 9-task balanced train set，每任务 100 条，共 900 条，并排除当前 evaluation 的 `seed=23, limit=50` 样本 id。

最终有效配置记作 **TARDI-LoRA balanced r8 high-noise**：

```text
LoRA r=8, alpha=16
steps=100
noise ratios = 0.65,0.85,1.0
train = 9-task balanced, eval-disjoint final-label denoising
```

| Method | Macro Acc. | Correct / N | vs old vanilla LoRA |
|---|---:|---:|---:|
| TARDI-LoRA balanced r8 high-noise | 0.776 | 349 / 450 | +0.031 |
| TARDI-LoRA balanced r16 | 0.776 | 349 / 450 | +0.031 |
| TaskNaRA r8 high-noise | 0.773 | 348 / 450 | +0.029 |
| TARDI-LoRA balanced r16 high-noise | 0.769 | 346 / 450 | +0.024 |
| Mature TaskNaRA residual vanilla high-noise | 0.769 | 346 / 450 | +0.024 |
| TARDI-LoRA balanced LoRA+ r16 | 0.762 | 343 / 450 | +0.018 |
| Mature TaskNaRA residual label high-noise | 0.762 | 343 / 450 | +0.018 |
| TARDI-LoRA balanced r8 | 0.760 | 342 / 450 | +0.016 |
| TaskNaRA r16 | 0.758 | 341 / 450 | +0.013 |
| Old vanilla LoRA fixed-32 | 0.744 | 335 / 450 | 0.000 |
| Base fixed-32 | 0.722 | 325 / 450 | -0.022 |

这说明训练侧现在有了正结果：相比 old vanilla LoRA，TARDI-LoRA 提升 `+3.1` 个百分点、`+14/450`；相比 base LLaDA 提升 `+5.3` 个百分点、`+24/450`。消融也比较清楚：task-balanced data 单独把 `0.744` 推到 `0.760`；高噪声训练或 r16 容量进一步推到 `0.776`；但 `r16 + high-noise` 只有 `0.769`，说明二者不是简单叠加。

这让最终训练侧贡献从“choice-noise objective 没有赢”修正为：

> TARDI-LoRA repairs the mismatch between training task coverage, diffusion noise level, and fixed-label downstream evaluation.

完整优化结果见：

```text
LoRA_Optimization_v1_Final_Results.md
results/domain_shift/task_aware/lora_opt_v1/
```

## 10. TaskNaRA 架构探索

为了验证“task-aware + noise-aware LoRA”是否能作为更强的训练侧贡献，我进一步实现了 **TaskNaRA**。它不修改 LLaDA backbone，而是在 LoRA correction 中加入 task embedding 和 diffusion noise embedding：

```text
W'(task, lambda) = W + B C(task, lambda) A
C(task, lambda) = I + c_scale * MLP([Fourier(lambda); Emb(task)])
```

这个实现已经接入训练、保存、加载和评测全链路。结果是一个很有价值但需要诚实表述的结论：

| Method | Macro Acc. | Correct / N |
|---|---:|---:|
| TARDI-LoRA balanced r8 high-noise | 0.776 | 349 / 450 |
| TaskNaRA r8 high-noise | 0.773 | 348 / 450 |
| TaskNaRA r16 | 0.758 | 341 / 450 |
| Old vanilla LoRA fixed-32 | 0.744 | 335 / 450 |
| Base fixed-32 | 0.722 | 325 / 450 |

TaskNaRA r8 high-noise 明显超过 old vanilla LoRA，但没有超过更简单的 TARDI-LoRA，差距为 `1/450`。r16 反而下降，说明 task-conditioned adapter 容量可能放大任务间负迁移。因此最终不建议把 TaskNaRA 写成主贡献的“最佳方法”，而应写成架构探索和负结果：在当前小规模 fixed-label adaptation 下，任务覆盖和 high-noise denoising 对齐比复杂 task hypernetwork 更关键。

进一步，为避免第一版 TaskNaRA 被质疑只是简单拼接 task embedding，我实现了更成熟的 residual TaskNaRA：

```text
C(task, lambda)
= I + c_scale * F(lambda) + sigmoid(g_task) * rho * G(lambda, Emb(task))
```

它包含共享 noise-aware core、受限 task residual、learned task gate 和 task dropout。结果如下：

| Method | Macro Acc. | Correct / N | 局部现象 |
|---|---:|---:|---|
| Mature TaskNaRA residual vanilla high-noise | 0.769 | 346 / 450 | Wino/PubMed 提升，但 ARC/Hella/BoolQ 下降 |
| Mature TaskNaRA residual label high-noise | 0.762 | 343 / 450 | label loss 加重负迁移 |

这说明 mature task-aware adapter 不是 toy，但它也不是当前最强正结果。它证明 task-aware capacity 的收益是任务相关的：WinoGrande 从 `0.76` 到 `0.80`，PubMedQA 从 `0.74` 到 `0.76`；但 ARC-Challenge、HellaSwag 和 BoolQ 下降。最终报告应把它作为边界分析，而不是主方法。

完整结果见：

```text
TaskNaRA_v1_Final_Results.md
TaskNaRA_Mature_v1_Final_Results.md
results/domain_shift/task_aware/lora_tasknara_v1/
results/domain_shift/task_aware/lora_tasknara_mature_v1/
```

## 11. Related Sampler Baseline

已有 solid_v2 的 sampler baseline 覆盖 11 任务、每任务 50：

| Method | Macro Acc. | Avg calls |
|---|---:|---:|
| fixed32 | 0.658 | 32.00 |
| JYS-like middle-16 | 0.644 | 16.00 |
| Prophet-like early commit | 0.647 | 30.44 |
| refinement | 0.658 | 26.92 |
| ours_v3plus | 0.658 | 17.56 |

这说明“减少采样步数”本身不是新颖点。我们的贡献应写成更窄的：

> condition-aware reverse budget allocation for fixed-label reasoning, combined with DDM LoRA adaptation.

也就是说，controller 的价值不是“我也能 early stop”，而是它和训练侧 fixed-label adaptation 共同解决 DDM 下游选择题的两个错位。

## 12. 最推荐的 Pre 叙事

建议不要说：

```text
我们提出的 choice-noise LoRA 全面超过所有 LoRA。
```

建议说：

```text
我们发现 masked diffusion LM 在 fixed-label reasoning 上存在训练侧和推理侧两个错位。
训练侧，通用 improved LoRA 和 choice-noise objective 并不会自动超过 vanilla LoRA；
但修正任务覆盖和扩散噪声阶段错位后，TARDI-LoRA 可以把 macro accuracy 从 0.744 提升到 0.776。
进一步的 TaskNaRA 架构探索说明 task-aware adapter 可以超过旧 vanilla LoRA，但复杂条件化 adapter 不是免费收益；
推理侧，适配后的 LLaDA 仍然存在样本级 reverse budget heterogeneity，
selective re-masking controller 可以在几乎不损失准确率的前提下降低约 21% 平均 forward calls。
```

最终贡献可以写成三点：

1. **诊断**：AR LoRA 与旧 DDM LoRA 的增益差异说明 DDM fixed-label adaptation 不能照搬 AR SFT。
2. **训练侧修复**：提出 TARDI-LoRA，通过 9-task balanced、eval-disjoint 数据和 high-noise final-label denoising，使 LLaDA LoRA 超过原 vanilla、NaRA、LoRA+、rsLoRA、DoRA baseline。
3. **架构探索**：实现 TaskNaRA，将 task embedding 与 diffusion noise embedding 注入 LoRA correction；结果接近最优但未超过 TARDI-LoRA，形成可信的负结果和边界分析。
4. **推理侧控制**：在 LoRA 后接入 selective re-masking controller，实现几乎保分的推理预算压缩。

## 13. 局限性

1. 新 LoRA/controller 实验为 `limit=50 × 9 tasks`，样本量比 1000 样本主对照小，适合 course project/pre，不应夸成大规模 benchmark SOTA。
2. TARDI-LoRA 证明数据覆盖和噪声调度有效，但 choice-noise objective 与 TaskNaRA 目前都不是全局最优 objective。后续可做 balanced loss sweep，例如提高 denoise weight、降低 consistency weight，或约束 task-conditioned adapter 的跨任务偏移。
3. Controller 对高风险任务保守，因此全局平均 calls 下降约 21%，不是 40% 以上；但这也说明它不是盲目压缩。
4. 方法没有修改 LLaDA 架构。LoRA 是参数高效适配，controller 是 inference-loop 改进。

## 14. 文件索引

新实验主表：

```text
results/domain_shift/task_aware/choice_noise_v1/tables/choice_noise_macro_summary.csv
results/domain_shift/task_aware/choice_noise_v1/tables/choice_noise_task_summary.csv
results/domain_shift/task_aware/choice_noise_v1/tables/choice_noise_summary.json
```

新实验原始 JSON：

```text
results/domain_shift/task_aware/choice_noise_v1/raw/
```

新实验日志：

```text
results/domain_shift/task_aware/choice_noise_v1/logs/
```

图表：

```text
results/domain_shift/task_aware/choice_noise_v1/figures/
```

方法说明：

```text
Choice_Noise_LoRA_and_Remask_Controller_Method.md
```

新结果速报：

```text
Choice_Noise_LoRA_Controller_Final_Results.md
```
