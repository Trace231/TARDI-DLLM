# LLaDA Solid Experiment Report

## 1. Problem and Thesis

我们不把贡献写成“改 LLaDA 架构”。主张是：LLaDA 的 masked diffusion trajectory 暴露了不同任务、不同样本的反向去噪预算需求，因此可以用 task/sample-aware controller 在 accuracy-cost 之间做风险受控分配。

核心方法是 **Calibrated Forward-aware Risk Controller**：先用低预算 probe 估计 posterior confidence 与 label disagreement，再按风险路由到 cheap / medium / full reverse budget。LoRA 只作为辅助审计，用来回答普通下游微调是否能解决同一类行为边界。

## 2. Exploration Process

1. 先跑 typed final-label benchmark，避免 prompt 格式混淆。
2. 再做 old adaptive router，发现它很省算力，但 WinoGrande 上会越过 full-budget 边界。
3. 接着做 forward-aware probe，把早期 confidence、fallback 与最终正确性关联起来。
4. 最后加入 calibration threshold 与 multi-choice disagreement policy，形成主方法。
5. 用 PubMedQA/C-Eval 作为负例边界：证明 controller 不是万能调参，而是在可由 reverse budget 影响的区域有效。
6. 最后补充 ARC-Challenge、HellaSwag、BoolQ 与 GSM8K coverage addendum，检查结论是否只依赖原始选择题集合。

## 3. Method

Controller 的 inference loop 可以抽象为三段：

- **Forward probe**：用 cheap reverse trajectory 得到初始 label posterior 与 top probability。
- **Scout decision**：检测 binary 任务中的低置信度、近邻混淆，以及 multi-choice disagreement。
- **Calibrated route**：在 cheap / medium / full 预算之间选择；高风险 binary 样本保守 fallback，多选任务默认忽略轻微 disagreement，减少不必要的 full-budget 调用。

## 4. Mathematical Idea

离散扩散可以看成 token 状态空间上的连续时间马尔可夫链或其离散化近似。反向生成时，模型给出近似 posterior / score，用有限步数从 masked/corrupted state 回到数据分布。这里的关键量不是单个 token 的局部置信度，而是最终 label 的 hitting time：达到并稳定在最终答案所需的反向步数。

因此 controller 优化的是一个 cost-sensitive risk：

`min_pi E[1{y_hat_pi(x) != y} + lambda C(pi, x)]`

其中策略 `pi` 根据 probe trajectory 的 uncertainty 选择预算。若某任务 hitting time 分布更晚，比如 WinoGrande，策略应更保守；若某任务早期 label posterior 已稳定，比如 CommonsenseQA，则可以更激进地节省预算。

## 5. Main 1000 Results

| Method | Task | Acc | 95% CI | Avg Calls | Seconds | Route/Fallback |
|---|---|---:|---:|---:|---:|---|
| 32step | WinoGrande | 0.756 | [0.728, 0.782] | 32.000 | 1555.6 | `{"fixed": 1.0}` |
| 32step | CommonsenseQA | 0.819 | [0.794, 0.842] | 32.000 | 1550.2 | `{"fixed": 1.0}` |
| old_adaptive | WinoGrande | 0.736 | [0.708, 0.762] | 9.363 | 408.5 | `{"fallback": 0.011, "fixed": 0.989}` |
| old_adaptive | CommonsenseQA | 0.816 | [0.791, 0.839] | 9.858 | 432.3 | `{"fallback": 0.026, "fixed": 0.974}` |
| forward_aware | WinoGrande | 0.756 | [0.728, 0.782] | 17.560 | 847.7 | `{"accepted_fast": 0.648, "post_fallback": 0.014, "pre_fallback": 0.338}` |
| forward_aware | CommonsenseQA | 0.818 | [0.793, 0.841] | 10.824 | 523.6 | `{"accepted_fast": 0.943, "post_fallback": 0.057}` |
| calibrated | WinoGrande | 0.756 | [0.728, 0.782] | 17.560 | 852.4 | `{"32": 0.338, "8": 0.648, "8->32": 0.014}` |
| calibrated | CommonsenseQA | 0.817 | [0.792, 0.840] | 9.064 | 442.0 | `{"8": 0.998, "8->32": 0.002}` |

## Paired McNemar

| Task | Reference | Candidate | n | Ref Only | Cand Only | Net | p |
|---|---|---|---:|---:|---:|---:|---:|
| WinoGrande | 32step | calibrated | 1000 | 3 | 3 | 0 | 1.0000 |
| CommonsenseQA | 32step | calibrated | 1000 | 8 | 6 | -2 | 0.7905 |
| WinoGrande | old_adaptive | calibrated | 1000 | 25 | 45 | 20 | 0.0225 |
| CommonsenseQA | old_adaptive | calibrated | 1000 | 0 | 1 | 1 | 1.0000 |
| WinoGrande | forward_aware | calibrated | 1000 | 0 | 0 | 0 | 1.0000 |
| CommonsenseQA | forward_aware | calibrated | 1000 | 4 | 3 | -1 | 1.0000 |

## 6. Threshold Robustness

| Threshold | Task | Acc | Avg Calls | Route Rates |
|---:|---|---:|---:|---|
| 0.60 | WinoGrande | 0.754 | 15.704 | `{"32": 0.194, "8": 0.742, "8->32": 0.064}` |
| 0.60 | CommonsenseQA | 0.812 | 9.000 | `{"8": 1.0}` |
| 0.65 | WinoGrande | 0.756 | 16.456 | `{"32": 0.268, "8": 0.7, "8->32": 0.032}` |
| 0.65 | CommonsenseQA | 0.812 | 9.000 | `{"8": 1.0}` |
| 0.70 | WinoGrande | 0.756 | 17.768 | `{"32": 0.344, "8": 0.64, "8->32": 0.016}` |
| 0.70 | CommonsenseQA | 0.812 | 9.000 | `{"8": 1.0}` |
| 0.75 | WinoGrande | 0.756 | 20.568 | `{"32": 0.466, "8": 0.522, "8->32": 0.012}` |
| 0.75 | CommonsenseQA | 0.812 | 9.000 | `{"8": 1.0}` |
| 0.80 | WinoGrande | 0.756 | 22.936 | `{"32": 0.578, "8": 0.42, "8->32": 0.002}` |
| 0.80 | CommonsenseQA | 0.812 | 9.000 | `{"8": 1.0}` |

## 7. Boundary Negative Cases

| Method | Task | Acc | Avg Calls | Notes |
|---|---|---:|---:|---|
| 8step | PubMedQA | 0.557 |  | label-prior / knowledge-boundary probe |
| 8step | C-Eval CN | 0.585 |  | label-prior / knowledge-boundary probe |
| 32step | PubMedQA | 0.660 | 32.000 | label-prior / knowledge-boundary probe |
| 32step | C-Eval CN | 0.585 | 32.000 | label-prior / knowledge-boundary probe |
| calibrated | PubMedQA | 0.680 | 33.000 | label-prior / knowledge-boundary probe |
| calibrated | C-Eval CN | 0.585 | 33.000 | label-prior / knowledge-boundary probe |

## 8. Expanded Downstream Coverage

原始主实验聚焦 WinoGrande 与 CommonsenseQA，因为它们代表两类非常不同的 reverse diffusion behavior：二元语义绑定与多选常识推理。为了避免结论显得只在两个选择题任务上成立，我们补充了一个 coverage addendum，加入科学推理、情境续写、阅读判断与长链数学 answer-only 任务。

这些实验不是为了重新调一个覆盖所有任务的 controller，而是用来回答两个问题：

- LLaDA 的 reverse-budget sensitivity 是否也出现在新任务上？
- 与 AR 模型 Qwen2.5-7B 相比，LLaDA 的能力分布和预算依赖有什么差异？

| Task | Type | LLaDA 8-step | LLaDA 32-step | Calibrated | Qwen2.5-7B | Main Observation |
|---|---|---:|---:|---:|---:|---|
| ARC-Challenge | science reasoning | 0.823 | 0.857 | 0.857 | 0.890 | 32-step 比 8-step 高 3.3%，Qwen 略强 |
| HellaSwag | situation continuation | 0.760 | 0.780 | 0.780 | 0.850 | 预算收益较小，Qwen 明显更强 |
| BoolQ | reading yes/no | 0.843 | 0.867 | 0.853 | 0.823 | LLaDA 32-step 强于 Qwen，但 controller 略低于 full budget |
| GSM8K | answer-only math reasoning | 0.320 | 0.580 | - | 0.770 | 长链数学极度预算敏感，但 Qwen 仍明显更强 |

对应 delta 如下：

| Task | 32-step - 8-step | Calibrated - 32-step | LLaDA 32-step - Qwen |
|---|---:|---:|---:|
| ARC-Challenge | +0.033 | +0.000 | -0.033 |
| HellaSwag | +0.020 | +0.000 | -0.070 |
| BoolQ | +0.023 | -0.013 | +0.043 |
| GSM8K | +0.260 | - | -0.190 |

Coverage addendum 的结论是：reverse budget sensitivity 不只存在于 WinoGrande/CommonsenseQA。GSM8K 上 8-step 到 32-step 的提升达到 26 个百分点，说明长链 answer-only 推理对反向扩散预算非常敏感；ARC-Challenge、HellaSwag、BoolQ 也有 2-3 个点的 32-step 收益。另一方面，Qwen2.5-7B 在 ARC、HellaSwag 和 GSM8K 上更强，说明 LLaDA 不是全面优于 AR；我们的主张应写成 **LLaDA 的推理过程具有可观测、可控制的任务依赖性**，而不是 LLaDA 在所有任务上都更强。

Calibrated controller 在 ARC-Challenge 和 HellaSwag 上选择全 32-step，因此守住了 full-budget 边界，但没有加速；在 BoolQ 上平均调用约 12.87 次，准确率为 0.853，低于 32-step 的 0.867。这个负面结果很重要：它说明当前 controller 的校准主要来自 WinoGrande/CommonsenseQA，不应夸大为一次性泛化到所有任务。更合理的表述是：该方法提供了一个可扩展的风险控制框架，后续需要加入更多任务的 trajectory statistics 来学习更细的 task-risk estimator。

完整 coverage 文件见 `coverage_addendum/tables/coverage_summary.csv` 与 `coverage_addendum/reports/Coverage_Addendum_Report.md`。

## 9. Controlled LoRA-v2 Adaptation Audit

为避免把不同训练目标、不同训练步数的 checkpoint 误写成公平比较，我们补充了 controlled LoRA-v2。该实验使用同一份 `domain_mix_final_typed_control_seed23.jsonl` 训练集，LoRA 均为 `r=8, alpha=16, dropout=0.05`，训练 200 update steps，并在同一 100-sample typed final-label evaluation 上比较 gain。AR 采用其原生 causal final-label SFT；DDM 采用 LLaDA 原生 diffusion GRPO final-label reward。原已有 DDM LoRA checkpoint 作为额外参照，而不是 controlled comparison 主体。

共享训练集共 469 条：CommonsenseQA/SciQ/PubMedQA/WinoGrande 各 100，MMLU-Pro 50，C-Eval CN 19。C-Eval 和 MMLU-Pro 样本数较少，因此 LoRA-v2 结论应定位为 exploratory controlled audit，而非大规模微调结论。

| Task | Qwen Base | Qwen LoRA-v2 | AR Gain | LLaDA Base | LLaDA LoRA-v2 | DDM Gain |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | 0.380 | 0.400 | +0.020 | 0.350 | 0.350 | +0.000 |
| PubMedQA | 0.510 | 0.770 | +0.260 | 0.750 | 0.750 | +0.000 |
| C-Eval CN | 0.730 | 0.750 | +0.020 | 0.580 | 0.570 | -0.010 |
| SciQ | 0.970 | 0.970 | +0.000 | 0.920 | 0.920 | +0.000 |
| WinoGrande | 0.630 | 0.720 | +0.090 | 0.630 | 0.630 | +0.000 |
| CommonsenseQA | 0.820 | 0.860 | +0.040 | 0.870 | 0.870 | +0.000 |

该结果说明，在相同数据和相同 LoRA 参数预算下，AR 的 final-label SFT 更容易把监督信号转化为 `p(y|x)` 的增益；LLaDA 的 DDM-native GRPO LoRA 在 200 steps 下几乎不改变 held-out final-label accuracy。理论上，AR LoRA 直接扰动最终答案 token 的条件分布，而 DDM LoRA 扰动的是一族时间相关的反向转移分布；最终答案分布是多步转移复合后的 hitting distribution，因此小规模低秩更新未必能稳定改变最终决策边缘分布。这个结果不说明 DDM 不能微调，而说明 DDM 的 PEFT 适配路径可能比 AR 的 final-token SFT 更依赖训练目标、步数和轨迹级 credit assignment。

完整表格见 `tables/lora_control_v2.csv`，独立报告见 `reports/LoRA_Control_v2_Report.md`。

## 10. Trajectory and Error Analysis

| Task | n | Mean First Final Step | Mean Flip Count | Late Instability |
|---|---:|---:|---:|---:|
| CommonsenseQA | 200 | 7.44 | 0.715 | 0.715 |
| WinoGrande | 200 | 15.89 | 0.345 | 0.325 |

### Error Taxonomy

| Comparison | Task | Bucket | Count | Rate | Controller Route Counts |
|---|---|---|---:|---:|---|
| full_budget_vs_controller | WinoGrande | 32step_only | 3 | 0.003 | `{"8": 3}` |
| full_budget_vs_controller | WinoGrande | both_correct | 753 | 0.753 | `{"32": 222, "8": 521, "8->32": 10}` |
| full_budget_vs_controller | WinoGrande | both_wrong | 241 | 0.241 | `{"32": 116, "8": 121, "8->32": 4}` |
| full_budget_vs_controller | WinoGrande | calibrated_only | 3 | 0.003 | `{"8": 3}` |
| full_budget_vs_controller | CommonsenseQA | 32step_only | 8 | 0.008 | `{"8": 8}` |
| full_budget_vs_controller | CommonsenseQA | both_correct | 811 | 0.811 | `{"8": 809, "8->32": 2}` |
| full_budget_vs_controller | CommonsenseQA | both_wrong | 175 | 0.175 | `{"8": 175}` |
| full_budget_vs_controller | CommonsenseQA | calibrated_only | 6 | 0.006 | `{"8": 6}` |
| old_router_vs_controller | WinoGrande | both_correct | 711 | 0.711 | `{"32": 182, "8": 524, "8->32": 5}` |
| old_router_vs_controller | WinoGrande | both_wrong | 219 | 0.219 | `{"32": 92, "8": 124, "8->32": 3}` |
| old_router_vs_controller | WinoGrande | calibrated_only | 45 | 0.045 | `{"32": 40, "8->32": 5}` |
| old_router_vs_controller | WinoGrande | old_adaptive_only | 25 | 0.025 | `{"32": 24, "8->32": 1}` |
| old_router_vs_controller | CommonsenseQA | both_correct | 816 | 0.816 | `{"8": 814, "8->32": 2}` |
| old_router_vs_controller | CommonsenseQA | both_wrong | 183 | 0.183 | `{"8": 183}` |
| old_router_vs_controller | CommonsenseQA | calibrated_only | 1 | 0.001 | `{"8": 1}` |
| uncalibrated_vs_calibrated | WinoGrande | both_correct | 756 | 0.756 | `{"32": 222, "8": 524, "8->32": 10}` |
| uncalibrated_vs_calibrated | WinoGrande | both_wrong | 244 | 0.244 | `{"32": 116, "8": 124, "8->32": 4}` |
| uncalibrated_vs_calibrated | CommonsenseQA | both_correct | 814 | 0.814 | `{"8": 812, "8->32": 2}` |
| uncalibrated_vs_calibrated | CommonsenseQA | both_wrong | 179 | 0.179 | `{"8": 179}` |
| uncalibrated_vs_calibrated | CommonsenseQA | calibrated_only | 3 | 0.003 | `{"8": 3}` |
| uncalibrated_vs_calibrated | CommonsenseQA | forward_aware_only | 4 | 0.004 | `{"8": 4}` |

## 11. Limitations and Final Takeaway

- 不声称修改 LLaDA 架构；这是 inference-loop/controller 改进。
- WinoGrande 用于展示 high-risk binary semantic binding 需要保守预算控制。
- CommonsenseQA 用于展示 low-risk multi-choice 可减少不必要 fallback。
- PubMedQA/C-Eval 作为边界负例：采样预算不能弥补 label bias 或知识缺口。
- LoRA gain audit 不追求 SOTA，只验证“普通参数高效微调是否能替代 trajectory-aware controller”。
- Coverage addendum 表明方法边界更清楚：GSM8K 等长链任务强烈依赖 reverse budget，但当前 controller 尚未针对开放数值推理设计；ARC/HellaSwag/BoolQ 说明新任务上的 task-risk estimator 仍需进一步校准。

最终可讲成一句话：**我们没有把 LLaDA 改成另一个模型，而是把 masked diffusion 的反向轨迹从一次性生成过程变成可观测、可校准、可控成本的决策过程。**
