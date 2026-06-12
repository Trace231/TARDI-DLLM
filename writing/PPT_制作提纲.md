# PPT 制作提纲

## 第 1 页：题目与一句话主张

题目：

```text
面向固定标签推理的掩码扩散语言模型适配与反向推理控制
```

一句话主张：

```text
本文从训练侧对齐固定标签去噪目标，并从推理侧控制反向扩散预算。
```

## 第 2 页：问题动机

讲清楚两个错位：

1. 普通 LoRA 的训练分布和固定标签评测分布不一致。
2. 固定步数采样无法适应不同样本的反向去噪难度。

建议配图：

```text
figures/trajectory_metrics.png
```

## 第 3 页：训练侧方法

方法名：

```text
TARDI-LoRA
```

核心三点：

1. 任务均衡训练集。
2. 评测样本去重。
3. 高噪声终标签去噪。

建议配图：

```text
figures/lora_macro_comparison.png
```

## 第 4 页：训练侧结果

重点数字：

```text
基础模型：0.722
旧版 LoRA：0.744
TARDI-LoRA：0.776
```

建议配图：

```text
figures/lora_task_heatmap.png
```

任务条件化 LoRA 作为补充消融即可：我们试过更复杂结构，但没有超过 TARDI-LoRA。

## 第 5 页：为什么需要推理控制

讲轨迹差异：

```text
CommonsenseQA 平均 7.44 步首次到达最终答案
WinoGrande 平均 15.89 步首次到达最终答案
```

结论：

```text
不同任务的反向扩散预算需求不同，统一固定步数会造成预算错配。
```

建议配图：

```text
figures/trajectory_metrics.png
```

## 第 6 页：选择性再掩码推理控制

流程：

1. 低预算侦察。
2. 风险估计。
3. 低风险直接接受。
4. 高风险对低置信位置再掩码并追加预算。

建议配图：

```text
figures/controller_route_distribution.png
```

## 第 7 页：主实验结果

重点数字：

```text
WinoGrande：32 步 0.756，校准控制器 0.756，调用从 32 降到 17.56
CommonsenseQA：32 步 0.819，校准控制器 0.817，调用从 32 降到 9.06
```

建议配图：

```text
figures/controller_accuracy_cost.png
```

## 第 8 页：阈值稳健性

讲法：

```text
阈值从 0.60 到 0.80 时，准确率基本稳定，成本随保守程度变化。
```

建议配图：

```text
figures/threshold_robustness.png
```

## 第 9 页：边界与负例

重点：

```text
PubMedQA 对预算敏感，C-Eval 更像知识缺口。
```

建议配图：

```text
figures/boundary_cases.png
```

## 第 10 页：扩展任务覆盖

讲法：

```text
GSM8K 对扩散步数高度敏感；ARC、HellaSwag、BoolQ 也有预算效应。
但 LLaDA 并非全面超过自回归模型。
```

建议配图：

```text
figures/coverage_comparison.png
figures/step_sweep_by_task.png
```

## 第 11 页：最终贡献

三点贡献：

1. TARDI-LoRA：修正固定标签推理中的训练分布和噪声阶段错位。
2. 选择性再掩码控制：把反向扩散过程变成风险可控的预算分配。
3. 边界分析：区分轨迹敏感错误、知识缺口和长链推理瓶颈。

## 第 12 页：汇报边界

容易被误解的说法：

1. 把工作讲成修改 LLaDA 主体结构。
2. 把任务条件化 LoRA 讲成最终主贡献。
3. 把自适应采样本身讲成全新方法。
4. 暗示方法在所有任务上都提升。

推荐说法：

```text
本文提出面向固定标签任务的训练对齐协议与反向扩散推理控制框架。
```
