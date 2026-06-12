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
figures/inference_joint_analysis.png
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

这一页放 9 数据集主表，突出改进 LoRA 相对基础模型和旧版 LoRA 的增量：

```text
tables/paper_lora_task_main.csv
```

## 第 5 页：旧版 LoRA、TARDI-LoRA 与自回归 LoRA 的横向对比

重点数字：

```text
9 个任务宏平均：
LLaDA 基础模型：0.722
旧版 DDM LoRA：0.744，增益 +0.022
TARDI-LoRA：0.776，增益 +0.053
Qwen 基础模型：0.744
Qwen LoRA：0.784，增益 +0.040
```

这一页必须三组一起讲：

```text
旧版 DDM LoRA：说明普通扩散 LoRA 有一点收益，但不稳定。
TARDI-LoRA：说明改进训练方式后，DDM LoRA 的平均增益明显扩大。
Qwen LoRA：作为自回归 LoRA 参照，说明 TARDI-LoRA 的平均增益不弱于这次 9 任务 Qwen LoRA，但最终准确率仍略低于 Qwen LoRA。
```

建议配图：

```text
figures/ar_vs_tardi_lora_gain.png
```

建议配表：

```text
tables/paper_ar_ddm_lora_comparison.csv
```

## 第 6 页：为什么需要推理控制

讲轨迹差异：

```text
CommonsenseQA 平均 6.57 步首次到达最终答案
WinoGrande 平均 16.81 步首次到达最终答案
```

结论：

```text
不同任务的反向扩散预算需求不同，统一固定步数会造成预算错配。
```

建议配图：

```text
figures/inference_joint_analysis.png
```

## 第 7 页：选择性再掩码推理控制

流程：

1. 低预算侦察。
2. 风险估计。
3. 用样本级风险收益目标函数选择 8/16/24/32。
4. 高风险样本对低置信位置再掩码。
5. step sweep 曲线只作为分析证据，不作为默认控制器输入。

补充一句机制审计：

```text
4-step online 更细，但在 WinoGrande 上掉到 0.700，说明反复重掩码会扰动答案轨迹。
trajectory-probe cascade 和 schedule ensemble 也没有超过风险门控低置信再掩码。
主方法保留 8-step scout + 风险门控再修。
```

100 样本验证：

```text
WinoGrande：0.740 / 13.91 次调用，路由分布 8:59%, 16:24%, 24:12%, 32:5%
CommonsenseQA：0.800 / 9.92 次调用，路由分布 8:76%, 16:20%, 24:2%, 32:2%
```

建议配图：

```text
figures/controller_route_distribution.png
```

## 第 8 页：主实验结果

重点数字：

```text
WinoGrande：32 步 0.756，校准控制器 0.756，调用从 32 降到 17.56
CommonsenseQA：32 步 0.819，校准控制器 0.817，调用从 32 降到 9.06
```

建议配图：

```text
figures/controller_accuracy_cost.png
```

## 第 9 页：阈值稳健性

讲法：

```text
阈值从 0.60 到 0.80 时，准确率基本稳定，成本随保守程度变化。
```

建议配图：

```text
figures/threshold_robustness.png
```

## 第 10 页：边界与负例

重点：

```text
PubMedQA 对预算敏感，C-Eval 更像知识缺口。
```

建议配图：

```text
figures/boundary_cases.png
```

## 第 11 页：扩展任务覆盖

讲法：

```text
GSM8K 对扩散步数高度敏感；ARC、HellaSwag、BoolQ 也有预算效应。
但 LLaDA 并非全面超过自回归模型。
```

建议配图：

```text
figures/coverage_comparison.png
figures/step_sweep_by_task.png
figures/inference_joint_analysis.png
```

这一页的步数扫描已经补到 9 个任务、每任务 100 条样本、14 个步点：

```text
2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 32, 40
```

重点讲：

```text
PubMedQA：2 步到 40 步 +0.36
BoolQ：+0.14
WinoGrande：+0.05
CommonsenseQA 和 SciQ：低步数基本饱和
```

## 第 12 页：最终贡献

三点贡献：

1. TARDI-LoRA：修正固定标签推理中的训练分布和噪声阶段错位。
2. 选择性再掩码控制：把反向扩散过程变成风险可控的预算分配。
3. 边界分析：区分轨迹敏感错误、知识缺口和长链推理瓶颈。

## 第 13 页：结束页

```text
本文提出面向固定标签任务的训练对齐协议与反向扩散推理控制框架。
```

收束到两个创新点：

1. TARDI-LoRA 提升固定标签适配质量。
2. 选择性再掩码控制器提升推理预算使用效率。
