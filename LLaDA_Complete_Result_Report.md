# LLaDA Task-aware Reverse Diffusion Inference 完整结果报告

## 摘要

本项目研究 LLaDA 这类 masked diffusion language model 在下游任务上的推理行为边界。我们没有尝试重新预训练或结构级改造 LLaDA backbone，而是从工程上改进其反向扩散推理过程：先用低成本 forward label probe 和 8-step cheap scout 感知样本风险，再由 calibrated controller 决定接受 cheap decode 还是 fallback 到 32-step。

最终形成的主方法是：

> **Calibrated Forward-aware Risk Controller**：一个面向 masked diffusion language model 的任务/样本感知反向预算分配器。

核心结论：

- LLaDA 在不同任务上的反向扩散行为差异显著，不能简单用统一 8-step 或统一 32-step。
- WinoGrande 这类二选一语义绑定任务需要更保守的风险控制；过快接受 8-step 会掉点。
- CommonsenseQA 这类常识多选任务中，8-step 通常已经足够，过度 fallback 只是浪费计算。
- 在 1000 样本评测中，calibrated controller 相比 old adaptive router 在 WinoGrande 上显著更稳，McNemar paired test `p=0.0225`；在 CommonsenseQA 上保持精度并减少平均 forward calls。
- 200 样本轨迹分析显示，WinoGrande 首次进入合法 final label 的平均 step 为 `15.89`，CommonsenseQA 为 `7.44`，支持“不同任务具有不同 label hitting time”的理论解释。

这不是一个单纯加速工作，而是一个关于 diffusion inference behavior boundary 的工程化研究：哪些任务/样本需要更长的反向扩散过程，哪些不需要。

## 1. 作业背景与研究定位

作业要求关注基于 CTMC 的 discrete diffusion model、masked diffusion model，以及更大规模 diffusion language model，例如 LLaDA。直接改进 LLaDA 架构并不现实，原因包括：

- 7B/8B 级模型结构改造和预训练成本过高。
- LoRA 微调只改 final-label 格式或小规模下游数据，难以构成稳定模型架构创新。
- 如果只做 benchmark，对报告来说又不够出彩。

因此本项目选择一个更可落地、也更有研究味的方向：

> 在不改变 LLaDA 参数和 backbone 架构的前提下，研究 masked reverse diffusion 的中间轨迹和任务差异，并设计 task-aware inference controller。

这个方向的优势是：

- 与 LLaDA 的非自回归 masked diffusion 机制强相关。
- 有 CTMC、reverse process、hitting time、posterior uncertainty 等数学叙事。
- 工程上可以真实实现并验证，而不是只停留在综述。
- 可解释性强，能用错例和轨迹分析讲清楚为什么有效、在哪里失效。

## 2. 探索过程

### 2.1 第一阶段：理解 LLaDA 与 CTMC / DDM 的关系

Masked diffusion language model 的基本思想是：

- 正向过程逐渐把 token 破坏成 mask 或噪声状态。
- 反向过程学习从带 mask 的序列中恢复原始 token。
- 与 AR 模型逐 token 从左到右生成不同，LLaDA 是并行去噪，多个 token 可以同时被填回。

从 CTMC 角度看，token 状态位于离散状态空间中，正向过程由生成矩阵 `Q_t` 或其离散化版本决定。若 `x_t` 表示时刻 `t` 的 token 状态，则正向边缘分布可写作：

```math
p_t(x_t \mid x_0) = \left[\exp(tQ)\right]_{x_0, x_t}.
```

对于 masked diffusion，一个常见特例是 mask absorbing process：token 可能跳到 `[MASK]`，一旦 mask，在正向过程中保持 mask。反向过程则学习：

```math
p_\theta(x_0 \mid x_t, t).
```

这解释了为什么 LLaDA 看起来像 BERT，但比 BERT 多了一个时间维度 `t` 和一个从高噪声到低噪声的反向生成过程。

### 2.2 第二阶段：基础 benchmark 与跨域对比

我们先在多个任务上比较 LLaDA-8B、Qwen2.5-7B 和 LLaDA typed LoRA。

每任务 100 样本，typed final-label 评测结果：

| Task | Qwen2.5-7B | LLaDA-8B | LLaDA typed LoRA |
|---|---:|---:|---:|
| MMLU-Pro | 0.35 | 0.44 | 0.44 |
| PubMedQA | 0.57 | 0.72 | 0.72 |
| C-Eval CN | 0.73 | 0.56 | 0.54 |
| SciQ | 0.93 | 0.88 | 0.88 |
| WinoGrande | 0.70 | 0.78 | 0.78 |
| CommonsenseQA | 0.80 | 0.79 | 0.80 |

初步结论：

- LLaDA 有竞争力，但不是全面优于 AR。
- LoRA 对 final-label 格式帮助有限，不能作为主要创新点。
- 不同任务呈现出明显行为边界：有些任务更受采样预算影响，有些主要受知识缺口影响。

### 2.3 第三阶段：固定步数与 sampler schedule 探索

我们比较了 8-step、32-step 以及若干非均匀 schedule。

早期观察：

| Task | 8-step Acc | 32-step Acc | Observation |
|---|---:|---:|---|
| WinoGrande | 约 0.77 | 约 0.78 | 8-step 接近 32-step，但存在风险样本 |
| CommonsenseQA | 约 0.78 | 约 0.79 | 8-step 已经很强 |
| PubMedQA | 约 0.56 | 约 0.72 | 8-step 明显偏向 `maybe` |

Sampler variants 小样本探索：

| Sampler | WinoGrande | CommonsenseQA | Forward Calls |
|---|---:|---:|---:|
| uniform 8-step | 0.76 | 0.76 | 8 |
| uniform 32-step | 0.78 | 0.80 | 32 |
| front-loaded 8-step | 0.78 | 0.80 | 8 |
| back-loaded 8-step | 0.80 | 0.78 | 8 |
| middle-heavy 8-step | 0.76 | 0.82 | 8 |
| predictor-corrector 8-step | 0.76 | 0.78 | 16 |

这个阶段的结论是：

- schedule 改动确实会影响结果，但小样本上不够稳。
- predictor-corrector 增加计算但没有稳定收益。
- 仅靠 schedule 不足以构成强贡献，需要引入样本级风险感知。

### 2.4 第四阶段：从错例分析到 router

我们发现，许多错误不是“整体模型不会”，而是 cheap decode 在某些样本上过早接受了错误答案。于是设计 coarse-to-fine 流程：

```text
8-step cheap decode
  -> 读取 label/confidence/validity
  -> 高风险则 fallback 到 32-step
```

最初的 old adaptive router 很快，但 WinoGrande 上会掉点。这说明“只追求少调用”不是一个好目标，必须显式守住行为边界。

### 2.5 第五阶段：Forward-aware 与 calibrated controller

为了解决 old adaptive 的过度激进问题，我们加入 forward label probe：

```text
Question + Choices + "Final answer: [MASK]"
```

在合法 label 集合上计算后验分布：

```math
\pi_\theta(y \mid c) =
\frac{\exp z_y(c)}{\sum_{y' \in \mathcal{Y}} \exp z_{y'}(c)} ,
```

其中 `c` 是上下文，`\mathcal{Y}` 是合法标签集合，例如 `{A, B}` 或 `{A, B, C, D, E}`。

这一步提供两个风险信号：

- `probe_top_prob`: 标签后验是否集中。
- `probe_scout_agreement`: forward posterior 的 top label 是否与 8-step reverse scout 一致。

之后我们进一步发现：CommonsenseQA 上 probe-scout disagreement 经常不代表错误，过度 fallback 会浪费。因此最终做成 calibrated policy：

- binary task 上保守使用 disagreement fallback。
- multi-choice task 上忽略 disagreement，只在 invalid label 时 fallback。

这就是最终主方法。

## 3. 方法：Calibrated Forward-aware Risk Controller

### 3.1 输入与输出

输入：

- prompt / question / choices；
- task type；
- legal label set；
- LLaDA 模型；
- cheap budget `B_cheap=8`；
- full budget `B_full=32`。

输出：

- final prediction；
- route: `8`, `32`, `8->32`；
- forward call count；
- trajectory log；
- risk features。

### 3.2 Forward Label Probe

先构造只含一个 final answer mask 的 probe：

```text
{question}
{choices}
Final answer: [MASK]
```

模型在 `[MASK]` 位置输出 vocabulary logits。我们只保留 legal labels，并归一化：

```math
\pi_\theta(y \mid c) =
\frac{\exp z_y(c)}{\sum_{y' \in \mathcal{Y}} \exp z_{y'}(c)} .
```

定义：

```math
s(c) = \max_{y \in \mathcal{Y}} \pi_\theta(y \mid c),
```

```math
H(c) = - \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid c)\log \pi_\theta(y \mid c).
```

其中 `s(c)` 是 top confidence，`H(c)` 是标签后验熵。

### 3.3 Cheap Reverse Scout

然后跑 8-step reverse diffusion：

```math
\hat{y}_8 = D_\theta(c; B=8).
```

记录：

- parsed final label；
- 是否属于 legal label；
- 与 probe top label 是否一致；
- answer contract 是否满足。

### 3.4 Calibrated Route Policy

最终 route policy：

```text
if task is binary:
    run forward label probe
    if probe_top_prob < 0.70:
        use 32-step
    else:
        run 8-step scout
        if scout invalid or scout disagrees with probe:
            use 32-step
        else:
            accept 8-step
else:
    run forward label probe
    run 8-step scout
    if scout invalid:
        use 32-step
    else:
        accept 8-step
```

这个策略没有使用样本 id 或答案泄漏；它只用推理过程中自然可获得的模型信号。

### 3.5 目标函数视角

可以把 controller 看成一个成本敏感决策问题。给定特征 `\phi(x)`，选择预算 `b \in {8, 32}`，希望最小化：

```math
\mathcal{R}(\pi)
= \mathbb{E}_{x}
\left[
\ell\left(y, D_\theta(x; \pi(\phi(x)))\right)
+ \lambda C\left(\pi(\phi(x))\right)
\right],
```

其中：

- `\ell` 是分类错误损失；
- `C(b)` 是 forward calls 或 wall-clock cost；
- `\lambda` 控制准确率与成本的权衡；
- `\pi` 是 route policy。

本项目没有训练一个复杂 router，而是用可解释规则近似这个目标。这样更适合课程作业：稳定、可解释、可复现。

## 4. 数学思想

### 4.1 CTMC 与反向过程

离散扩散可以视为离散状态空间 `\mathcal{X}` 上的 CTMC。正向过程的 generator 为 `Q_t`：

```math
\frac{d p_t}{dt} = p_t Q_t.
```

若 `Q_t = Q` 固定，则：

```math
p_t = p_0 \exp(tQ).
```

反向过程理论上可以由时间反转 CTMC 给出。对状态 `x` 和 `y`，反向速率具有如下形式：

```math
\bar{q}_t(y \rightarrow x)
= q_t(x \rightarrow y)\frac{p_t(x)}{p_t(y)}.
```

真实的 `p_t(x)` 不可得，因此模型学习近似的 denoising distribution 或 score-like quantity。

### 4.2 Masked diffusion 的训练目标

Masked diffusion LM 通常通过随机采样时间 `t`，mask 一部分 token，然后训练模型恢复原 token。简化写法：

```math
\mathcal{L}(\theta)
= \mathbb{E}_{x_0, t, x_t}
\left[
- \sum_{i \in \mathcal{M}_t}
\log p_\theta(x_{0,i} \mid x_t, t)
\right].
```

这与 BERT 的 masked language modeling 类似，但多了时间变量 `t` 和从高噪声到低噪声的反向生成解释。BERT 更像固定 corruption level 的 denoising autoencoder；LLaDA 则把不同噪声水平组织成 diffusion process。

### 4.3 ELBO 与工程方法的关系

ELBO 给出的核心观点是：模型在不同噪声水平上学习近似反向条件分布。推理时，固定步数采样相当于选择一个离散化网格来近似连续反向过程。

如果某类任务在较早 step 就进入稳定答案区域，则继续采样的边际收益小；如果某类任务需要更长时间才能进入正确 label basin，则过早停止会增加错误风险。

因此我们的 controller 可以理解为在近似反向过程上做 adaptive budget allocation。

### 4.4 Hitting Time 解释

定义合法 final label 集合：

```math
\mathcal{A} = \{x: \text{parsed final label is legal}\}.
```

定义 hitting time：

```math
\tau_\mathcal{A}
= \inf\{k: x_k \in \mathcal{A}\}.
```

如果任务 A 的 `\mathbb{E}[\tau_\mathcal{A}]` 明显大于任务 B，则统一使用同样 step budget 不合理。

我们的 trace200 结果正好支持这一点：

| Task | Mean First Final Step |
|---|---:|
| WinoGrande | 15.89 |
| CommonsenseQA | 7.44 |

这说明 WinoGrande 的 label basin 更晚出现，因此更需要 conservative fallback；CommonsenseQA 更早稳定，因此可以接受 8-step。

### 4.5 Posterior Uncertainty 解释

Forward probe 估计的是标签边缘后验：

```math
\pi_\theta(y \mid c).
```

当 `\max_y \pi_\theta(y \mid c)` 低或 entropy 高时，说明模型在 final-label space 上不确定。对于 binary semantic binding task，这种不确定性与 cheap decode 风险相关，因此用作 fallback 信号。

这也是为什么我们不是拍脑袋写规则，而是用模型自身的 posterior uncertainty 做风险感知。

## 5. 实验设置

### 5.1 模型

- Diffusion LM: LLaDA-8B-Instruct。
- AR baseline: Qwen2.5-7B。
- LoRA baseline: LLaDA typed final-label LoRA。

### 5.2 任务

主要任务：

- WinoGrande：二选一语义绑定 / 代词消解。
- CommonsenseQA：五选一常识推理。
- PubMedQA：医学 yes/no/maybe。
- C-Eval CN：中文专业知识。
- SciQ：科学问答。
- MMLU-Pro：通用复杂多选。

主方法重点验证 WinoGrande 与 CommonsenseQA，因为它们代表两类不同 reverse diffusion behavior：

- WinoGrande: high-risk binary semantic binding。
- CommonsenseQA: low-risk multi-choice commonsense。

### 5.3 评测指标

- Accuracy。
- Average forward calls。
- Wall-clock seconds。
- Route rate。
- Fallback rate。
- Wilson 95% confidence interval。
- McNemar paired test。
- Trajectory first-final-step。

## 6. 结果

### 6.1 跨任务基线结果

| Task | Qwen2.5-7B | LLaDA-8B | LLaDA typed LoRA |
|---|---:|---:|---:|
| MMLU-Pro | 0.35 | 0.44 | 0.44 |
| PubMedQA | 0.57 | 0.72 | 0.72 |
| C-Eval CN | 0.73 | 0.56 | 0.54 |
| SciQ | 0.93 | 0.88 | 0.88 |
| WinoGrande | 0.70 | 0.78 | 0.78 |
| CommonsenseQA | 0.80 | 0.79 | 0.80 |

结论：

- LLaDA 在 PubMedQA、WinoGrande 上表现强。
- Qwen 在 C-Eval、SciQ 上更强。
- LoRA 没有稳定提升，说明微调不是本项目最优方向。

### 6.2 500 样本 held-out 结果

| Method | WinoGrande Acc | Wino Calls | CommonsenseQA Acc | CQA Calls |
|---|---:|---:|---:|---:|
| 32-step baseline | 0.756 | 32.000 | 0.810 | 32.000 |
| Old adaptive | 0.732 | 9.462 | 0.810 | 9.924 |
| Forward-aware | 0.756 | 17.768 | 0.812 | 10.856 |
| Calibrated controller | 0.756 | 17.768 | 0.812 | 9.000 |

解释：

- Old adaptive 很快，但 WinoGrande 掉点。
- Forward-aware 修复 WinoGrande 掉点。
- Calibrated controller 在 CommonsenseQA 上去掉不必要 fallback，精度不变但调用更低。

### 6.3 1000 样本主结果

| Method | Task | Acc | Wilson 95% CI | Avg Calls | Route / Fallback |
|---|---|---:|---:|---:|---|
| Calibrated | WinoGrande | 0.756 | [0.728, 0.782] | 17.560 | 32: 33.8%, 8: 64.8%, 8->32: 1.4% |
| Old adaptive | WinoGrande | 0.736 | [0.708, 0.762] | 9.363 | fallback: 1.1% |
| Calibrated | CommonsenseQA | 0.817 | [0.792, 0.840] | 9.064 | 8: 99.8%, 8->32: 0.2% |
| Old adaptive | CommonsenseQA | 0.816 | [0.791, 0.839] | 9.858 | fallback: 2.6% |

McNemar paired comparison：

| Task | Both Correct | Old Only | Calibrated Only | Neither | Net | p-value |
|---|---:|---:|---:|---:|---:|---:|
| WinoGrande | 711 | 25 | 45 | 219 | +20 | 0.0225 |
| CommonsenseQA | 816 | 0 | 1 | 183 | +1 | 1.0000 |

主结论：

- WinoGrande: calibrated controller 相比 old adaptive 多修复 20 个样本，paired test 达到显著水平。
- CommonsenseQA: calibrated controller 与 old adaptive 精度基本持平，但平均 calls 从 `9.858` 降到 `9.064`。
- 这不是单纯少算，而是根据任务风险重新分配预算。

### 6.4 200 样本轨迹分析

| Task | Trace N | Accuracy | Avg Calls | Mean First Final Step | Mean Flip Count | Late Instability |
|---|---:|---:|---:|---:|---:|---:|
| WinoGrande | 200 | 0.750 | 17.040 | 15.89 | 0.015 | 0.000 |
| CommonsenseQA | 200 | 0.815 | 9.000 | 7.44 | 0.030 | 0.030 |

这个结果非常关键：

- WinoGrande 的 final label 平均更晚出现，因此 cheap 8-step 更容易错过必要的语义绑定过程。
- CommonsenseQA 的 final label 平均更早出现，因此 8-step 已经足够。
- 这支持“不同任务具有不同 reverse diffusion hitting time”这一理论解释。

## 7. 错例驱动分析

### 7.1 WinoGrande 的行为边界

WinoGrande 是二选一任务，但并不简单。它要求模型根据句子语义判断代词或空缺指向哪一个候选项。

错误模式：

- 8-step scout 过早锁定局部语义。
- forward probe 置信度低或与 scout 不一致。
- old adaptive 因为过度追求低调用，接受了部分高风险 cheap decode。

Calibrated controller 的作用：

- 对 binary task 使用 probe confidence 和 probe-scout agreement。
- 对高风险样本 fallback 到 32-step。
- 因此在 1000 样本上相比 old adaptive 多对 20 个样本。

### 7.2 CommonsenseQA 的行为边界

CommonsenseQA 是五选一常识推理。实验发现，很多样本在 8-step 已经进入正确 label basin。

错误模式：

- 错误主要来自常识/语义本身，而不是采样预算不足。
- probe-scout disagreement 在该任务上不一定代表错误。

Calibrated controller 的作用：

- 不再对 multi-choice disagreement 过度 fallback。
- 只在 invalid label 时 fallback。
- 精度基本不变，调用数更低。

### 7.3 PubMedQA 与 C-Eval 的边界

PubMedQA：

- 8-step 明显偏向 `maybe`。
- 更像 label prior / uncertainty calibration 问题。
- 不适合简单接受 cheap decode，应默认高预算或做专门校准。

C-Eval：

- 错误更多来自领域知识缺口。
- 增加 diffusion steps 无法补知识。
- 后续更适合结合 RAG 或领域提示，而不是继续调 sampler。

这两个负结果很重要：它们说明我们的 controller 不是万能技巧，而是针对 sampling-budget-sensitive tasks 最有效。

## 8. 稳健性讨论

### 8.1 样本量

本项目经历了从 100 到 500 再到 1000 的扩展：

- 100 样本用于探索任务差异和 sampler 候选。
- 500 样本用于 held-out 验证。
- 1000 样本用于主结论稳健性。
- 200 样本用于轨迹记录，因为 trajectory logging 计算成本更高。

因此主结论不是只基于 toy sample。

### 8.2 成对检验

我们不仅比较 accuracy，还做 paired comparison。WinoGrande 上：

```text
old_only = 25
calibrated_only = 45
net = +20
p = 0.0225
```

这说明 calibrated controller 的提升不是只来自随机波动。

### 8.3 负结果保留

我们保留了几个负结果：

- Typed LoRA 没有稳定提升。
- Predictor-corrector sampler 成本更高但收益不足。
- 16-step intermediate route 没有带来稳定收益。
- PubMedQA / C-Eval 不能靠简单预算控制解决。

这些负结果让报告更可信：我们不是只挑有利结果讲。

### 8.4 不过拟合说明

当前 controller 没有使用：

- 样本 id。
- gold label。
- 数据集特定答案词泄漏。
- 人工 hard-coded 错例列表。

使用的信号都是推理时自然可得的：

- legal label posterior。
- scout prediction。
- output validity。
- task type。

因此它是可迁移的推理策略，而不是对测试集记忆。

## 9. 工程产物

本地报告与计划：

```text
/Users/thomaswang/Documents/New project/LLaDA_Research_Grade_Execution_Plan.md
/Users/thomaswang/Documents/New project/LLaDA_Calibrated_Controller_Report.md
/Users/thomaswang/Documents/New project/LLaDA_Error_Trajectory_Analysis_Plan.md
/Users/thomaswang/Documents/New project/LLaDA_Complete_Result_Report.md
```

本地脚本：

```text
/Users/thomaswang/Documents/New project/scripts/eval_llada_adaptive_router.py
/Users/thomaswang/Documents/New project/scripts/eval_llada_forward_aware_router.py
/Users/thomaswang/Documents/New project/scripts/eval_llada_budget_controller.py
/Users/thomaswang/Documents/New project/scripts/analyze_llada_error_trajectory.py
```

远程核心结果：

```text
/data/llada_eval/results/domain_shift/task_aware/llada8b_calibrated_controller_wino_cqa_limit1000_seed23.json
/data/llada_eval/results/domain_shift/task_aware/llada8b_adaptive_router_wino_cqa_limit1000_seed23.json
/data/llada_eval/results/domain_shift/task_aware/llada8b_calibrated_controller_trace_wino_cqa_limit200_seed23.json
/data/llada_eval/results/domain_shift/task_aware/error_trajectory_analysis_trace200/
```

分析图表：

```text
route_distribution.png
confidence_accuracy.png
accuracy_cost_pareto.png
trajectory_stabilization.png
error_taxonomy.csv
trajectory_metrics.csv
analysis_summary.json
```

## 10. 可复现实验命令

### 10.1 Calibrated controller 1000 样本

```bash
python scripts/eval_llada_budget_controller.py \
  --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct \
  --tasks winogrande,commonsenseqa \
  --limit 1000 \
  --seed 23 \
  --binary-medium-threshold 0.70 \
  --multi-disagreement-policy ignore \
  --out results/domain_shift/task_aware/llada8b_calibrated_controller_wino_cqa_limit1000_seed23.json
```

### 10.2 Old adaptive router 1000 样本

```bash
python scripts/eval_llada_adaptive_router.py \
  --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct \
  --tasks winogrande,commonsenseqa \
  --limit 1000 \
  --seed 23 \
  --out results/domain_shift/task_aware/llada8b_adaptive_router_wino_cqa_limit1000_seed23.json
```

### 10.3 Trace200

```bash
python scripts/eval_llada_budget_controller.py \
  --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct \
  --tasks winogrande,commonsenseqa \
  --limit 200 \
  --seed 23 \
  --binary-medium-threshold 0.70 \
  --multi-disagreement-policy ignore \
  --collect-trace \
  --trace-stride 2 \
  --out results/domain_shift/task_aware/llada8b_calibrated_controller_trace_wino_cqa_limit200_seed23.json
```

### 10.4 轨迹与错例分析

```bash
python scripts/analyze_llada_error_trajectory.py \
  --baseline32 results/domain_shift/llada8b_final_label_typed_wino_cqa_limit500_seed23.json \
  --old-adaptive results/domain_shift/task_aware/llada8b_adaptive_router_wino_cqa_limit500_seed23.json \
  --forward results/domain_shift/task_aware/llada8b_forward_aware_wino_cqa_limit500_seed23.json \
  --calibrated results/domain_shift/task_aware/llada8b_calibrated_controller_wino_cqa_limit500_seed23.json \
  --trace results/domain_shift/task_aware/llada8b_calibrated_controller_trace_wino_cqa_limit200_seed23.json \
  --out-dir results/domain_shift/task_aware/error_trajectory_analysis_trace200
```

## 11. 如何在最终作业里讲故事

推荐主线：

1. LLaDA 不是 AR，它的推理是 masked reverse diffusion，因此中间轨迹可观测。
2. 固定 32-step 是保守但浪费，固定 8-step 是快但会越过行为边界。
3. 我们先用 benchmark 和错例分析发现任务差异。
4. 然后把错例分析工程化为 forward-aware risk controller。
5. 最后用 1000 样本和 trajectory hitting time 证明这个 controller 不是 toy trick。

一句话总结：

> 我们把 masked diffusion LM 的反向生成过程从一个固定采样器，变成了一个可观测、可解释、可校准的任务感知推理控制问题。

## 12. 局限与未来工作

局限：

- 没有改 LLaDA backbone，因此不能声称架构创新。
- 当前主要验证在 final-label classification / multiple-choice 格式。
- 对领域知识缺口类任务，推理预算控制帮助有限。
- 当前 controller 是规则式，还没有训练可泛化的 learned risk predictor。

未来工作：

- 用更多任务训练轻量 risk predictor，例如 logistic regression 或 decision tree。
- 扩展到开放式生成任务，记录 span-level 或 token-level stabilization。
- 将 controller 与 RAG 结合：当判断为知识缺口时触发检索，而不是增加 diffusion steps。
- 探索更理论化的 stopping rule，例如基于 posterior entropy 或 label hitting time 的置信界。

## 13. 最终结论

本项目的贡献不是“让 LLaDA 更快”这么简单，而是提出并验证了一个更细的观点：

> 对 masked diffusion language model 来说，推理计算量应该按任务和样本风险分配。不同任务进入稳定答案区域的时间不同，统一采样预算会同时造成浪费和风险。

在工程上，我们实现了 forward probe、cheap scout、calibrated route policy 和 trajectory logger；在实验上，我们完成了从 100 样本探索到 1000 样本验证；在理论上，我们用 CTMC reverse process、posterior uncertainty 和 hitting time 解释方法为何合理。

因此，这个工作可以作为课程大作业中“理论基础 + 工程创新 + 实证验证”三者结合的主线。
