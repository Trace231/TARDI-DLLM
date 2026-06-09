# Choice-aware Noise-aware LoRA + Selective Re-masking Controller

## 1. 研究定位

本项目不把重点放在重新训练一个新的 diffusion language model，也不声称修改了 LLaDA 的 backbone 架构。核心问题是：

> masked diffusion language model 在选择题、判断题、医学问答、中文知识题等 fixed-label reasoning 任务上，训练目标和推理预算都可能与最终决策目标错位。

因此我们把方法组织成两个互补模块：

1. **训练侧适配**：Choice-aware Noise-aware LoRA，让 LoRA 梯度直接作用到合法选项 posterior 和不同噪声阶段的去噪行为上。
2. **推理侧控制**：Selective Re-masking Controller，根据早期轨迹风险动态分配反向扩散预算，并只对低置信位置追加去噪。

这两个模块服务于同一个目标：让 LLaDA 更适合 fixed-label reasoning，而不是泛泛地做 LoRA 或泛泛地减少采样步数。

## 2. 两个 mismatch

### 2.1 Adaptation mismatch

普通 AR 模型做选择题 LoRA 时，训练信号通常直接落在 answer token 或 final-label token 上：

$$
\mathcal{L}_{AR}
= -\log p_{\theta}(y \mid x).
$$

但 LLaDA 的训练和推理是 masked denoising family。若直接把普通 SFT/LoRA 套到 LLaDA 上，梯度容易被完整 completion 的去噪损失摊薄，最终不一定有效改变合法标签空间上的边界：

$$
p_{\theta}(y \mid x, \text{Final answer: [MASK]}).
$$

这就是我们前面 LoRA audit 中观察到的现象：vanilla DDM LoRA 不一定能稳定转化为 final-label accuracy gain。

### 2.2 Inference mismatch

固定 32 步采样对所有样本给同样 reverse budget，但不同任务和不同样本的答案稳定时间不同。WinoGrande 这类语义绑定任务往往需要更长轨迹，CommonsenseQA、SciQ、ARC 的一部分样本则可能早期已经稳定。

因此固定步数带来两个问题：

1. 简单样本浪费计算。
2. 困难样本需要局部修复，而不是简单全量重跑。

## 3. Choice-aware Noise-aware LoRA

### 3.1 Label-space posterior objective

对每个 fixed-label 样本，我们构造统一格式：

```text
Question + Choices
Final answer: [LABEL]
```

设合法标签空间为：

$$
\mathcal{Y}(x)=\{A,B,C,D,\ldots\}
$$

或二/三分类标签：

$$
\mathcal{Y}(x)=\{\text{yes},\text{no}\},\quad
\{\text{yes},\text{no},\text{maybe}\}.
$$

我们不只优化完整 target 的 token-level denoising，而是显式优化 final-label 位置在合法选项集合上的 posterior：

$$
\mathcal{L}_{choice}
=
-\log
\frac{\exp z_{y}}
{\sum_{y'\in\mathcal{Y}(x)} \exp z_{y'}}.
$$

这里 \(z_y\) 是 final-label mask 位置上标签 \(y\) 的 logit。这个目标让 LoRA 更新直接改变选择题决策边界。

### 3.2 Noise-aware denoising objective

为了不把 LLaDA 退化成一个只会填最终标签的分类器，我们同时在多个 mask ratio 下训练：

```text
noise ratios = 0.15, 0.35, 0.65, 0.85
```

对应损失：

$$
\mathcal{L}_{denoise}
=
-\sum_{i\in M_t}
\log p_{\theta}(x_i \mid x_{\bar M_t}, t).
$$

它保留 diffusion LM 的去噪结构，使 adapter 在早期高噪声和后期低噪声状态下都能被监督。

### 3.3 Cross-noise consistency

同一个样本在不同 mask ratio 下，合法标签 posterior 不应该剧烈变化。我们加入一个轻量 consistency term：

$$
\mathcal{L}_{cons}
=
\mathrm{KL}
\left(
p_{\theta}(\mathcal{Y}\mid x,t_a)
\;\|\;
p_{\theta}(\mathcal{Y}\mid x,t_b)
\right).
$$

总目标为：

$$
\mathcal{L}
=
\mathcal{L}_{choice}
+
\lambda \mathcal{L}_{denoise}
+
\gamma \mathcal{L}_{cons}.
$$

当前实现中：

```text
lambda = 0.15
gamma  = 0.05
LoRA rank = 8
LoRA alpha = 16
dropout = 0.05
```

### 3.4 Ablation 设计

为了证明不是简单“多训了一下”，实验保留三个 adapter：

| Adapter | Objective | 作用 |
|---|---|---|
| Vanilla LoRA | denoising only | 普通 DDM LoRA 基线 |
| Label-focused LoRA | choice posterior + denoise | 检验 final-label objective mismatch |
| Choice-noise LoRA | choice + denoise + cross-noise consistency | 检验 noise/time awareness 是否有额外价值 |

如果结果满足：

```text
Choice-noise LoRA >= Label-focused LoRA >= Vanilla LoRA
```

并且在 held-out 任务上不崩，就可以支撑“fixed-label adaptation 需要 choice posterior 和 diffusion noise stage 同时对齐”的结论。

## 4. Selective Re-masking Controller

适配后的模型仍然会面对推理预算错配。我们把反向扩散步数看成一个 cost-sensitive decision：

$$
K^*(x)
=
\arg\min_{K\in\{8,16,24,32\}}
\left[
\mathbb{E}\ell(y,\hat y_K\mid x)
+ \lambda C(K)
\right].
$$

控制器使用以下信号估计风险：

1. forward label probe 的 top probability、margin、entropy；
2. 8-step scout 的预测标签、合法性、置信度；
3. 轨迹中的 label flip、late stabilization、低置信 token；
4. 任务类型和标签空间大小。

根据风险分数 \(r(x)\)，动态选择：

$$
K(x)\in\{8,16,24,32\}.
$$

当需要追加预算时，控制器不是直接从全 mask 重跑，而是对当前序列中低置信位置重新 mask，再继续 denoising：

$$
x^{refine}_{i}
=
\begin{cases}
\text{[MASK]}, & i\in U(x),\\
x_i, & i\notin U(x),
\end{cases}
$$

其中 \(U(x)\) 是由 token confidence 和风险分数确定的局部不稳定位置集合。

## 5. 为什么这个组合比单独模块更强

单独做 LoRA 的问题是：即使 final-label posterior 更准，仍然可能在困难样本上需要更多 reverse steps。

单独做 controller 的问题是：它只能修复采样轨迹敏感型错误，无法弥补 final-label decision boundary 没有适配的问题。

组合后的主张是：

> Choice-aware noise-aware LoRA 提升 fixed-label posterior 的质量；selective re-masking controller 在此基础上把 reverse computation 分配给仍然不稳定的样本和位置，从而得到更高质量、更高效率的 DDM 选择题推理系统。

## 6. 当前实验矩阵

主实验使用同一批 held-out 样本和同一评测格式：

```text
tasks = mmlu_pro, pubmedqa, ceval_computer_network, sciq,
        winogrande, commonsenseqa, arc_challenge, hellaswag, boolq
limit = 50
seed = 23
```

对比方法：

| Method | 目的 |
|---|---|
| LLaDA base fixed-32 | 满预算基础线 |
| LLaDA vanilla LoRA fixed-32 | 普通 DDM LoRA |
| LLaDA label-focused LoRA fixed-32 | choice posterior 对齐 |
| LLaDA choice-noise LoRA fixed-32 | choice + noise/time 对齐 |
| LLaDA choice-noise LoRA + controller | 质量适配后的预算控制 |
| Existing sampler baselines | JYS-like、Prophet-like、fixed 8/16/32 等相关工作对照 |
| Qwen AR / Qwen AR LoRA | 判断 AR LoRA gain 与 DDM LoRA gain 差异 |

## 7. 预期结论写法

如果 choice-noise LoRA 有正增益：

> Vanilla DDM LoRA does not reliably transfer final-label supervision into accuracy gains, while choice-aware noise-aware LoRA improves the legal-label posterior by aligning adaptation with both the fixed-label decision space and diffusion noise stages.

如果 controller 在 adapter 上继续省算：

> Even after task adaptation, samples remain heterogeneous in reverse diffusion difficulty. Selective re-masking preserves most of the adapted model's full-budget behavior while reducing average denoising calls.

如果某些任务不提升：

> The method is not a knowledge injection mechanism. On knowledge-limited or calibration-sensitive tasks such as PubMedQA and C-Eval, adaptation and controller gains are bounded by the base model's domain knowledge and label prior.

## 8. 最终 paper 叙事

建议标题：

```text
Making Diffusion Language Models Work for Fixed-label Reasoning:
Choice-aware Adaptation and Risk-aware Reverse Inference
```

三条贡献：

1. 诊断 vanilla DDM LoRA 在 fixed-label reasoning 上的 objective mismatch。
2. 提出 Choice-aware Noise-aware LoRA，用合法标签 posterior 和跨噪声一致性适配 LLaDA。
3. 提出 Selective Re-masking Controller，在适配后继续做样本级预算分配和局部低置信修复。

这个叙事比“LoRA + 加速器”更稳，因为它们共同回答同一个问题：如何让 masked diffusion LM 在 fixed-label downstream reasoning 中同时更准、更省。
