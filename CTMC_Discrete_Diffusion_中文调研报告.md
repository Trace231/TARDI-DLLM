# 基于 CTMC 的 Discrete Diffusion Model 与 Masked Diffusion Model 调研报告

> 版本：2026-05-25 调研草稿  
> 适用场景：课程小组研究报告，可继续扩展实验、图表和复现结果。

## 摘要

离散扩散模型（Discrete Diffusion Model, DDM）把连续扩散模型的“逐步加噪、学习反向去噪”思想迁移到离散数据，例如文本 token、类别变量、图结构和离散图像像素。早期方法多采用离散时间马尔可夫链，例如 Multinomial Diffusion 和 D3PM；后续工作将前向加噪过程统一为连续时间马尔可夫链（Continuous-Time Markov Chain, CTMC），使得离散状态空间上的扩散可以借用生成元、反向时间过程、score/ratio matching 等连续时间工具。

Masked Diffusion Model 是离散扩散在语言建模中最重要的一类实例。它通常把 `[MASK]` 作为吸收态，前向过程随机把原 token 替换为 mask，反向过程则并行或半并行地预测被 mask 的 token。近年的 MDLM、SEDD、LLaDA 等模型表明，非自回归或半自回归的扩散语言模型可以在生成可控性、并行采样、填充/编辑任务上形成与自回归语言模型不同的优势，但在长文本一致性、采样步数、似然估计和大规模训练稳定性方面仍有挑战。

## 1. 研究背景

连续数据上的 DDPM/SDE 扩散模型使用高斯噪声逐渐破坏数据，并训练神经网络学习反向去噪。但文本、类别标签、分子图等数据没有自然的“加高斯噪声”操作。离散扩散模型的核心问题是：如何在有限或可数状态空间上定义合理的前向破坏过程，并学习其反向生成过程。

现有离散扩散大致可分为三类：

1. 离散时间转移矩阵方法：每一步用矩阵 \(Q_t\) 把 token 转移到其他 token 或 mask，例如 D3PM。
2. CTMC 连续时间方法：用生成元/rate matrix \(R_t\) 定义 token 跳转率，例如 Campbell 等人的 continuous-time discrete denoising framework。
3. Masked/absorbing diffusion 方法：前向只允许 clean token 逐渐变成 `[MASK]`，反向学习从 mask 中恢复原 token，例如 MDLM、LLaDA。

## 2. 基于 CTMC 的 Discrete Diffusion Model 基本原理

### 2.1 状态空间与前向 CTMC

设离散数据 \(x \in \mathcal X\)，文本场景下 \(\mathcal X=\{1,\dots,K\}^L\)，其中 \(K\) 是词表大小，\(L\) 是序列长度。CTMC 用一个随时间变化的生成元 \(R_t\) 定义从状态 \(x\) 跳到状态 \(y\) 的瞬时速率：

\[
R_t(x,y) \ge 0,\quad x\ne y,\qquad R_t(x,x)=-\sum_{y\ne x}R_t(x,y).
\]

前向分布满足 Kolmogorov forward equation：

\[
\frac{d}{dt}q_t = q_t R_t.
\]

若 \(R_t\) 是时间非齐次的，则从 \(0\) 到 \(t\) 的转移矩阵可写为时间有序指数：

\[
P_{0,t}=\mathcal T \exp\left(\int_0^t R_s ds\right).
\]

在语言模型中通常假设各 token 位置独立加噪，因此序列级转移可以分解为逐位置转移，降低计算复杂度。

### 2.2 常见前向加噪设计

常见 CTMC 前向过程包括：

1. Uniform corruption：token 以某个 rate 跳到词表中其他 token。最终分布接近均匀分布。
2. Mask/absorbing corruption：token 只会跳到 `[MASK]`，`[MASK]` 是吸收态。最终分布接近全 mask。
3. Structured corruption：根据词汇相似性、图邻接关系、离散图像像素距离等设计非均匀跳转率。

吸收态 mask 过程尤其适合文本，因为 `[MASK]` 与 BERT 式条件预测任务天然兼容。若单个 token 在 \(t\) 时刻保留 clean 的概率为 \(\alpha_t\)，则前向边缘分布常写为：

\[
q(x_t \mid x_0)=
\begin{cases}
\alpha_t, & x_t=x_0,\\
1-\alpha_t, & x_t=\texttt{[MASK]},\\
0, & \text{otherwise}.
\end{cases}
\]

其中 \(\alpha_t\) 随时间单调下降。

### 2.3 反向 CTMC 与 density ratio/score

生成时要从噪声分布 \(q_T\) 反向采样到数据分布。CTMC 的反向时间过程仍是 CTMC，其反向跳转率与前向跳转率以及边缘分布比值有关。概念上，若前向在 \(t\) 时刻允许 \(y\to x\)，则反向中 \(x\to y\) 的速率与

\[
R_t(y,x)\frac{q_t(y)}{q_t(x)}
\]

成正比。这里的 \(\frac{q_t(y)}{q_t(x)}\) 就是离散状态空间上的 density ratio，也常被称为离散 score 的核心对象。

因此，CTMC 离散扩散模型通常训练神经网络学习以下目标之一：

1. 预测 clean data：学习 \(p_\theta(x_0 \mid x_t,t)\)，再由 Bayes 公式构造反向转移。
2. 预测 density ratio/score：学习 \(s_\theta(x,t,y)\approx q_t(y)/q_t(x)\)，直接参数化反向 CTMC。
3. 预测 mask token 分布：在 absorbing/masked diffusion 中，学习每个 masked 位置的 token 条件分布。

### 2.4 训练目标

离散扩散训练目标通常来自变分下界（ELBO）、去噪 score matching 或交叉熵。D3PM 使用离散时间 ELBO，并引入辅助 cross entropy loss 提升训练效果。CTMC 框架则可以将离散时间极限推广到连续时间：前向跳转在无穷小时间内发生，训练目标可写成对时间 \(t\)、当前噪声状态 \(x_t\) 和可能跳转状态的期望。

对 masked diffusion，训练形式通常非常接近 BERT 的 masked language modeling：随机采样时间 \(t\)，按 mask 概率遮蔽 token，然后用交叉熵预测原 token。区别在于 diffusion 模型把不同 mask 比例组织成一个连续或离散时间生成过程，并在采样时从高 mask 比例逐步去 mask。

### 2.5 采样

CTMC 采样有两类常见方式：

1. Exact/Gillespie-style jump simulation：根据总跳转率抽样下一次跳转时间和跳转目标。理论自然，但大词表文本中计算较重。
2. Tau-leaping / discretized reverse process：把时间区间分成若干步，在每一步并行更新若干 token。实际大模型通常采用这种近似，以换取速度。

对 masked diffusion，采样经常采用“逐步解 mask”策略：从全 mask 序列开始，模型预测所有 mask 位置的 token 分布，根据置信度或预设 schedule 固定一部分 token，剩余位置继续 mask，直到所有位置被填充。

## 3. Masked Diffusion Model 基本原理

Masked Diffusion Model 可以看作 absorbing-state discrete diffusion 的语言建模特例。

### 3.1 前向过程

给定文本 \(x_0=(x_0^1,\dots,x_0^L)\)，对每个位置独立采样是否 mask：

\[
x_t^i =
\begin{cases}
x_0^i, & \text{with probability } \alpha_t,\\
\texttt{[MASK]}, & \text{with probability } 1-\alpha_t.
\end{cases}
\]

随着 \(t\) 增大，\(\alpha_t\) 下降，序列逐渐变成全 mask。这个过程有三个好处：

1. 语义明确：mask 表示未知 token，而不是随机错误 token。
2. 训练稳定：目标可直接使用 token-level cross entropy。
3. 并行友好：每一步可以同时预测多个 mask 位置。

### 3.2 反向过程

反向过程从高噪声状态开始，例如全 mask。模型 \(p_\theta(x_0^i\mid x_t,t)\) 对每个 mask 位置给出 token 分布。采样时可选择：

1. 一次性填充：速度快，但质量较差。
2. 固定步数迭代填充：每一步填充一部分位置。
3. 置信度驱动填充：优先固定模型最有把握的位置。
4. 半自回归/block-wise 填充：每次生成一个块，块内使用扩散式并行去噪。

### 3.3 与 BERT 和自回归 LM 的关系

Masked diffusion 与 BERT 的相似点是二者都训练 masked token prediction；不同点是 BERT 通常不是一个完整生成模型，而 masked diffusion 规定了从全 mask 到 clean text 的反向采样轨迹。

与自回归 LM 相比，masked diffusion 的特点是：

1. 生成顺序更灵活：不必严格从左到右。
2. 更适合 infilling/editing：可在任意位置保持已知 token，只对未知位置去噪。
3. 有潜在并行优势：单步可更新多个 token。
4. 采样质量依赖迭代调度：步数过少可能不连贯，步数过多会牺牲速度。

## 4. 代表性文献脉络

### 4.1 Multinomial Diffusion 与 D3PM

Hoogeboom 等人的 multinomial diffusion 将扩散模型推广到 categorical data，用多项分布转移替代高斯噪声。Austin 等人的 D3PM 系统研究了离散状态空间中的转移矩阵设计，包括 uniform、absorbing 和 discretized Gaussian-like transitions，并展示了文本和离散图像任务上的效果。

D3PM 的重要贡献是证明离散扩散不必局限于简单随机替换，转移矩阵本身可以编码数据结构。其不足是离散时间步数、转移矩阵和 ELBO 项的设计较复杂，对大规模语言建模还不是最简洁的训练范式。

### 4.2 Continuous-Time Discrete Denoising

Campbell、Benton、De Bortoli 等人的工作将离散 denoising diffusion 置于 CTMC 框架下，使用 rate matrix 和反向时间 CTMC 统一描述离散数据生成。这一框架的重要意义在于：

1. 连续时间下可以更自然地处理不同噪声 schedule。
2. 反向过程的理论形式清楚依赖 density ratio。
3. 离散时间扩散可以看作 CTMC 的数值离散化。
4. 可以与 Gillespie simulation、tau-leaping 等 CTMC 采样技术结合。

### 4.3 SEDD：Score Entropy Discrete Diffusion

SEDD 强调直接学习离散数据分布的 score/density ratio，并提出 score entropy 训练目标。它把“连续空间中的 score”替换为离散空间中相邻状态概率比的估计，从而避免只依赖 clean-token posterior 的间接参数化。SEDD 在语言建模上展示了 diffusion LM 的可行性，但大词表和长序列下的采样效率仍是重要问题。

### 4.4 MDLM：Masked Diffusion Language Models

MDLM 将 masked diffusion 语言模型简化到一个实用框架：前向使用 mask corruption，训练目标接近加权 masked token cross entropy，采样从全 mask 逐步恢复文本。MDLM 的关键价值在于把理论扩散目标、BERT 式 MLM 训练和实际文本生成采样连接起来，使 masked diffusion 语言模型更容易复现和扩展。

### 4.5 LLaDA：Large Language Diffusion Models

LLaDA 是大规模 diffusion language model 的代表方向。它试图证明扩散式语言模型也能扩展到 instruction following、对话、推理等大语言模型任务。LLaDA 通常采用 mask-based diffusion 训练和迭代式生成：模型不是按左到右逐 token 预测，而是在多个 denoising step 中逐渐确定整段文本。

从研究角度看，LLaDA 的意义不只是“把 diffusion LM 做大”，还在于挑战了“大语言模型必须自回归”的默认假设。但与成熟自回归 LLM 相比，diffusion LLM 仍需解决采样延迟、长上下文一致性、对齐训练、工具调用和部署生态等问题。

### 4.6 Scaling up / Scaling beyond Masked Diffusion

2024-2026 年的一条新线索是系统研究 diffusion language model 的 scaling behavior。Scaling up Masked Diffusion Models on Text 关注 masked diffusion 在更大模型和更多数据上的表现；Scaling Beyond Masked Diffusion Language Models 则进一步比较 masked、uniform-state 和二者插值的离散扩散。它们提醒我们：perplexity 在同一模型族内部有参考价值，但跨噪声类型或跨算法比较时，采样速度、质量和 FLOPs Pareto frontier 同样重要。

这对课程报告很有价值，因为它说明“Masked diffusion 是当前主流”不等于“Masked diffusion 一定是最终形态”。均匀噪声、插值噪声、block-wise 生成和 few-step sampler 都可能成为后续研究方向。

### 4.7 Block Diffusion 与半自回归折中

Block Diffusion 类方法在自回归和扩散之间折中：块与块之间可以保持因果顺序，块内使用 diffusion 并行生成。这类方法试图同时获得自回归模型的长程稳定性和扩散模型的并行更新能力，是后续扩散语言模型的重要工程方向。

## 5. 算法设计要点

### 5.1 噪声 schedule

噪声 schedule 决定不同时间点的 mask 比例或跳转强度。过快加噪会导致模型在高噪声区很难恢复语义；过慢加噪则增加训练和采样成本。文本 masked diffusion 常用单调下降的 \(\alpha_t\)，并在训练中随机采样不同 mask ratio。

### 5.2 参数化方式

常见参数化包括：

1. \(x_0\)-prediction：输出 clean token 分布，最接近 MLM。
2. score/ratio prediction：输出离散 density ratio，更贴近 CTMC 反向理论。
3. hybrid parameterization：训练时使用 MLM 交叉熵，采样时结合置信度、temperature、top-k/top-p 等语言生成技巧。

### 5.3 采样调度

采样质量高度依赖调度：

1. 每步填充多少 token。
2. 是否允许已填 token 被重新 mask 或重采样。
3. 如何用置信度选择 token。
4. 采样步数与速度/质量的折中。

实际系统中，步数从几十步到上百步不等。若目标是交互式文本生成，减少采样步数是关键。

## 6. 可选复现方案：Masked Diffusion Model

若课程项目需要复现，建议优先选择 MDLM，因为它比从零实现 CTMC 反向跳转更直接。

推荐复现路线：

1. 阅读 MDLM 论文和官方代码，先跑通小模型推理或训练脚本。
2. 数据集选择 WikiText-103、OpenWebText 子集、Penn Treebank 或小规模中文语料。
3. 模型选择 Transformer encoder/decoder-only masked denoiser，先控制在 50M-200M 参数以内。
4. 前向过程采用 absorbing mask corruption。
5. 损失函数使用 masked token cross entropy，可按时间或 mask ratio 加权。
6. 采样从全 mask 开始，设置 32/64/128 denoising steps，对比不同步数的困惑度、生成样例和速度。

可做的实验对比：

1. 采样步数：16、32、64、128。
2. mask schedule：linear、cosine、logistic。
3. token 选择策略：随机填充、置信度最高优先、Gumbel 采样。
4. 与自回归小模型对比：生成速度、重复率、困惑度、人工可读性。

需要注意：完整复现 LLaDA 级别的大模型训练成本很高，不适合作为普通课程项目的基础目标。更可行的做法是复现小规模 MDLM，并把 LLaDA 作为扩展阅读和案例分析。

## 7. 更具体的研究方向

### 7.1 理论问题

1. 离散 score 的定义与连续 score 的关系。
2. CTMC 反向过程中的 density ratio 估计误差如何影响采样。
3. Masked diffusion 是否本质上学习了条件分布族 \(p(x_0^S\mid x_0^{\bar S})\)。
4. 离散时间 D3PM 与连续时间 CTMC 的极限关系。
5. 非自回归并行生成与长程一致性的理论矛盾。

### 7.2 算法问题

1. 更快采样：distillation、few-step sampler、block-wise sampler。
2. 更强条件控制：任意位置 infilling、编辑、约束解码。
3. 更好的 schedule 学习：让模型自适应决定每步解 mask 数量。
4. 混合 AR/diffusion：长程结构自回归，局部细节扩散。
5. 离散结构数据扩散：分子图、代码 AST、知识图谱。

### 7.3 下游任务

Masked diffusion 特别适合：

1. 文本 infilling 和改写。
2. 受约束生成，例如保留关键词或模板。
3. 代码补全中的中间片段填充。
4. 机器翻译和摘要中的非自回归生成。
5. 蛋白质序列、分子 SMILES 等离散序列建模。

## 8. 小组分工模板（不少于 5 人）

可在正式提交前替换为真实姓名和学号。

| 成员 | 主要贡献 |
|---|---|
| 成员 A | 负责 CTMC 离散扩散理论调研，整理生成元、反向 CTMC、density ratio 相关公式。 |
| 成员 B | 负责 D3PM、Multinomial Diffusion、SEDD 等代表性文献阅读与比较。 |
| 成员 C | 负责 Masked Diffusion/MDLM 原理整理，设计复现实验方案。 |
| 成员 D | 负责 LLaDA、Block Diffusion 等大规模 diffusion language model 扩展调研。 |
| 成员 E | 负责报告结构、图表绘制、参考文献整理和语言统一。 |
| 成员 F（可选） | 负责代码复现、实验记录、生成样例分析和 ablation 表格。 |

## 9. 建议报告结构

正式报告可按以下结构组织：

1. 引言：为什么离散数据需要专门的 diffusion model。
2. 相关工作：Multinomial Diffusion、D3PM、CTMC、SEDD、MDLM、LLaDA。
3. CTMC 离散扩散原理：前向过程、反向过程、训练目标、采样。
4. Masked Diffusion 原理：吸收态 mask、MLM 训练、迭代解 mask。
5. 复现实验或实验方案：数据集、模型、训练、采样、评价。
6. 大模型扩展：LLaDA 与 diffusion LLM 的机遇和瓶颈。
7. 讨论：与自回归 LM 的比较、理论问题、未来方向。
8. 分工说明。
9. 参考文献。

## 10. 参考文献与资料

[1] Hoogeboom et al. “Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions.” arXiv:2102.05379. https://arxiv.org/abs/2102.05379

[2] Austin et al. “Structured Denoising Diffusion Models in Discrete State-Spaces.” arXiv:2107.03006. https://arxiv.org/abs/2107.03006

[3] Campbell et al. “A Continuous Time Framework for Discrete Denoising Models.” arXiv:2205.14987. https://arxiv.org/abs/2205.14987

[4] Lou et al. “Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.” arXiv:2310.16834. https://arxiv.org/abs/2310.16834

[5] Ou et al. “Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data.” arXiv:2406.03736. https://arxiv.org/abs/2406.03736

[6] Sahoo et al. “Simple and Effective Masked Diffusion Language Models.” arXiv:2406.07524. https://arxiv.org/abs/2406.07524

[7] Nie et al. “Scaling up Masked Diffusion Models on Text.” arXiv:2410.18514. https://arxiv.org/abs/2410.18514

[8] Nie et al. “Large Language Diffusion Models.” arXiv:2502.09992. https://arxiv.org/abs/2502.09992

[9] Arriola et al. “Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.” arXiv:2503.09573. https://arxiv.org/abs/2503.09573

[10] Sahoo et al. “Scaling Beyond Masked Diffusion Language Models.” arXiv:2602.15014. https://arxiv.org/abs/2602.15014

[11] MDLM 官方项目页：https://s-sahoo.com/mdlm

[12] MDLM GitHub：https://github.com/kuleshov-group/mdlm

[13] LLaDA GitHub：https://github.com/ML-GSAI/LLaDA
