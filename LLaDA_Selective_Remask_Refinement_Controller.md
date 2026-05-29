# LLaDA Selective Re-masking Refinement Controller

## 1. 核心定位

本文的工程改进不改变 LLaDA-8B-Instruct 的 backbone、参数或训练目标，而是在 masked diffusion language model 的反向扩散推理环节加入一个 **条件感知的预算分配与局部再掩码精修控制器**。

它要解决的问题不是泛泛地“少跑几步”，而是：

> 对于不同任务、不同样本，LLaDA 从全 mask 状态到稳定答案所需的反向扩散时间不同。固定步数会造成计算预算错配：简单样本浪费步数，困难样本又可能过早提交。

因此，我们将推理过程建模为一个 cost-sensitive decision problem：

$$
K^*(x)
=
\arg\min_{K \in \mathcal{K}}
\left[
\mathbb{E}\bigl[\ell(y, \hat y_K)\mid x\bigr]
+
\lambda C(K)
\right],
$$

其中 \(x\) 是输入样本，\(K\) 是反向扩散预算，\(\mathcal{K}=\{8,16,24,32\}\)，\(C(K)\) 是 forward calls / latency 成本。控制器的目标是在不改模型参数的情况下，近似估计每个样本的 \(K^*(x)\)，并在高风险样本上进行局部 refinement。

当前实现文件：

```text
scripts/eval_llada_refinement_controller.py
```

实验输出 method name：

```text
selective_remask_refinement_controller
```

## 2. 与简单 adaptive sampling 的区别

简单版本通常是：

1. 先跑 8 步。
2. 如果置信度低或答案不合法，就重跑 32 步。
3. 否则直接接受 8 步结果。

这个版本太像 heuristic early stopping，也容易被认为只是 8/32 router。

当前机制做了三点增强：

1. **多预算而非二选一**  
   预算集合为 \(8,16,24,32\)，控制器根据风险分数选择逐级追加预算，而不是粗暴地 8 或 32。

2. **风险来自联合信号，而非单一置信度**  
   控制器融合 forward probe、8-step scout 轨迹、答案合法性、标签空间复杂度和 prompt 复杂度。

3. **追加预算时不是完全重跑，而是 selective re-masking refinement**  
   对 scout 已生成序列中低置信 token 重新 mask，只把额外计算集中到不稳定区域，相当于在当前反向轨迹上做局部修正。

所以更准确的创新点表述是：

> We formulate dLLM inference as condition-aware reverse-budget allocation and introduce a selective re-masking refinement controller that uses early trajectory risk signals to allocate marginal denoising computation to uncertain samples and uncertain generated positions.

中文表述：

> 本文将扩散语言模型的推理过程建模为条件感知的反向扩散预算分配问题，并提出选择性再掩码精修控制器：利用前向探测与早期反向轨迹估计样本风险，再将额外去噪预算分配给高风险样本及低置信生成位置。

## 3. 算法流程

### 3.1 Forward Probe

对每个样本先做一次低成本 label probe，得到候选标签分布：

```text
P_probe(y | x)
```

记录：

- top label
- top probability
- top-2 margin
- normalized entropy
- probe 是否可用

这些信号不是最终答案，只用于估计条件不确定性。

### 3.2 Direct Full Gate

有些样本一开始就不适合低预算 scout。例如：

- 二分类任务中 top probability 过低。
- 多选题中 entropy 过高。
- 任务 profile 本身不支持 fast candidate。

这类样本直接进入 32-step full budget，避免低预算错误污染后续判断。

当前默认规则：

```text
binary_direct_full_threshold = 0.52
multi_direct_full_threshold  = 0.36
multi_direct_full_entropy    = 0.98
```

### 3.3 8-step Scout

如果 probe 允许低预算探索，则先运行 8-step scout。

scout 不只是给一个答案，还记录早期轨迹统计：

- 中间检查点预测标签。
- 标签翻转次数 flip count。
- 首次达到最终标签的 step。
- valid label 出现次数。
- token fill confidence。

这些轨迹信号是 DDM 相比 AR 更有价值的地方：DDM 的答案不是一次性从左到右生成，而是在反向扩散中逐步稳定。因此“何时稳定”“是否频繁翻转”“低置信 token 集中在哪里”都可以作为推理控制信号。

### 3.4 Risk Score

控制器构造风险特征：

```text
probe_uncertainty
probe_entropy
margin_deficit
probe_scout_disagree
invalid_or_empty
flip_instability
late_first_final
low_fill_confidence
label_complexity
prompt_complexity
```

然后按任务类型使用不同权重。

二分类任务更重视：

- probe 与 scout 是否冲突。
- probe uncertainty。
- invalid / empty。
- margin deficit。

多选任务更重视：

- entropy。
- invalid / empty。
- label complexity。
- probe uncertainty。

风险分数：

$$
r(x)
=
\mathrm{clip}_{[0,1]}
\left(
\sum_i w_i \phi_i(x)
\right).
$$

这不是训练出的 classifier，而是一个可解释、可审计的 risk functional。它的好处是样本量较小时也可用，且每个决策都能落盘解释。

### 3.5 Multi-budget Allocation

根据风险分数选择目标预算：

```text
r < 0.24       -> keep 8-step scout
0.24 <= r < .38 -> refine to 16 steps
0.38 <= r < .56 -> refine to 24 steps
r >= 0.56      -> refine to 32 steps
```

即：

$$
K(x)=
\begin{cases}
8, & r(x)<\tau_{16},\\
16, & \tau_{16}\le r(x)<\tau_{24},\\
24, & \tau_{24}\le r(x)<\tau_{32},\\
32, & r(x)\ge\tau_{32}.
\end{cases}
$$

当前阈值：

```text
risk_t16 = 0.24
risk_t24 = 0.38
risk_t32 = 0.56
```

### 3.6 Selective Re-masking Refinement

如果目标预算大于当前预算，控制器不会简单丢弃 scout 结果重跑，而是：

1. 找到已生成 token 中置信度最低的一部分。
2. 将这些位置重新置为 `[MASK]`。
3. 用额外步数继续反向扩散。

再掩码比例由风险分数和目标预算决定：

$$
\rho(x,K)
=
\rho_{\min}
+
\bigl(\rho_{\max}-\rho_{\min}\bigr)r(x)
+
\Delta_K.
$$

当前默认：

```text
remask_min_fraction = 0.20
remask_max_fraction = 0.55
remask_min_tokens   = 4
```

直觉上，这相当于把额外计算预算从“整段输出平均分配”改成“集中修复最不稳定的 token 区域”。

### 3.7 Post-risk Check

每轮 refinement 后，控制器再次检查：

- 输出是否为空。
- 输出 label 是否非法。
- binary task 中 probe 高置信结果是否与 scout/refine 结果强冲突。

如果仍然高风险，则继续升级到下一个预算，最多执行：

```text
max_refinements = 3
```

因此最终路线可能是：

```text
8
8 -> 16
8 -> 24
8 -> 16 -> 32
8 -> 24 -> 32
32
```

这就是为什么当前机制不是固定 8/32 二选一，而是一个逐级、可解释、可落盘的动态控制器。

## 4. 理论叙事

### 4.1 Hitting Time

在 masked diffusion 中，可以把答案稳定看作反向过程首次进入某个 decision basin：

$$
\tau_y
=
\inf\{t:\hat y_t=\hat y_T\}.
$$

不同任务的 \(\tau_y\) 分布不同：

- WinoGrande 依赖代词消解和语义绑定，往往需要更晚稳定。
- CommonsenseQA 对早期候选波动更容忍，部分样本较早达到最终标签。

所以固定 \(K\) 的策略隐含假设：

$$
\tau_y(x) \approx \text{constant},
$$

而我们的实验证据恰好说明这个假设不成立。

### 4.2 Marginal Value of Computation

继续扩散一步的价值可写为：

$$
\Delta_k(x)
=
\mathbb{E}
\left[
\ell(y,\hat y_k)
-
\ell(y,\hat y_{k+1})
\mid x
\right].
$$

理想停止准则是：

$$
\text{continue if }
\Delta_k(x) > \lambda.
$$

但真实 \(\Delta_k(x)\) 不可观测。当前控制器用 probe uncertainty、trajectory instability 和 low fill confidence 近似估计继续采样的边际收益。

### 4.3 Selective Re-masking as Local Posterior Correction

对于已经填入的 token \(z_i\)，如果其模型置信度低，则该位置的 posterior uncertainty 高：

$$
U_i
=
1-\max_v p_\theta(z_i=v\mid z_{\setminus i},x,t).
$$

Selective re-masking 选择高 \(U_i\) 的位置重新进入 mask state，相当于在反向链中对不稳定位置做局部 posterior correction，而不是把整条序列推倒重来。

这也是它区别于普通 early stopping 的关键：它不仅决定何时停止，还决定额外预算应该作用在哪里。

## 5. 与已有采样改进的关系

| 方向 | 常见做法 | 本方法差异 |
| --- | --- | --- |
| Global schedule optimization | 设计固定的全局时间步分配 | 本方法按样本风险动态选择预算 |
| Early stopping | 输出稳定就提前停止 | 本方法显式估计 probe + trajectory 风险，并可继续 refinement |
| Token finalization | 稳定 token 提前冻结 | 本方法反过来重 mask 低置信 token，集中修复不稳定区域 |
| Remasking / backtracking | 对错误 token 回退修正 | 本方法把 remasking 嵌入样本级预算控制，并记录 route/risk |
| Adaptive computation | 改模型容量或专家路由 | 本方法不改 backbone，只改 inference loop |

因此论文/报告中应避免说：

```text
We propose adaptive sampling for dLLMs.
```

更稳妥的说法是：

```text
We propose a condition-aware reverse-budget controller with selective re-masking refinement for masked diffusion language models.
```

## 6. 当前落盘字段

每个样本都会保存：

```text
task
id
gold
pred
correct
output
profile
probe
risk_features
risk_score
scout_stats
route
route_steps
final_budget
seconds
forward_calls
meta
```

这些字段能支持三类分析：

1. **性能分析**  
   accuracy、avg forward calls、latency、route distribution。

2. **行为边界分析**  
   哪些任务进入 32-step，哪些任务停在 8/16/24。

3. **错例分析**  
   错误来自 invalid/empty、probe-scout disagreement、late stabilization，还是知识缺口。

## 7. 预期实验呈现方式

主表应至少包含：

| Method | Accuracy | Avg Calls | Final Budget Distribution | Invalid Rate |
| --- | ---: | ---: | --- | ---: |
| Fixed 8-step | - | - | 8 only | - |
| Fixed 16-step | - | - | 16 only | - |
| JYS-like 16-step | - | - | 16 only | - |
| Prophet early commit | - | - | early stop | - |
| Fixed 32-step | - | - | 32 only | - |
| Selective re-masking controller | - | - | 8/16/24/32 dynamic | - |

重点不是只证明“更快”，而是证明：

1. route distribution 非退化，确实有样本级动态分配。
2. WinoGrande 等语义绑定任务更常升级预算。
3. CommonsenseQA / BoolQ 中部分样本可以低预算接受。
4. 无效输出率不能上升，否则说明控制器牺牲了格式可靠性。
5. 与 Prophet / fixed schedule / JYS-like 对比时，本方法的独特性来自 risk-conditioned refinement，而不是单纯 early stop。

## 8. 报告中的主张边界

应该明确写：

- 我们没有修改 LLaDA 架构。
- 我们没有训练新的 DDM。
- 我们的方法是 inference-loop controller。
- 它适合处理 trajectory-sensitive errors。
- 它不能解决纯知识缺口，例如 C-Eval 中模型不知道答案的情况。
- 它也不能替代 LoRA / SFT，只能作为解码侧的正交改进。

推荐最终 thesis：

> LLaDA 的下游表现不仅由模型参数决定，也受反向扩散轨迹稳定性和推理预算分配影响。通过记录早期轨迹并进行条件感知的选择性再掩码精修，可以在不改模型架构的情况下，更合理地分配反向扩散计算预算，并揭示不同任务的 diffusion-time behavior boundary。

## 9. 后续可增强方向

当前机制已经比 8/32 router 更像研究级方法，但仍有三类可继续增强：

1. **阈值校准**  
   用 held-out calibration set 学习 \(\tau_{16},\tau_{24},\tau_{32}\)，而不是手设阈值。

2. **学习型 risk model**  
   用当前落盘的 `risk_features` 训练轻量 logistic / isotonic calibrator，预测 scout 是否会错。

3. **token-level refinement policy**  
   当前按低置信 token 重 mask，后续可加入 label-relevant span、attention saliency 或 answer-token neighborhood。

这些增强不需要改 LLaDA backbone，仍然保持“inference control”的主线。
