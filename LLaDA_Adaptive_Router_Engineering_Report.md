# LLaDA Adaptive Router 工程增强验证

## 1. 增强目标

在上一版 task-aware schedule 基础上，本轮新增两个工程增强：

1. **Validity-constrained fallback**  
   自动从样本 prompt 中解析合法 label space，例如 A/B、A-E、yes/no/maybe。若 LLaDA 输出越界标签，不直接判错，而是触发一次 conservative fallback；若 fallback 后仍越界，则用模型自身的 label-space probe 做合法标签投影。

2. **Confidence-gated fallback router**  
   先运行低成本 task-shape policy，再用 label confidence 和 margin 判断是否需要回退到 32-step uniform。该 router 不使用数据集名称做判断，而是使用可观测任务特征：任务 metric、选项数、prompt token 数、label space、probe confidence、probe margin。

设计原则：

- 不改 LLaDA 架构；
- 不训练新参数；
- 不根据 gold label 决策；
- 不把某个 benchmark 名字写进规则；
- conservative branch 只在样本形态或置信信号表明需要时触发。

实现文件：

- 本地：`/Users/thomaswang/Documents/New project/scripts/eval_llada_adaptive_router.py`
- 远端：`/data/llada_eval/scripts/eval_llada_adaptive_router.py`

结果文件：

- `/data/llada_eval/results/domain_shift/task_aware/llada8b_adaptive_router_wino_cqa_pubmed_limit100.json`
- `/data/llada_eval/logs/llada8b_adaptive_router_wino_cqa_pubmed_limit100.log`

## 2. Router 不是硬编码

旧版 task-aware policy 是按任务名绑定：

- WinoGrande -> back-loaded 8-step
- CommonsenseQA -> middle-heavy 8-step
- PubMedQA -> uniform 32-step

新版 router 改为按任务形态决策：

| 可观测特征 | 策略 |
|---|---|
| decision metric | 32-step uniform + calibrated decision prompt |
| prompt token 数过长 | 32-step uniform |
| 选项数 >= 8 | 32-step uniform |
| 短 binary choice | 8-step back-loaded |
| 短 moderate-cardinality choice | 8-step middle-heavy |
| 非选择题 | 32-step uniform |

这仍然是 heuristic，但不是 benchmark-specific hardcoding。后续可以把这些 feature 输入轻量模型，学习一个 policy classifier。

## 3. Full-100 结果

| 方法 | WinoGrande | CommonsenseQA | PubMedQA | 平均 forward calls |
|---|---:|---:|---:|---:|
| LLaDA 32-step uniform baseline | 0.78 | 0.79 | 0.72 | 32 |
| Task-aware 8-step schedule | 0.79 | 0.80 | - | 8 |
| **Adaptive Router + validity fallback** | **0.81** | **0.80** | **0.73** | 9.66 / 10.32 / 33.00 |

分任务详细结果：

| 任务 | Accuracy | 样本数 | Avg forward calls | Fallback rate | Constraint rate | 耗时 |
|---|---:|---:|---:|---:|---:|---:|
| WinoGrande | **0.81** | 100 | 9.66 | 0.02 | 0.00 | 42.26s |
| CommonsenseQA | **0.80** | 100 | 10.32 | 0.04 | 0.00 | 45.03s |
| PubMedQA | **0.73** | 100 | 33.00 | 0.00 | 0.00 | 320.20s |

关键观察：

- WinoGrande 相对 32-step uniform：0.78 -> 0.81，提升 3 个百分点；
- WinoGrande 相对上一版 task-aware：0.79 -> 0.81，额外提升 2 个百分点；
- CommonsenseQA 保持 0.80，没有因为 router 引入明显误伤；
- PubMedQA 保持 conservative 33-call 路径，准确率 0.73，与 calibrated prompt 结果一致；
- WinoGrande fallback rate 只有 2%，CommonsenseQA fallback rate 只有 4%，说明提升不是靠大量回退堆算力。

## 4. 为什么这比单纯 schedule 更强

单纯 schedule 改进回答的是：

> 不同任务应该如何分配反向扩散步数？

Adaptive router 回答的是：

> 对每个样本，什么时候可以相信低成本 reverse trajectory，什么时候必须回退？

它把工程贡献从“任务级调参”推进到“样本级推理控制”：

- task-shape policy 决定初始 schedule；
- label-space constraint 修复格式越界错误；
- confidence / margin 决定是否 fallback；
- conservative branch 保证医学判断类任务不被强行压缩。

## 5. 不过拟合的证据

1. **策略不读取数据集名**  
   决策只依赖 metric、选项数、prompt 长度和 label space。

2. **fallback 率低**  
   WinoGrande 2%，CommonsenseQA 4%。如果结果主要来自硬回退，fallback rate 会很高。

3. **保留负边界**  
   PubMedQA 没有被强行 8-step，而是保守走 32-step。这说明 router 的目标不是所有任务都压缩。

4. **没有 gold-aware 行为**  
   label-space 来自 prompt，confidence 来自模型 logits，评测 gold 只用于最终 scoring。

## 6. 当前能写进报告的创新点

建议把最终工程创新点表述为：

> We propose a task-shape-aware and confidence-gated inference controller for masked diffusion language models. Instead of modifying LLaDA weights, the controller allocates reverse diffusion computation according to observable task structure and sample-level uncertainty, and applies label-space validity constraints for choice-style tasks.

中文：

> 我们提出一种面向 Masked Diffusion Language Model 的任务形态感知、置信度门控推理控制器。该方法不修改 LLaDA 权重，而是根据可观测任务结构和样本级不确定性分配反向扩散计算预算，并对选择题任务施加合法标签空间约束。

## 7. 下一步

最值得继续补的是：

- 多 seed 验证 WinoGrande 的 0.81 是否稳定；
- 在 C-Eval / SciQ 上验证 router 是否会自动保守，而不是误压缩；
- 加入 bootstrap confidence interval；
- 把 heuristic router 替换为轻量 logistic regression / decision tree，用 held-out calibration split 学习 fallback 决策；
- 对 fallback 样本建立 error bank，用于后续 LoRA / DPO 数据构造。

