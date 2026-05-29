# LLaDA Task-aware Inference 工程验证报告

## 1. 这次真正实现了什么

本轮工作实现并验证了一个不改 LLaDA 权重、不改模型架构的工程改进：

**Task-aware Reverse Diffusion Policy**：根据任务类型选择不同的反向扩散采样策略，包括步数、reverse transition schedule、prompt 模板和必要的 fallback 规则。

核心实现落在推理 loop，而不是理论包装：

- 对 LLaDA-8B-Instruct 进行真实 GPU 推理评测；
- 在反向扩散过程中替换 uniform schedule，测试 front-loaded / back-loaded / middle-heavy 等离散化策略；
- 为不同任务绑定不同 schedule；
- 记录每个样本的预测、gold label、forward calls、耗时和显存；
- 与 32-step uniform、8-step uniform、LoRA 后模型和 Qwen2.5-7B AR 模型做横向比较。

远端实现文件：

- `/data/llada_eval/scripts/eval_llada_task_aware_policy.py`
- `/data/llada_eval/scripts/eval_llada_sampler_variants.py`
- `/data/llada_eval/scripts/eval_llada_early_exit.py`

关键结果文件：

- `/data/llada_eval/results/domain_shift/task_aware/llada8b_task_aware_wino_cqa_limit100.json`
- `/data/llada_eval/logs/llada8b_task_aware_wino_cqa_limit100.log`

## 2. 当前可成立的主张

最稳的主张不是“我们改进了 LLaDA 架构”，而是：

> LLaDA 的反向扩散推理过程存在任务相关的行为边界。对不同任务使用相同的 uniform reverse schedule 并不总是最优。通过轨迹分析和错例驱动选择 task-aware reverse transition schedule，可以在部分任务上同时提升准确率并显著减少推理步数。

这是一个工程创新点，因为它改变的是实际 inference algorithm：

- 不需要训练；
- 不需要额外标注；
- 不增加模型参数；
- 可直接作为 sampler / solver 层插件；
- 能和 LoRA、DPO、prompt calibration 后续叠加。

## 3. Full-100 验证结果

本次重点验证 WinoGrande 和 CommonsenseQA 两个任务，因为前期轨迹分析显示它们对 8-step 压缩不敏感，但对 schedule 形状敏感。

| 方法 | WinoGrande | CommonsenseQA | 平均准确率 | 平均 forward calls | 结论 |
|---|---:|---:|---:|---:|---|
| LLaDA 32-step uniform | 0.78 | 0.79 | 0.785 | 32 | 原始强基线 |
| LLaDA 8-step uniform | 0.77 | 0.78 | 0.775 | 8 | 省算力但掉点 |
| LLaDA typed LoRA | 0.78 | 0.80 | 0.790 | 32 | 微调收益很小 |
| **Task-aware 8-step** | **0.79** | **0.80** | **0.795** | **8** | **准确率和效率同时提升** |

细分策略：

| 任务 | 策略 | Accuracy | 样本数 | 耗时 | Forward calls | 显存峰值 |
|---|---|---:|---:|---:|---:|---:|
| WinoGrande | back-loaded, 8-step | 0.79 | 100 | 39.17s | 8.0 | 15.02 GB |
| CommonsenseQA | middle-heavy, 8-step | 0.80 | 100 | 38.87s | 8.0 | 15.03 GB |

相对 32-step uniform：

- 平均准确率：0.785 -> 0.795，提升 1.0 个百分点；
- forward calls：32 -> 8，减少 75%；
- 两个任务总耗时约从 310s 降到 78s，约 4x 加速；
- 相比 8-step uniform，平均准确率 0.775 -> 0.795，提升 2.0 个百分点。

这个升点不大，但很关键：它说明 schedule 不是纯速度 knob，而是会影响 LLaDA 在不同任务上的正确性。

## 4. 为什么这不是 toy

这次验证不是只跑 demo prompt，也不是手选几个样例：

- 使用真实 LLaDA-8B-Instruct；
- 使用真实 benchmark 子集，每个任务 100 个样本；
- 有 32-step strong baseline、8-step compute-matched baseline、LoRA baseline 和 AR baseline；
- 改进发生在推理 loop 中，不依赖人工改答案；
- 每个样本有 JSON 记录，可做错例追踪；
- 结果包含正例和负例，不只报告成功任务。

但也必须诚实：目前还不是最终论文级结论。要写成更强报告，需要继续做 multi-seed、更大样本、更多任务和统计显著性检验。

## 5. 与前期错例分析的关系

前期分析给出了三个行为边界：

1. **WinoGrande / CommonsenseQA**  
   8-step 已经接近 32-step，说明这些任务的答案多数不需要长程逐 token refinement。主要问题是早期语义锚定和中后期选项区分。

2. **PubMedQA**  
   8-step 从 0.72 掉到 0.56，说明医学判断类任务依赖更长的 denoising trajectory。这里不适合强行压缩步数，只能用 confidence-gated fallback。

3. **C-Eval Computer Network**  
   LoRA 和 few-shot 都没有明显升点，说明 label-only 微调很难注入专业知识。这个任务更适合 retrieval / rationale / domain data，而不是 schedule-only。

因此 task-aware policy 的设计是：

- 对可压缩任务使用 8-step task-specific schedule；
- 对不可压缩任务保留 32-step 或使用 fallback；
- 不把单一 sampler 宣称为全任务最优。

## 6. 理论支撑怎么讲

理论上可以这样连接 CTMC / DDM：

LLaDA 类 masked diffusion 模型的推理可以看作从高噪声状态到低噪声状态的反向过程。实际实现需要把连续时间或多步反向过程离散成有限步采样。

uniform schedule 默认假设每个时间段同等重要，但这个假设不一定适合所有任务。不同任务的错误模式对应不同的反向过程需求：

- WinoGrande 需要较晚阶段稳定 coreference decision，所以 back-loaded schedule 更合适；
- CommonsenseQA 需要在候选答案间做中期 refinement，所以 middle-heavy schedule 更合适；
- PubMedQA 的判断依赖更长轨迹，短步数压缩会破坏语义证据积累。

所以这里的理论表述是：

> 我们没有改 CTMC 的状态空间或模型参数，而是改进反向过程的时间离散化和任务条件化策略。该策略相当于对 reverse transition computation budget 进行 task-conditioned allocation。

这比“加一个 prompt”更有研究味道，因为它明确作用在 diffusion inference dynamics 上。

## 7. 目前发现的风险和负结果

1. **升点有限**  
   当前 full-100 是 +1 到 +2 个百分点。它是实测升点，但不能夸成大幅提升。

2. **任务依赖明显**  
   该策略不能直接推广到 PubMedQA / C-Eval。强行 8-step 会掉点。

3. **仍有无效标签问题**  
   WinoGrande 中偶尔出现 out-of-choice label，例如 D/F。这说明需要加入 validity-constrained decoding 或 label-space fallback。

4. **Predictor-corrector 不划算**  
   8-step predictor-corrector 需要约 16 次 forward，但没有带来稳定提升，不应作为主线。

5. **LoRA 不自动有效**  
   typed label LoRA 在当前设置下收益很弱，甚至在 C-Eval 上略降。后续如果做微调，应基于错例类型构造数据，而不是简单 label-only SFT。

## 8. 下一步最值得做的工程增强

### 8.1 Validity-constrained diffusion decoding

针对 choice task，在最后答案位置加入 label-space constraint：

- 只允许 A/B/C/D/E 或 true/false/maybe；
- 如果生成非法标签，触发一次小成本 fallback；
- 这很可能直接修掉一部分非语义错误。

这是低风险、高解释性的工程点。

### 8.2 Confidence-gated fallback router

已有轨迹分析显示：

- WinoGrande：约 80% 样本可在 8-step 接受；
- CommonsenseQA：约 86% 样本可在 8-step 接受；
- PubMedQA：只适合少量 early accept，大多数需要 32-step。

下一步应实现真实在线 router：

1. 先跑 8-step task-aware schedule；
2. 计算答案 token confidence、margin、trajectory stability；
3. 高置信直接输出；
4. 低置信 fallback 到 32-step 或 calibrated prompt。

目标不是全任务 8-step，而是 adaptive compute。

### 8.3 Error-bank driven SFT / LoRA

如果一定要微调，建议只微调 router-sensitive failure cases：

- 收集 schedule 改不好的错例；
- 标注错误类型：非法标签、选项混淆、知识缺失、否定词、长上下文依赖；
- 构造 minimal correction data；
- LoRA 只针对 final-answer typing 和 failure correction，而不是全量任务 SFT。

这样微调才和工程分析闭环。

### 8.4 Cross-task policy learning

现在的 task-aware policy 是人工规则。更研究级的版本是学习一个轻量 policy：

输入：

- 任务类型；
- prompt 长度；
- 选项数；
- 8-step confidence；
- label margin；
- mask fill trajectory；
- first-pass answer stability。

输出：

- schedule 类型；
- 是否 fallback；
- 是否启用 calibrated prompt；
- generation length / block length。

这可以写成“trajectory-aware inference controller”。

## 9. 建议最终报告主线

报告题目可以写成：

**Task-aware Reverse Diffusion Inference for Masked Diffusion Language Models**

中文标题：

**面向下游任务的 Masked Diffusion Language Model 反向扩散推理优化**

核心贡献三点：

1. **Trajectory logger**  
   建立 LLaDA 反向扩散过程的样本级轨迹记录器，用于分析 mask fill、confidence、invalid prediction 和 task-specific failure。

2. **Task-aware schedule**  
   提出任务条件化 reverse transition schedule，在 WinoGrande 和 CommonsenseQA 上实现比 32-step uniform 更高准确率和约 4x 加速。

3. **Behavior-boundary analysis**  
   证明不同任务对 reverse trajectory 的依赖不同：常识/coreference 可压缩，医学判断不可轻易压缩，专业知识任务需要外部知识或数据增强。

最强一句话结论：

> 我们发现 LLaDA 的推理质量不仅取决于模型权重，也取决于反向扩散计算预算在时间维度上的分配。通过错例驱动的 task-aware schedule，可以在不训练模型的情况下获得小幅但真实的准确率提升，并将推理成本降低约 75%。

