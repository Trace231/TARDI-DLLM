# Task-aware Reverse Diffusion Inference for LLaDA

## 0. 项目定位

本项目不以“重新训练或改造 LLaDA 主干架构”为目标。原因是 7B/8B 级 diffusion language model 的预训练、结构级改造和大规模继续训练超出当前资源。我们的目标是做一个更现实、也更有研究味的方向：

> 在不改 backbone 参数的前提下，研究 LLaDA 在不同下游任务上的反向扩散推理行为边界，并提出 task-aware 的反向转移策略，使模型在保持准确率的同时降低推理开销，或在特定任务上提升稳定性。

核心思想是：LLaDA 的反向生成不是普通 AR token-by-token decoding，而是一个 masked reverse diffusion process。这个过程暴露出中间轨迹、mask 解开顺序、token 置信度、label 收敛时间等可观测信号。我们要把这些信号变成工程系统，而不是只跑 benchmark。

## 1. 研究问题

### RQ1：不同下游任务是否需要不同的 diffusion inference policy？

已有实验表明，统一 32-step uniform sampler 不是最优策略：

- WinoGrande / CommonsenseQA 上，8-step 已经接近 32-step。
- PubMedQA 上，8-step 明显偏向 `maybe`，从 32-step 的 0.72 降到 0.56。
- C-Eval 计算机网络上，few-shot 和 label-only LoRA 都无法补领域知识。

因此我们要验证：

> LLaDA 的最优推理策略是否由任务类型、轨迹置信度和错误类型共同决定？

### RQ2：采样器改动能否作为现实可行的模型改进？

队友报告里的 RK / Trapezoidal / Adaptive Mask 思路本质上是采样器或反向转移策略，而不是主干架构。我们将其落地为：

- 非均匀 unmask schedule；
- task-aware step budget；
- confidence-gated fallback；
- failure-aware router；
- 可选的 predictor-corrector sampler 作为负结果对照。

### RQ3：能否把错例分析转成可运行的 inference router？

不是人工总结“哪些错了”，而是构建一个工程模块：

```text
输入：任务类型 + 8-step 轨迹特征 + 置信度 + 输出格式
输出：接受 8-step / fallback 32-step / 使用校准 prompt / 触发知识增强
```

这使错例分析从报告内容变成系统组件。

## 2. 理论支撑

理论部分只服务于工程设计，不作为主要贡献。

### 2.1 CTMC 视角

Discrete diffusion 可以视为离散状态空间上的连续时间马尔可夫过程。生成矩阵 `Q` 描述 token 状态如何随时间转移。Masked diffusion 是其中一个特殊情形：正向过程逐渐把 token mask 掉，反向过程学习从 mask 状态恢复原 token。

这个视角给出两个工程启发：

- 反向转移速率不一定要在每一步均匀分配。
- 不同任务可能对应不同的有效 mixing / denoising 难度。

### 2.2 采样步数与离散化误差

RK / Trapezoidal 等高阶 solver 的理论动机是减少离散化误差。但我们的初步实验显示：

- WinoGrande: 8/16/32/64 steps = 0.77/0.77/0.78/0.78。
- 64-step 相比 8-step 近 8 倍计算，但准确率只多约 1 个点。

因此，对于某些任务，错误不主要来自数值离散化误差，而来自语义绑定或任务决策边界。

### 2.3 Adaptive Mask 的工程化解释

队友报告中的 Adaptive Mask 思路可以转化为我们的工程目标：

> 不同 token、不同任务、不同失败类型不应该使用同一套反向转移策略。

我们不重新训练一个 adaptive mask 模型，而是在 inference loop 中设计 task-aware reverse transition policy。

## 3. 已有实验基础

### 3.1 大样本评测

6 个任务，每个任务 100 样本，LLaDA-8B 与 Qwen2.5-7B 对比：

| 任务 | Qwen2.5-7B | LLaDA-8B | LLaDA typed LoRA |
|---|---:|---:|---:|
| MMLU-Pro | 0.35 | 0.44 | 0.44 |
| PubMedQA | 0.57 | 0.72 | 0.72 |
| C-Eval 计算机网络 | 0.73 | 0.56 | 0.54 |
| SciQ | 0.93 | 0.88 | 0.88 |
| WinoGrande | 0.70 | 0.78 | 0.78 |
| CommonsenseQA | 0.80 | 0.79 | 0.80 |

结论：

- LLaDA 有竞争力，但不是全面优于 AR。
- label-only LoRA 不是通用解。
- 下游任务存在明显行为边界。

### 3.2 Adaptive decoding 初步结果

8-step 与 32-step 对比：

| 任务 | 8-step acc | 32-step acc | 同 final label | 8-step 时间 | 32-step 时间 |
|---|---:|---:|---:|---:|---:|
| WinoGrande | 0.77 | 0.78 | 95/100 | 39.3s | 154.7s |
| CommonsenseQA | 0.78 | 0.79 | 95/100 | 39.1s | 155.2s |
| PubMedQA | 0.56 | 0.72 | 74/100 | 73.3s | 292.5s |

Confidence-gated fallback 模拟：

| 任务 | 接受 8-step 比例 | 模拟准确率 | 平均 steps | steps 节省 |
|---|---:|---:|---:|---:|
| WinoGrande | 80/100 | 0.78 | 12.8 | 60% |
| CommonsenseQA | 86/100 | 0.80 | 11.4 | 64% |
| PubMedQA | 18/100 | 0.72 | 27.7 | 14% |

### 3.3 Sampler variant 初步结果

两个任务各 50 样本：

| Sampler | WinoGrande | CommonsenseQA | forward calls |
|---|---:|---:|---:|
| uniform 8-step | 0.76 | 0.76 | 8 |
| uniform 32-step | 0.78 | 0.80 | 32 |
| front-loaded 8-step | 0.78 | 0.80 | 8 |
| back-loaded 8-step | 0.80 | 0.78 | 8 |
| middle-heavy 8-step | 0.76 | 0.82 | 8 |
| predictor-corrector 8-step | 0.76 | 0.78 | 16 |

结论：

- schedule 改动有用，且不增加 forward calls。
- predictor-corrector 不划算。
- 最优 schedule 具有任务依赖性。

## 4. 主要贡献设计

### Contribution 1：Trajectory Logger

构建 LLaDA 去噪轨迹记录器，记录：

- 每个 diffusion step 的 decoded text；
- parsed final label；
- mask remaining / filled ratio；
- selected token confidence mean/min；
- 8-step 与 32-step 预测是否一致；
- 是否发生 answer contract error；
- 是否出现 early wrong lock / late correction / late corruption。

对应代码：

```text
/data/llada_eval/scripts/eval_llada_early_exit.py
```

研究价值：

- 将 diffusion 中间过程变成可观测对象。
- 为 task-aware inference router 提供特征。
- 与 AR 模型形成区分。

### Contribution 2：Task-aware Reverse Transition Scheduler

不再使用固定 uniform unmask schedule，而是根据任务类型选择不同反向转移策略。

候选策略：

| 策略 | 含义 | 预期适用 |
|---|---|---|
| uniform | 每步解开近似相同 token 数 | baseline |
| front-loaded | 早期解开更多 token | 需要快速形成全局语义 |
| back-loaded | 后期解开更多 token | 指代/语义绑定类任务 |
| middle-heavy | 中间阶段解开更多 token | 常识多选、干扰项任务 |
| predictor-corrector | 每步预测后再校正 | 高阶 solver 对照 |

当前初步结果显示：

- WinoGrande 偏好 back-loaded；
- CommonsenseQA 偏好 middle-heavy；
- predictor-corrector 成本高且收益差。

### Contribution 3：Confidence-gated Step Allocation

流程：

```text
1. 先跑 cheap 8-step。
2. 读取 final selected-token confidence。
3. 如果置信度高，接受 8-step。
4. 如果置信度低，fallback 到 32-step。
```

这不是简单 early exit。因为 LLaDA 的 `Final answer: X` 往往到最后几步才拼完整，in-loop text-stability early exit 很难触发。更合理的策略是 coarse-to-fine：

```text
coarse 8-step decode -> confidence check -> optional 32-step refinement
```

### Contribution 4：Failure-aware Router

构建一个可解释 router，根据任务类型和轨迹特征选择推理策略。

初版规则：

| 任务类型 | cheap policy | fallback |
|---|---|---|
| 指代 / 语义绑定 | back-loaded 8-step | 32-step |
| 常识多选 | middle-heavy 或 front-loaded 8-step | 32-step |
| 医学 yes/no/maybe | 32-step + calibration | 不使用 8-step accept |
| 领域知识 | 32-step | RAG / domain hint |
| 格式敏感 | typed final-label | parser-constrained prompt |

后续可以训练轻量分类器：

- Logistic Regression；
- Decision Tree；
- Random Forest；
- 不使用大模型，避免工程过重。

输入特征：

```text
task_id
metric_type
option_count
pred_8
conf_min_8
conf_mean_8
filled_ratio_curve
label_valid
question_length
same_pred_under_schedule_variants
```

预测目标：

```text
safe_8 = pred_8 == pred_32
```

或更严格：

```text
safe_8 = correct_8 == correct_32
```

### Contribution 5：Error Bank

构建错例库：

```text
question
task
gold
pred_8
pred_32
confidence
schedule
failure_type
recommended_policy
```

用途：

- 分析每类错误的轨迹特征；
- 支持 error-driven routing；
- 给报告提供具体案例；
- 后续可加入检索：新样本与错例库相似时触发对应策略。

## 5. 实验设计

### 5.1 任务集合

主实验任务：

| 类型 | 任务 | 用途 |
|---|---|---|
| 专业多选 | MMLU-Pro | general knowledge / multi-option |
| 医学决策 | PubMedQA | calibration boundary |
| 中文领域知识 | C-Eval Computer Network | domain knowledge gap |
| 科学知识 | SciQ | science MC |
| 指代推理 | WinoGrande | global dependency |
| 常识推理 | CommonsenseQA | commonsense distractor |

扩展任务：

- ARC-Challenge；
- BoolQ；
- HellaSwag；
- GSM8K scalar answer。

### 5.2 Baselines

| Baseline | 描述 |
|---|---|
| Qwen2.5-7B-Instruct | AR baseline |
| LLaDA 32-step uniform | default diffusion baseline |
| LLaDA 8-step uniform | cheap decoding baseline |
| LLaDA typed LoRA | adaptation baseline |
| calibrated prompt | decision calibration baseline |

### 5.3 Proposed methods

| 方法 | 描述 |
|---|---|
| Task-aware schedule | 按任务选择 front/back/middle schedule |
| Confidence-gated fallback | 8-step high confidence 接受，否则 32-step |
| Failure-aware router | 任务类型 + 轨迹特征选择策略 |
| Error-bank routing | 利用相似错例决定 fallback 类型 |

### 5.4 指标

准确率指标：

- accuracy；
- macro average；
- per-label accuracy；
- invalid output rate；
- 8-step vs 32-step label agreement。

效率指标：

- wall-clock time；
- average forward calls；
- average diffusion steps；
- step reduction；
- peak VRAM。

路由指标：

- accepted ratio；
- fallback ratio；
- safe accept precision；
- unsafe accept rate；
- leave-one-task-out router accuracy。

错例指标：

- improved flips；
- regressed flips；
- high-confidence wrong；
- low-confidence correct；
- schedule-sensitive examples；
- task-specific error type distribution。

## 6. 关键实验矩阵

### Experiment A：跨任务 trajectory profiling

对每个任务跑：

```text
8-step uniform trace
32-step uniform trace
```

输出：

- pred_8；
- pred_32；
- confidence；
- label agreement；
- error type。

目标：

> 证明不同任务对步数和轨迹置信度的依赖不同。

### Experiment B：sampler schedule ablation

对 WinoGrande、CommonsenseQA、PubMedQA、C-Eval 至少四个任务测试：

```text
uniform
front-loaded
back-loaded
middle-heavy
predictor-corrector
```

目标：

> 验证 schedule 是可调节的 inference policy，且最优策略与任务有关。

### Experiment C：confidence-gated fallback

策略：

```text
if conf_min_8 >= threshold:
    accept 8-step
else:
    fallback 32-step
```

对每个任务画：

```text
accuracy vs average steps
```

目标：

> 证明部分任务可以大幅省计算，而部分任务不适合省。

### Experiment D：Failure-aware router

训练方式：

- 使用 5 个任务训练 router；
- 留 1 个任务测试；
- 做 leave-one-task-out。

模型：

- rule-based router；
- logistic regression；
- decision tree。

目标：

> 证明 task-aware + trajectory feature 比单一 confidence threshold 更稳。

### Experiment E：Counterfactual option sensitivity

对 CommonsenseQA / C-Eval / MMLU-Pro：

- 打乱选项顺序；
- 删除一个明显错误选项；
- 替换相近干扰项；
- 观察 LLaDA 8-step / 32-step 是否稳定。

目标：

> 判断错误来自语义理解，还是选项位置/干扰项敏感性。

这是错例驱动的可选增强实验，很适合提升研究深度。

## 7. 工程实现计划

### Module 1：Trajectory Runner

输入：

```text
model path
task
limit
steps
schedule
sampler
```

输出：

```json
{
  "task": "...",
  "id": "...",
  "gold": "...",
  "pred": "...",
  "correct": true,
  "trace": [
    {
      "step": 1,
      "filled_ratio": 0.125,
      "pred": "",
      "selected_conf_mean": 0.91,
      "selected_conf_min": 0.63,
      "text": "..."
    }
  ]
}
```

### Module 2：Sampler Variants

已实现：

```text
/data/llada_eval/scripts/eval_llada_sampler_variants.py
```

需要扩展：

- 对更多任务跑；
- 输出 schedule-sensitive examples；
- 支持 task-aware schedule config。

### Module 3：Adaptive Policy Simulator

输入：

```text
8-step records
32-step records
confidence threshold
```

输出：

- simulated accuracy；
- average steps；
- accepted ratio；
- fallback ratio；
- unsafe accept examples。

### Module 4：Router

初版 rule-based：

```python
if task in ["winogrande"]:
    schedule = "back_loaded"
    steps = 8
    fallback = "32_uniform_if_low_conf"
elif task in ["commonsenseqa"]:
    schedule = "middle_heavy"
    steps = 8
    fallback = "32_uniform_if_low_conf"
elif task in ["pubmedqa"]:
    schedule = "uniform"
    steps = 32
    prompt = "calibrated_decision"
elif task in ["ceval_computer_network"]:
    schedule = "uniform"
    steps = 32
    fallback = "rag_or_domain_hint"
```

进阶版：

```text
classifier(features) -> accept_8 / fallback_32 / calibrated / knowledge_augmented
```

### Module 5：Report Generator

自动生成：

- 任务表；
- schedule ablation 表；
- accuracy-step 曲线；
- error type 分布；
- 代表错例；
- router 决策案例。

## 8. 预期结果

### 8.1 正向结果

我们希望得到：

- WinoGrande / CommonsenseQA 上用 8-step 或 adaptive schedule 保持接近 32-step 的 accuracy；
- 平均 steps 降低 50%-65%；
- task-aware schedule 优于固定 uniform 8-step；
- router 优于单一 confidence threshold；
- PubMedQA / C-Eval 的失败说明 task-aware policy 必要。

### 8.2 可接受的负结果

以下负结果同样有价值：

- predictor-corrector 不提升；
- PubMedQA 不适合 cheap decoding；
- C-Eval 不适合靠 sampler 修；
- 高置信错误无法靠增加 steps 修。

这些负结果支撑行为边界主线。

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| schedule ablation 在大样本上收益消失 | sampler 改进不够强 | 保留为任务依赖发现，主打 router 和 compute allocation |
| router 泛化差 | 研究贡献变弱 | 使用 rule-based + leave-one-task-out 分析，强调边界而非万能 |
| PubMedQA 结果不稳定 | calibration 模块弱 | 改为失败案例，说明 decision tasks 不能 cheap decode |
| C-Eval 无法提升 | 无知识增强结果 | 作为 domain knowledge boundary，建议 RAG/SFT |
| 轨迹文件过大 | 工程负担 | 只保存 checkpoint steps 和摘要特征 |

## 10. 最终报告结构

建议标题：

> Task-aware Reverse Diffusion Inference for Masked Diffusion Language Models

中文标题：

> 面向下游任务的 Masked Diffusion Language Model 自适应反向扩散推理

报告结构：

1. 引言：LLaDA 与 masked diffusion LM 的特点。
2. 理论动机：CTMC / reverse transition / mask schedule。
3. 问题定义：固定 sampler 在跨任务上的局限。
4. 基础评测：LLaDA vs Qwen，6 个任务。
5. 轨迹分析：8-step vs 32-step，任务边界。
6. Sampler 改进：非均匀 schedule 与 predictor-corrector。
7. Task-aware Router：策略设计与模拟结果。
8. 错例分析：contract / calibration / knowledge / dependency。
9. 讨论：为什么不是改 backbone，为什么 sampler policy 是现实改进。
10. 结论与未来工作。

## 11. 交付物

### 代码

- `eval_domain_shift.py`
- `eval_llada_early_exit.py`
- `eval_llada_sampler_variants.py`
- `simulate_adaptive_policy.py`
- `train_failure_router.py`
- `build_error_bank.py`

### 数据

- large-sample benchmark JSON；
- trajectory traces；
- sampler ablation outputs；
- adaptive decoding records；
- error bank CSV。

### 报告

- `large_sample_typed_final_label_report.md`
- `targeted_intervention_report.md`
- `adaptive_decoding_report.md`
- `sampler_variant_report.md`
- final research report。

## 12. 时间安排

### Day 1：补齐跨任务 trajectory

- MMLU-Pro / C-Eval / SciQ 8-step trace；
- 与已有 PubMedQA / WinoGrande / CommonsenseQA 对齐；
- 生成 8-vs-32 summary。

### Day 2：完整 sampler ablation

- 四种 schedule；
- predictor-corrector 对照；
- 每任务 100 样本；
- 输出 schedule-sensitive examples。

### Day 3：adaptive policy / router

- confidence threshold sweep；
- rule-based router；
- logistic regression / decision tree router；
- leave-one-task-out。

### Day 4：错例库与 counterfactual

- error bank；
- 多选题 option sensitivity；
- 代表错例整理。

### Day 5：报告与图表

- 主表；
- accuracy vs steps；
- router curve；
- 错例 taxonomy；
- final report。

## 13. 最终主张

我们最终不声称：

> LLaDA 全面优于 AR 模型。

也不声称：

> 我们改造了 LLaDA 的主干架构。

我们要声称：

> LLaDA 的反向扩散推理存在明显任务边界。通过记录去噪轨迹并利用任务类型与置信度信号，可以构建 task-aware reverse diffusion inference policy，在常识/指代类任务上显著降低推理开销，同时避免在医学决策和领域知识任务上错误地使用 cheap decoding。

这是一个兼具理论动机、工程实现和实证分析的研究级计划。

