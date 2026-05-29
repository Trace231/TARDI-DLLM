# LLaDA Task-aware Diffusion Inference: Research-grade Execution Plan

## 0. One-sentence Thesis

我们不声称改了 LLaDA 主干架构，而是提出并验证一个 **Calibrated Forward-aware Risk Controller**：利用 masked diffusion 反向过程可观测的 forward probe、cheap scout、label stability 和 route decision，把固定步数采样改成任务/样本感知的反向预算分配，从而在不微调 backbone 的条件下，守住高风险任务的准确率边界，并在低风险任务上显著减少反向扩散调用。

## 1. 研究定位

### 1.1 不做什么

- 不做 7B/8B 级 LLaDA 预训练或结构级重训。
- 不把 sampler 改动包装成模型架构改动。
- 不只跑 benchmark 后堆表格。
- 不用硬编码样本 id、答案词或数据集泄漏特征。

### 1.2 做什么

我们做一个面向 masked diffusion language model 的推理控制系统：

```text
prompt/task
  -> forward label probe
  -> cheap reverse scout
  -> risk features
  -> calibrated route decision
  -> accept cheap decode or fallback expensive decode
  -> trajectory/error logging
```

核心可解释变量：

- `probe_top_prob`: 在 `Final answer: [MASK]` 上的标签后验峰值。
- `probe_entropy`: 标签边缘分布的不确定性。
- `probe_scout_agreement`: 一步 forward label posterior 与 8-step reverse trajectory 是否一致。
- `invalid_label`: 输出是否违反 answer contract。
- `first_final_step`: 轨迹首次进入合法 final label 子集的时间。
- `route`: `8`, `32`, `8->32`。

## 2. 理论叙事

### 2.1 CTMC / DDM 对应

Masked diffusion 可看作离散状态空间上的连续时间马尔可夫过程。正向过程把原始 token 推向 mask/噪声状态，反向过程学习从噪声状态恢复数据分布。

我们的工程控制器对应于：

- 固定 reverse schedule: 每个样本使用同一离散化网格。
- Task-aware schedule: 根据任务族改变反向预算分配。
- Forward-aware risk gate: 在反向解码前估计标签边缘后验的不确定性。
- Fallback: 当 cheap trajectory 可能偏离高预算 reverse process 时，增加反向计算预算。

### 2.2 为什么不是普通 early exit

LLaDA 的答案通常不是 AR 那样逐 token 固定下来，而是在多个 mask 位置并行去噪。许多样本的 final label 在早期并不稳定，甚至合法 label 出现后还可能被修正。因此我们不把目标定义为“提前停止生成”，而定义为：

> 对每个样本分配足够但不过量的 reverse diffusion budget。

这比简单 early exit 更贴合 diffusion trajectory。

### 2.3 可检验假设

H1: 不同任务存在不同 label hitting time 分布，统一 32-step 不是最优预算分配。

H2: forward probe 的低置信度和 probe-scout disagreement 能识别 cheap decode 的高风险样本。

H3: 经过任务校准后，controller 能避免两种失败：

- 过度接受 cheap decode，导致 WinoGrande 这类语义绑定任务掉点。
- 过度 fallback，导致 CommonsenseQA 这类低风险多选任务浪费计算。

## 3. 已完成基线

### 3.1 跨任务模型对比

每任务 100 样本，typed final-label 评测：

| Task | Qwen2.5-7B | LLaDA-8B | LLaDA typed LoRA |
|---|---:|---:|---:|
| MMLU-Pro | 0.35 | 0.44 | 0.44 |
| PubMedQA | 0.57 | 0.72 | 0.72 |
| C-Eval CN | 0.73 | 0.56 | 0.54 |
| SciQ | 0.93 | 0.88 | 0.88 |
| WinoGrande | 0.70 | 0.78 | 0.78 |
| CommonsenseQA | 0.80 | 0.79 | 0.80 |

结论：LLaDA 不是全面优于 AR；LoRA 也不是万能解。这个结果支撑我们转向“行为边界 + 推理控制”。

### 3.2 500 样本 held-out 验证

| Method | WinoGrande Acc | Wino Calls | CommonsenseQA Acc | CQA Calls |
|---|---:|---:|---:|---:|
| 32-step baseline | 0.756 | 32.000 | 0.810 | 32.000 |
| Old adaptive | 0.732 | 9.462 | 0.810 | 9.924 |
| Forward-aware | 0.756 | 17.768 | 0.812 | 10.856 |
| Calibrated controller | 0.756 | 17.768 | 0.812 | 9.000 |

成对结论：

- WinoGrande: calibrated 与 32-step 完全持平，修复 old adaptive 的掉点。
- CommonsenseQA: calibrated 与 forward-aware 精度相同，但把平均调用从 10.856 降到 9.000。

### 3.3 1000 样本扩展验证

| Method | Task | Acc | Wilson 95% CI | Avg Calls | Route / Fallback |
|---|---|---:|---:|---:|---|
| Calibrated | WinoGrande | 0.756 | [0.728, 0.782] | 17.560 | 32: 33.8%, 8: 64.8%, 8->32: 1.4% |
| Old adaptive | WinoGrande | 0.736 | [0.708, 0.762] | 9.363 | fallback: 1.1% |
| Calibrated | CommonsenseQA | 0.817 | [0.792, 0.840] | 9.064 | 8: 99.8%, 8->32: 0.2% |
| Old adaptive | CommonsenseQA | 0.816 | [0.791, 0.839] | 9.858 | fallback: 2.6% |

McNemar paired comparison:

| Task | Both Correct | Old Only | Calibrated Only | Neither | Net | p-value |
|---|---:|---:|---:|---:|---:|---:|
| WinoGrande | 711 | 25 | 45 | 219 | +20 | 0.0225 |
| CommonsenseQA | 816 | 0 | 1 | 183 | +1 | 1.0000 |

这给出主结果：

- WinoGrande: calibrated 相比 old adaptive 在大样本成对测试中显著更稳。
- CommonsenseQA: calibrated 不牺牲准确率，且比 old adaptive 更省平均调用。

## 4. 主方法设计

### 4.1 Forward Label Probe

构造 prompt:

```text
{question and choices}
Final answer: [MASK]
```

只在 legal labels 上归一化，得到：

```text
p(A | context), p(B | context), ...
```

理论解释：这是 masked diffusion model 在强约束标签空间上的边缘后验近似。

工程用途：

- binary task 上低置信度直接 fallback。
- cheap scout 与 probe top label 不一致时，判为 reverse trajectory risk。
- multi-choice task 上不盲目 fallback，需要校准。

### 4.2 Cheap Reverse Scout

先跑 8-step reverse diffusion，得到：

- final label；
- output validity；
- selected token confidence；
- 与 probe top label 是否一致。

Scout 的功能不是最终答案本身，而是风险探测。

### 4.3 Calibrated Route Policy

当前主策略：

```text
if task is binary:
    if probe_top_prob < 0.70:
        use 32-step
    else:
        run 8-step scout
        if scout invalid or scout disagrees with probe:
            use 32-step
        else:
            accept 8-step
else:
    run 8-step scout
    if scout invalid:
        use 32-step
    else:
        accept 8-step
```

为什么 multi-choice 不用 disagreement fallback：

- 500/1000 样本结果显示，CommonsenseQA 上 disagreement fallback 会过度保守。
- calibrated 版本保留 answer-contract 风险，但移除多选 disagreement 的过度 fallback。

### 4.4 Negative Ablation: 16-step Intermediate

已测三层 route:

```text
8 -> 16 -> 32
```

初步结果不优：

- WinoGrande 准确率无明显提升，调用增加。
- CommonsenseQA 也没有收益。

报告中应作为负结果保留：不是所有“更细 budget”都有价值，真正有效的是风险信号与任务校准。

## 5. 错例与轨迹分析

### 5.1 错误分类

按样本成对比较划分：

- `saved_by_controller`: old adaptive 错，calibrated 对。
- `lost_by_controller`: old adaptive 对，calibrated 错。
- `both_wrong`: 两者都错。
- `both_correct`: 两者都对。
- `baseline_only`: 32-step 对，controller 错。
- `controller_only`: controller 对，32-step 错。

### 5.2 轨迹记录指标

对 trace 子集记录：

- 每隔 `trace_stride=2` 的 decoded final segment。
- legal label 首次出现 step。
- final label 是否反复跳变。
- 是否 late correction。
- 是否 late corruption。
- 最终 route 与 correctness。

目标图表：

- first-final-step histogram by task。
- route distribution by task。
- confidence vs accuracy calibration curve。
- accuracy-cost Pareto scatter。
- error taxonomy stacked bar。

### 5.3 行为边界解释

WinoGrande:

- 二选一、语义绑定、代词消解；
- cheap decode 容易在局部语义上早锁；
- forward-aware fallback 有必要。

CommonsenseQA:

- 五选一、常识多选；
- 8-step 通常足够进入正确标签 basin；
- probe-scout disagreement 不一定代表错误，过度 fallback 只是浪费。

PubMedQA:

- yes/no/maybe 分布偏置明显；
- 8-step 容易偏向 maybe；
- 应作为高风险任务，默认高预算或需要专门校准。

C-Eval / domain knowledge:

- 错误主要来自知识缺口而非采样预算；
- controller 不是主要解，后续可用 RAG 或领域提示增强。

## 6. 实验矩阵

### 6.1 Main Results

| Experiment | Tasks | N | Purpose | Status |
|---|---|---:|---|---|
| Cross-model typed final-label | 6 tasks | 100 each | LLaDA vs AR vs LoRA | Done |
| 500 held-out controller | Wino/CQA | 500 each | Full four-way comparison | Done |
| 1000 controller scale-up | Wino/CQA | 1000 each | Larger-sample validation | Done |
| 200 trace subset | Wino/CQA | 200 each | Stabilization/hitting-time analysis | Done |

### 6.2 Ablations

| Ablation | Purpose | Expected Use |
|---|---|---|
| Old adaptive vs calibrated | Show risk signal matters | Main contrast |
| Forward-aware vs calibrated | Show task calibration reduces over-fallback | Main contrast |
| 8/16/32 fixed budget | Locate speed/accuracy frontier | Supporting |
| 16-step intermediate route | Negative result | Honest ablation |
| Multi disagreement ignore vs fallback | Explain CQA calibration | Key ablation |
| Binary threshold sweep | Check robustness of 0.70 | Optional |

### 6.3 Statistical Protocol

- Accuracy: Wilson 95% confidence interval。
- Pairwise method comparison: McNemar exact/binomial test。
- Cost: average forward calls and wall-clock seconds。
- Route behavior: route frequency and fallback rate。
- Trace: hitting time distribution, not only examples。

## 7. 工程产物

Remote scripts:

```text
/data/llada_eval/scripts/eval_llada_adaptive_router.py
/data/llada_eval/scripts/eval_llada_forward_aware_router.py
/data/llada_eval/scripts/eval_llada_budget_controller.py
/data/llada_eval/scripts/analyze_llada_error_trajectory.py
```

Local mirrored scripts:

```text
/Users/thomaswang/Documents/New project/scripts/eval_llada_adaptive_router.py
/Users/thomaswang/Documents/New project/scripts/eval_llada_forward_aware_router.py
/Users/thomaswang/Documents/New project/scripts/eval_llada_budget_controller.py
/Users/thomaswang/Documents/New project/scripts/analyze_llada_error_trajectory.py
```

Key result files:

```text
/data/llada_eval/results/domain_shift/task_aware/llada8b_calibrated_controller_wino_cqa_limit1000_seed23.json
/data/llada_eval/results/domain_shift/task_aware/llada8b_adaptive_router_wino_cqa_limit1000_seed23.json
/data/llada_eval/results/domain_shift/task_aware/llada8b_calibrated_controller_trace_wino_cqa_limit200_seed23.json
/data/llada_eval/results/domain_shift/task_aware/error_trajectory_analysis_trace200/
```

Local reports:

```text
/Users/thomaswang/Documents/New project/LLaDA_Calibrated_Controller_Report.md
/Users/thomaswang/Documents/New project/LLaDA_Error_Trajectory_Analysis_Plan.md
/Users/thomaswang/Documents/New project/LLaDA_Research_Grade_Execution_Plan.md
```

## 8. 最终报告结构

### Section 1: Motivation

LLaDA 的非自回归 masked diffusion inference 提供了 AR 模型没有的中间轨迹和预算控制空间。问题不是“能不能更快”，而是“哪些样本需要更长的反向扩散过程”。

### Section 2: Method

介绍 forward probe、cheap scout、calibrated route policy 和 trajectory logger。

### Section 3: Theory

用 CTMC / hitting time / posterior uncertainty 解释为什么 route decision 有理论支撑。

### Section 4: Experiments

展示 6-task baseline、500 held-out、1000 scale-up、ablation 和 trajectory analysis。

### Section 5: Error Analysis

展示 saved/lost cases、Wino vs CQA 的行为差异、PubMedQA/C-Eval 的边界。

### Section 6: Limitations

- 没有改 backbone 架构。
- 当前 controller 主要验证在选择题/final-label 格式。
- 对知识缺口类任务不能靠 sampler 解决。
- 未来可以把 rule-based controller 训练成轻量 risk predictor。

### Section 7: Takeaway

Masked diffusion LM 的价值不只在生成方式不同，还在于它暴露了一个可控的反向随机过程。我们的工作把这个过程变成可测、可解释、可优化的工程系统。

## 9. 接下来立即执行

1. 生成 1000 样本 calibrated vs old adaptive 的错误分类表。
2. 用 trace200 更新 hitting-time 统计和轨迹图。
3. 更新最终中文报告，保留负结果。
4. 如果时间允许，补一个 threshold sweep:

```text
binary threshold in {0.60, 0.65, 0.70, 0.75, 0.80}
```

验收标准：

- 主结果至少包含 1000 样本对照。
- trajectory analysis 不低于 200 样本子集。
- 每个 claim 都能追溯到脚本和 json 结果。
- 报告里明确区分“已验证结果”和“后续工作”。

## 10. Trace200 结果补充

200 样本轨迹分析已经完成：

| Task | Trace N | Accuracy | Avg Calls | Mean First Final Step | Mean Flip Count | Late Instability |
|---|---:|---:|---:|---:|---:|---:|
| WinoGrande | 200 | 0.750 | 17.040 | 15.89 | 0.015 | 0.000 |
| CommonsenseQA | 200 | 0.815 | 9.000 | 7.44 | 0.030 | 0.030 |

解释：

- WinoGrande 平均更晚进入 final-label basin，因此需要更保守的 forward-aware route。
- CommonsenseQA 平均在更早 step 已经进入合法 final label，且 calibrated controller 让全部样本走 8-step，说明多选 disagreement fallback 在该任务上属于过度保守。
- 这个结果把“节省时间”提升成“任务依赖反向扩散 hitting time 不同”的理论化解释。
