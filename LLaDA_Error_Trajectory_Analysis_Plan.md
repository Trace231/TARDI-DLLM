# LLaDA 错例与轨迹分析完整计划

## 0. 目标

本阶段目标不是再做一个新 sampler，而是把已有工程改进解释扎实：

> LLaDA 的 reverse diffusion budget 应该由样本级 posterior uncertainty 和 trajectory stability 决定，而不是统一固定步数或盲目压缩步数。

最终交付：

- 大样本结果表；
- 错例分型；
- posterior confidence vs accuracy 图；
- route/risk-budget 图；
- trajectory stabilization 图；
- 可直接放进报告/PPT 的结论段。

## 1. 实验样本量

### 主任务

使用更大样本量：

- WinoGrande：`n=1000`
- CommonsenseQA：`n=1000`

如果数据集 split 不足 1000，则使用可用最大样本数。

### 对照方法

保留四个方法：

1. `32-step uniform baseline`
2. `old adaptive router`
3. `forward-aware router`
4. `calibrated controller`

其中 500 样本已有完整结果；1000 样本阶段优先跑：

- calibrated controller；
- old adaptive router；
- 若时间允许，再补 32-step baseline。

原因：calibrated 与 old adaptive 是行为边界的核心对比；32-step baseline 成本较高，可作为最终确认。

## 2. 错例分型

对每个样本按方法结果分成：

| Type | 定义 | 解释 |
|---|---|---|
| easy-fast | 8-step 正确，32-step 也正确 | posterior 尖锐，快速路径足够 |
| repaired-by-fallback | 8-step 错，fallback/32-step 正确 | 需要更多 reverse budget |
| harmed-by-fallback | 8-step 正确，fallback/32-step 错 | 保守路径并不总是更优 |
| hard-both-wrong | fast 与 32-step 都错 | 模型知识/推理能力边界 |
| disagreement-risk | probe top label 与 scout pred 不一致 | 反向轨迹不稳定信号 |
| low-confidence-risk | probe top probability 低 | posterior uncertainty 高 |

重点不是列很多错例，而是统计每类比例，并挑 2-3 个代表例子。

## 3. 图表

### Figure 1: Probe Confidence Calibration

横轴：forward probe top probability bucket  
纵轴：accuracy  
分任务画 WinoGrande / CommonsenseQA。

目的：

- 证明 probe confidence 不是装饰，它确实预测准确率；
- 支撑理论说法：`p_theta(x_0^label | x_t=[MASK], context)` 是有效 posterior risk signal。

### Figure 2: Route Distribution

统计 calibrated controller 的 route：

- `8`
- `32`
- `8->32`

目的：

- 显示 WinoGrande 需要明显更多 conservative budget；
- CommonsenseQA 大多数样本可以安全走 fast path；
- 这就是 task-aware / label-space-aware compute allocation。

### Figure 3: Accuracy-Cost Pareto

横轴：avg forward calls  
纵轴：accuracy  
点：32-step、old adaptive、forward-aware、calibrated。

目的：

- old adaptive：很快但 Wino 掉点；
- forward-aware：修复 Wino，但 CQA 有冗余 fallback；
- calibrated：保留精度并减少冗余 fallback；
- 说明我们不是只追求快，而是在 Pareto frontier 上移动。

### Figure 4: Trajectory Stabilization

使用 `--collect-trace` 对每个任务至少 `n=200` 跑 trajectory recorder。

统计：

- first valid label step；
- final label first appears step；
- label flip count；
- invalid-to-valid transition step；
- final two steps是否仍发生 label change。

目的：

- 证明不同任务 label stabilization time 不同；
- 支撑“反向轨迹稳定边界”叙事；
- 给理论老师一个更像离散随机过程/吸收过程的解释入口。

## 4. 轨迹指标定义

对每个 sample 的 trace：

```text
first_valid_step = min step where parsed label is legal
first_final_step = min step where parsed label == final_pred
flip_count = number of times parsed label changes among legal labels
late_instability = whether parsed label changes after 75% steps
valid_ratio_curve = legal_label_indicator over denoising steps
```

这些指标能对应到理论语言：

- first valid step：进入合法 label subset 的 hitting time；
- first final step：到达最终 label basin 的 hitting time；
- flip count：trajectory instability；
- late instability：discretization/schedule risk。

## 5. 预期结论

我们希望得到三条结论：

1. Posterior uncertainty predicts risk
   - 低 probe confidence 样本准确率明显低；
   - binary WinoGrande 对低 confidence 更敏感。

2. Trajectory stabilization is task-dependent
   - CommonsenseQA 的 label 往往较早稳定；
   - WinoGrande 更依赖 late denoising，部分样本最后几步才合法/稳定。

3. Calibrated controller improves risk-cost tradeoff
   - old adaptive 暴露行为边界；
   - forward-aware 修复 Wino；
   - calibrated 去掉 CQA 上无收益 fallback，成为更干净的 Pareto 点。

## 6. 工程步骤

1. 写分析脚本
   - 输入多个 JSON result；
   - 输出 CSV summary；
   - 输出 PNG 图；
   - 输出 markdown 分析摘要。

2. 跑 1000 样本 calibrated controller
   - WinoGrande + CommonsenseQA；
   - 记录 route、probe、correct。

3. 跑 1000 样本 old adaptive router
   - 用于确认 old adaptive 的行为边界在更大样本仍存在。

4. 跑 trace recorder
   - 每任务 `n=200`；
   - `trace_stride=1` 或 `2`；
   - 如果全量 trace 太慢，先跑 `n=100`，再扩到 `n=200`。

5. 生成图表与报告
   - `confidence_accuracy.png`
   - `route_distribution.png`
   - `accuracy_cost_pareto.png`
   - `trajectory_stabilization.png`
   - `LLaDA_Error_Trajectory_Analysis_Report.md`

## 7. 风险控制

- 不把 1000 样本结果和 500 样本结果混在同一个主表里；
- 若 1000 样本 baseline 32-step 来不及跑，则只说“500 paired baseline confirmed”，1000 作为 controller stability check；
- 不声称模型架构改进，只说 inference-time controller / sampler policy；
- 不声称统计显著提升，除非 McNemar 或 bootstrap 支持；
- 对三档预算负结果如实保留，用来说明我们不是只挑好看的结果。
