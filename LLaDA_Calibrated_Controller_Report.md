# LLaDA Calibrated Risk Controller 更新报告

## 1. 本轮改进

在上一版 Forward-aware Router 的基础上，本轮接入两个工程增强：

1. Calibrated label-space risk controller
   - binary task：保留严格风险门控，低 forward confidence 或 probe/scout disagreement 时 fallback 到 32-step；
   - multi-choice task：经 calibration 发现 probe/scout disagreement fallback 对 CommonsenseQA 不增益，因此只在非法 label 时 fallback，不再对 disagreement 过度反应。

2. Trajectory recorder
   - 可选记录每个 denoising step 的 filled ratio、当前可解析 label、label validity；
   - 用于错例分析和解释反向扩散轨迹的稳定性边界；
   - 主实验默认关闭，避免影响推理成本。

实现文件：

- `/Users/thomaswang/Documents/New project/scripts/eval_llada_budget_controller.py`
- 服务器：`/data/llada_eval/scripts/eval_llada_budget_controller.py`

## 2. 为什么不是简单手调

我先试了两个三档预算变体：

- binary 中置信样本先跑 16-step；
- disagreement 时先插入 16-step，再决定是否 32-step。

100 样本筛选结果显示，这两个变体没有超过 forward-aware：WinoGrande 精度为 0.75，calls 约 20.5；CommonsenseQA 插入 16 只会增加 calls。因此没有把它作为主线。

真正有效的点来自 calibration/test split 分析：

- WinoGrande：需要 binary risk gate 才能恢复 32-step baseline；
- CommonsenseQA：fast 8-step 本身已经达到 0.812，disagreement fallback 没有精度收益，反而增加计算。

所以最终策略不是“所有任务都多 fallback”，而是按 label-space 和风险收益校准 fallback。

## 3. 主实验

命令：

```bash
python scripts/eval_llada_budget_controller.py \
  --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct \
  --tasks winogrande,commonsenseqa \
  --limit 500 \
  --seed 23 \
  --binary-medium-threshold 0.70 \
  --multi-disagreement-policy ignore \
  --out results/domain_shift/task_aware/llada8b_calibrated_controller_wino_cqa_limit500_seed23.json
```

结果文件：

- `/data/llada_eval/results/domain_shift/task_aware/llada8b_calibrated_controller_wino_cqa_limit500_seed23.json`

## 4. 结果

| Method | Task | Accuracy | Avg calls | Seconds | Route |
|---|---:|---:|---:|---:|---|
| 32-step baseline | WinoGrande | 0.756 | 32.000 | 773.8 | all 32 |
| old adaptive | WinoGrande | 0.732 | 9.462 | 205.3 | mostly 8 |
| forward-aware | WinoGrande | 0.756 | 17.768 | 429.8 | 64.0% 8, 36.0% fallback |
| calibrated controller | WinoGrande | 0.756 | 17.768 | 429.3 | 64.0% 8, 34.4% 32, 1.6% 8->32 |
| 32-step baseline | CommonsenseQA | 0.810 | 32.000 | 774.2 | all 32 |
| old adaptive | CommonsenseQA | 0.810 | 9.924 | 216.1 | mostly 8 |
| forward-aware | CommonsenseQA | 0.812 | 10.856 | 263.7 | 94.2% 8, 5.8% fallback |
| calibrated controller | CommonsenseQA | 0.812 | 9.000 | 218.1 | all 8 |

Calibrated controller 的收益：

- WinoGrande：保持 forward-aware 与 32-step baseline 同精度；
- CommonsenseQA：保持 0.812 精度，同时从 forward-aware 的 10.856 calls 降到 9.000 calls；
- 相比 32-step baseline，CommonsenseQA calls 降低 71.9%，WinoGrande calls 降低 44.5%。

## 5. 配对统计

| Task | Comparison | Ref only | Calibrated only | Net | McNemar p |
|---|---:|---:|---:|---:|---:|
| WinoGrande | vs 32-step | 2 | 2 | 0 | 1.0000 |
| WinoGrande | vs forward-aware | 0 | 0 | 0 | 1.0000 |
| WinoGrande | vs old adaptive | 14 | 26 | +12 | 0.0807 |
| CommonsenseQA | vs 32-step | 3 | 4 | +1 | 1.0000 |
| CommonsenseQA | vs forward-aware | 2 | 2 | 0 | 1.0000 |
| CommonsenseQA | vs old adaptive | 0 | 1 | +1 | 1.0000 |

这说明 calibrated controller 是 forward-aware 的 Pareto 改进：精度等价，CQA 计算更少。

## 6. Trajectory Recorder 验证

命令：

```bash
python scripts/eval_llada_budget_controller.py \
  --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct \
  --tasks winogrande,commonsenseqa \
  --limit 12 \
  --seed 23 \
  --binary-medium-threshold 0.70 \
  --multi-disagreement-policy ignore \
  --collect-trace \
  --trace-stride 2 \
  --out results/domain_shift/task_aware/llada8b_calibrated_controller_trace_wino_cqa_limit12_seed23.json
```

结果文件：

- `/data/llada_eval/results/domain_shift/task_aware/llada8b_calibrated_controller_trace_wino_cqa_limit12_seed23.json`

观察到的轨迹现象：

- 8-step fast route 中，CQA 往往在 step 6 已经能解析出稳定合法 label；
- 32-step WinoGrande 中，有些样本在 step 30 仍解析出非法 label，最后 step 32 才回到合法 label；
- 这说明不同任务的 label stabilization time 不同，可以支持“task-aware reverse budget allocation”的理论叙事。

## 7. 最终可讲的创新点

这版可以叫：

> Calibrated Forward-aware Risk Controller for Masked Diffusion Language Models.

核心不是“我让 LLaDA 更快了”，而是：

- 用模型自身 forward posterior 估计样本级不确定性；
- 用 probe/scout consistency 判断反向轨迹是否稳定；
- 用 calibration 判断某个任务/label-space 上 fallback 是否真的有风险收益；
- 用 trajectory recorder 提供错例驱动的轨迹证据。

更适合报告里的 claim：

> We allocate reverse diffusion compute according to calibrated posterior risk, rather than uniformly accelerating every denoising step.
