# LLaDA Adaptive Router Held-out 500 验证

## 设置

为避免过拟合，本轮冻结 adaptive router，不再调整规则或阈值，重新采样 `seed=23, limit=500`：

- WinoGrande: 500 samples
- CommonsenseQA: 500 samples

比较方法：

- **Adaptive router**: task-shape policy + confidence/validity fallback
- **32-step uniform baseline**: LLaDA 原始 32-step uniform final-label typed prompt

远端结果文件：

- `/data/llada_eval/results/domain_shift/task_aware/llada8b_adaptive_router_wino_cqa_limit500_seed23.json`
- `/data/llada_eval/results/domain_shift/llada8b_final_label_typed_wino_cqa_limit500_seed23.json`

## 结果

| Task | Method | Accuracy | 95% CI | Time | Avg calls |
|---|---:|---:|---:|---:|---:|
| WinoGrande | Adaptive router | 0.732 | [0.6915, 0.7689] | 205.28s | 9.46 |
| WinoGrande | 32-step uniform | 0.756 | [0.7165, 0.7916] | 773.80s | 32 |
| CommonsenseQA | Adaptive router | 0.810 | [0.7733, 0.8420] | 216.13s | 9.92 |
| CommonsenseQA | 32-step uniform | 0.810 | [0.7733, 0.8420] | 774.17s | 32 |

Paired comparison:

| Task | Both Correct | Baseline Only | Adaptive Only | Neither | Net Gain | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| WinoGrande | 350 | 28 | 16 | 106 | -12 | 0.0961 |
| CommonsenseQA | 402 | 3 | 3 | 92 | 0 | 1.0000 |

## 结论

这轮大样本验证修正了前面 `n=100` 的乐观结论。

1. **CommonsenseQA 是稳的效率提升点**  
   Adaptive router 与 32-step uniform 准确率完全相同，都是 0.810，但耗时从 774.17s 降到 216.13s，约 3.58x 加速；forward calls 从 32 降到约 9.92。

2. **WinoGrande 不能再作为准确率提升点**  
   Adaptive router 为 0.732，32-step uniform 为 0.756，差距 -2.4 个百分点。McNemar p=0.0961，未到 0.05 显著性，但方向上不支持“提升准确率”的主张。

3. **最诚实的主线应调整为 adaptive compute，而不是全面涨点**  
   目前可以说：task-aware/adaptive inference 在 CommonsenseQA 上实现了不降准确率的大幅加速；在 WinoGrande 上存在压缩带来的轻微精度风险，需要更保守的 fallback 或任务特定 confidence calibration。

## 报告口径建议

最终报告不要写“全任务提升准确率”。更好的主张是：

> We identify task-dependent behavior boundaries in LLaDA inference. For CommonsenseQA, a task-shape-aware adaptive router preserves 32-step accuracy while reducing inference computation by about 3.6x. For WinoGrande, aggressive compression reveals a boundary case where semantic coreference decisions are more sensitive to denoising budget.

中文：

> 我们发现 LLaDA 的反向扩散推理存在任务相关行为边界。对于 CommonsenseQA，任务形态感知的 adaptive router 能保持 32-step 准确率，同时将推理成本降低约 3.6 倍；而 WinoGrande 对压缩更敏感，说明共指语义任务需要更保守的反向扩散预算控制。

