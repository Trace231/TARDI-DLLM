# LLaDA / dLLM Task-aware Inference: Final Data Pack

> This file collects the experimental numbers that are worth reporting: main 1000-sample validation, threshold robustness, step-budget exploration, all-dataset comparison, AR/DDM comparison, LoRA audit, trajectory analysis, and negative boundary cases.

## 1. One-line Results

- **Main improved method (`ours_v3plus`)**: macro accuracy **0.658**, average calls **17.56** over 11 datasets at limit=50.
- It matches fixed 32-step macro accuracy (**0.658**) while reducing calls from **32.00** to **17.56** (**45.1% fewer calls**).
- Compared with the previous refinement controller, calls drop from **26.92** to **17.56** (**34.8% fewer calls**) at the same macro accuracy.
- JYS-like 16-step is cheaper (**16.00** calls) but lower accuracy (**0.644**), so `ours_v3plus` is the stronger accuracy-cost point.

## 2. Main 1000-sample Validation: WinoGrande and CommonsenseQA

| Task | Method | Acc | n | Avg calls | Seconds | Route rates |
| --- | --- | --- | --- | --- | --- | --- |
| WinoGrande | 32step | 0.756 | 1000 | 32.000 | 1555.6 | {"fixed": 1.0} |
| CommonsenseQA | 32step | 0.819 | 1000 | 32.000 | 1550.2 | {"fixed": 1.0} |
| WinoGrande | old_adaptive | 0.736 | 1000 | 9.363 | 408.5 | {"fallback": 0.011, "fixed": 0.989} |
| CommonsenseQA | old_adaptive | 0.816 | 1000 | 9.858 | 432.3 | {"fallback": 0.026, "fixed": 0.974} |
| WinoGrande | forward_aware | 0.756 | 1000 | 17.560 | 847.7 | {"accepted_fast": 0.648, "post_fallback": 0.014, "pre_fallback": 0.338} |
| CommonsenseQA | forward_aware | 0.818 | 1000 | 10.824 | 523.6 | {"accepted_fast": 0.943, "post_fallback": 0.057} |
| WinoGrande | calibrated | 0.756 | 1000 | 17.560 | 852.4 | {"32": 0.338, "8": 0.648, "8->32": 0.014} |
| CommonsenseQA | calibrated | 0.817 | 1000 | 9.064 | 442.0 | {"8": 0.998, "8->32": 0.002} |

### Paired McNemar Tests

| Task | Comparison | n | Candidate only | Reference only | Net | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| WinoGrande | calibrated vs 32step | 1000 | 3 | 3 | 0 | 1.0000 |
| CommonsenseQA | calibrated vs 32step | 1000 | 6 | 8 | -2 | 0.7905 |
| WinoGrande | calibrated vs old_adaptive | 1000 | 45 | 25 | 20 | 0.0225 |
| CommonsenseQA | calibrated vs old_adaptive | 1000 | 1 | 0 | 1 | 1.0000 |
| WinoGrande | calibrated vs forward_aware | 1000 | 0 | 0 | 0 | 1.0000 |
| CommonsenseQA | calibrated vs forward_aware | 1000 | 3 | 4 | -1 | 1.0000 |

**Interpretation.** On WinoGrande, calibrated routing matches full 32-step accuracy and significantly improves over the old adaptive router. On CommonsenseQA, calibrated routing is statistically indistinguishable from full 32-step while using far fewer calls.

## 3. Threshold Robustness Sweep

| Threshold | Task | Acc | Avg calls | Route rates |
| --- | --- | --- | --- | --- |
| 0.6 | WinoGrande | 0.754 | 15.704 | {"32": 0.194, "8": 0.742, "8->32": 0.064} |
| 0.6 | CommonsenseQA | 0.812 | 9.000 | {"8": 1.0} |
| 0.65 | WinoGrande | 0.756 | 16.456 | {"32": 0.268, "8": 0.7, "8->32": 0.032} |
| 0.65 | CommonsenseQA | 0.812 | 9.000 | {"8": 1.0} |
| 0.7 | WinoGrande | 0.756 | 17.768 | {"32": 0.344, "8": 0.64, "8->32": 0.016} |
| 0.7 | CommonsenseQA | 0.812 | 9.000 | {"8": 1.0} |
| 0.75 | WinoGrande | 0.756 | 20.568 | {"32": 0.466, "8": 0.522, "8->32": 0.012} |
| 0.75 | CommonsenseQA | 0.812 | 9.000 | {"8": 1.0} |
| 0.8 | WinoGrande | 0.756 | 22.936 | {"32": 0.578, "8": 0.42, "8->32": 0.002} |
| 0.8 | CommonsenseQA | 0.812 | 9.000 | {"8": 1.0} |

**Interpretation.** Around 0.65-0.75, WinoGrande accuracy is stable while cost increases smoothly with threshold. CommonsenseQA remains stable and mostly accepts the fast path. This supports that the 0.70 threshold is not a single-point artifact.

## 4. All-dataset Method Comparison, limit=50

### Macro Average

| Method | Tasks | Macro Acc | Macro Avg Calls | Total seconds |
| --- | --- | --- | --- | --- |
| ours_v3plus | 11 | 0.658 | 17.556 | 409.0 |
| ours_v3plus_aggressive_pubmed | 11 | 0.656 | 16.260 | 370.8 |
| refinement | 11 | 0.658 | 26.916 | 1727.1 |
| fixed32 | 11 | 0.658 | 32.000 | 1848.8 |
| jys_middle16 | 11 | 0.644 | 16.000 | 1052.2 |
| prophet | 11 | 0.647 | 30.444 | 1589.7 |

### Per-task Accuracy / Average Calls

| Task | Ours v3plus | Fixed 32 | JYS-like 16 | Prophet-like | Old refinement |
| --- | --- | --- | --- | --- | --- |
| mmlu_pro | 0.400/18.46 | 0.400/32.00 | 0.400/16.00 | 0.400/31.60 | 0.400/32.64 |
| pubmedqa | 0.640/32.00 | 0.620/32.00 | 0.620/16.00 | 0.640/32.00 | 0.640/33.00 |
| ceval_computer_network | 0.560/14.92 | 0.560/32.00 | 0.560/16.00 | 0.560/32.00 | 0.560/33.00 |
| sciq | 0.860/9.52 | 0.860/32.00 | 0.860/16.00 | 0.860/32.00 | 0.860/33.00 |
| winogrande | 0.760/13.32 | 0.760/32.00 | 0.740/16.00 | 0.760/32.00 | 0.760/13.32 |
| commonsenseqa | 0.800/9.82 | 0.800/32.00 | 0.800/16.00 | 0.800/32.00 | 0.800/10.04 |
| arc_challenge | 0.860/9.52 | 0.860/32.00 | 0.860/16.00 | 0.860/32.00 | 0.860/33.00 |
| hellaswag | 0.740/10.48 | 0.740/32.00 | 0.740/16.00 | 0.740/32.00 | 0.740/33.00 |
| boolq | 0.840/11.08 | 0.860/32.00 | 0.860/16.00 | 0.860/32.00 | 0.840/11.08 |
| gsm8k | 0.780/32.00 | 0.780/32.00 | 0.640/16.00 | 0.640/15.84 | 0.780/32.00 |
| drop_span | 0.000/32.00 | 0.000/32.00 | 0.000/16.00 | 0.000/31.44 | 0.000/32.00 |

**Interpretation.** The optimized controller keeps fixed-32 accuracy on MMLU-Pro, C-Eval, SciQ, WinoGrande, CommonsenseQA, ARC, HellaSwag, GSM8K, and DROP under this sample. It also preserves the PubMedQA gain seen in refinement while avoiding the 33-call probe overhead. The largest cost wins appear on 4-choice tasks and MMLU-Pro after enabling risk-gated fast paths.

## 5. Effect of Diffusion Budget / Step Count

### 5.1 JYS-like 16-step vs Fixed 32-step vs Ours

| Task | JYS16 Acc | Fixed32 Acc | 32-16 Delta | Ours Acc | Ours Calls | Ours Routes |
| --- | --- | --- | --- | --- | --- | --- |
| mmlu_pro | 0.400 | 0.400 | +0.000 | 0.400 | 18.46 | {"32": 0.08, "8": 0.2, "8->16": 0.26, "8->24": 0.3, "8->32": 0.16} |
| pubmedqa | 0.620 | 0.620 | +0.000 | 0.640 | 32.00 | {"32": 1.0} |
| ceval_computer_network | 0.560 | 0.560 | +0.000 | 0.560 | 14.92 | {"8": 0.34, "8->16": 0.32, "8->24": 0.34} |
| sciq | 0.860 | 0.860 | +0.000 | 0.860 | 9.52 | {"8": 0.82, "8->16": 0.14, "8->24": 0.04} |
| winogrande | 0.740 | 0.760 | +0.020 | 0.760 | 13.32 | {"32": 0.04, "8": 0.62, "8->16": 0.26, "8->24": 0.08} |
| commonsenseqa | 0.800 | 0.800 | +0.000 | 0.800 | 9.82 | {"8": 0.78, "8->16": 0.18, "8->24": 0.04} |
| arc_challenge | 0.860 | 0.860 | +0.000 | 0.860 | 9.52 | {"8": 0.82, "8->16": 0.14, "8->24": 0.04} |
| hellaswag | 0.740 | 0.740 | +0.000 | 0.740 | 10.48 | {"8": 0.7, "8->16": 0.24, "8->24": 0.06} |
| boolq | 0.860 | 0.860 | +0.000 | 0.840 | 11.08 | {"32": 0.02, "8": 0.84, "8->16": 0.08, "8->24": 0.06} |
| gsm8k | 0.640 | 0.780 | +0.140 | 0.780 | 32.00 | {"32": 1.0} |
| drop_span | 0.000 | 0.000 | +0.000 | 0.000 | 32.00 | {"32": 1.0} |

### 5.2 Pure 8-step vs Pure 32-step Where Available

| Task | 8-step Acc | 32-step Acc | 32-8 Delta | Calibrated Acc | Calibrated Calls |
| --- | --- | --- | --- | --- | --- |
| arc_challenge | 0.823 | 0.857 | +0.033 | 0.857 | 33.00 |
| hellaswag | 0.760 | 0.780 | +0.020 | 0.780 | 33.00 |
| boolq | 0.843 | 0.867 | +0.023 | 0.853 | 12.87 |
| gsm8k | 0.320 | 0.580 | +0.260 | - | - |
| pubmedqa | 0.557 | 0.660 | +0.103 | 0.680 | 33.00 |
| ceval_computer_network | 0.585 | 0.585 | +0.000 | 0.585 | 33.00 |

**Interpretation.** Increasing steps helps in some tasks, especially GSM8K and PubMedQA; it barely matters on C-Eval in this slice; and it gives small but visible gains on ARC/HellaSwag/BoolQ. This justifies a controller rather than a universal 8-step or 16-step rule.

### 5.3 Strict Fixed-step Sweep, limit=20 per dataset

This is an independent fixed-step sweep, not a trace-derived intermediate result. For each dataset, the same 20 examples are evaluated at steps `4,8,12,16,20,24,28,32`. GSM8K and DROP use `gen_length=96`; other tasks use `gen_length=32`.

| Task | 4 steps | 8 steps | 12 steps | 16 steps | 20 steps | 24 steps | 28 steps | 32 steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmlu_pro | 0.45 | 0.45 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| pubmedqa | 0.30 | 0.60 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| ceval_computer_network | 0.50 | 0.55 | 0.50 | 0.50 | 0.60 | 0.60 | 0.60 | 0.60 |
| sciq | 0.95 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| winogrande | 0.70 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| commonsenseqa | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| arc_challenge | 0.75 | 0.80 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| hellaswag | 0.55 | 0.60 | 0.65 | 0.65 | 0.60 | 0.60 | 0.60 | 0.60 |
| boolq | 0.50 | 0.65 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| gsm8k | 0.20 | 0.55 | 0.55 | 0.50 | 0.55 | 0.60 | 0.55 | 0.65 |
| drop_span | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Key observations.** C-Eval improves from 0.50 to 0.60 after 20 steps; WinoGrande reaches 0.75 by 8 steps and then plateaus; CommonsenseQA is already stable at 0.85 from 4 steps; ARC reaches 0.85 by 12 steps; HellaSwag improves up to 12-16 steps then fluctuates; GSM8K strongly benefits from more steps, from 0.20 at 4 steps to 0.65 at 32 steps; DROP remains 0.00 across all steps, indicating an evaluation/task-format boundary rather than a denoising-budget issue.

Data file: `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/step_sweep_limit20_4to32/tables/step_sweep_4to32_by_dataset_limit20_seed23.csv`
### 5.3 Fine-grained Trajectory Checkpoints

These numbers are computed from the collected controller traces, not from independent fixed-step reruns. `accuracy_invalid_as_wrong` treats empty / invalid intermediate labels as wrong; `accuracy_among_valid` conditions on samples that already emitted a valid label at that step.

| Task | Step | n trace | Valid rate | Acc, invalid as wrong | Acc among valid | Empty rate |
| --- | --- | --- | --- | --- | --- | --- |
| commonsenseqa | 1 | 200 | 0.000 | 0.000 | - | 1.000 |
| commonsenseqa | 2 | 200 | 0.000 | 0.000 | - | 1.000 |
| commonsenseqa | 4 | 200 | 0.000 | 0.000 | - | 1.000 |
| commonsenseqa | 6 | 200 | 0.995 | 0.260 | 0.261 | 0.005 |
| commonsenseqa | 8 | 200 | 1.000 | 0.815 | 0.815 | 0.000 |
| winogrande | 1 | 200 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 2 | 200 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 4 | 200 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 6 | 200 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 8 | 200 | 0.670 | 0.540 | 0.806 | 0.330 |
| winogrande | 10 | 66 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 12 | 66 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 14 | 66 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 16 | 66 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 18 | 66 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 20 | 66 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 22 | 66 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 24 | 66 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 26 | 66 | 0.000 | 0.000 | - | 1.000 |
| winogrande | 28 | 66 | 0.712 | 0.015 | 0.021 | 0.288 |
| winogrande | 30 | 66 | 1.000 | 0.000 | 0.000 | 0.000 |
| winogrande | 32 | 66 | 1.000 | 0.636 | 0.636 | 0.000 |

**Interpretation.** CommonsenseQA reaches valid labels by step 6 and jumps to high accuracy by step 8. WinoGrande remains empty until step 8 for many samples and needs later fallback traces for hard cases, supporting the task-dependent hitting-time story. The fallback subset after step 8 is high-risk by construction, so its later-step accuracy should not be read as the full-task fixed-step accuracy.

## 6. AR vs DDM Coverage Comparison

| Task | Qwen2.5-7B Acc | LLaDA 32 Acc | LLaDA 8 Acc | LLaDA calibrated Acc | LLaDA32 - Qwen |
| --- | --- | --- | --- | --- | --- |
| arc_challenge | 0.890 | 0.857 | 0.823 | 0.857 | -0.033 |
| hellaswag | 0.850 | 0.780 | 0.760 | 0.780 | -0.070 |
| boolq | 0.823 | 0.867 | 0.843 | 0.853 | +0.043 |
| gsm8k | 0.770 | 0.580 | 0.320 | - | -0.190 |

**Interpretation.** LLaDA is competitive or stronger on BoolQ in this sample, but Qwen is stronger on ARC, HellaSwag, and GSM8K. This supports the boundary claim: inference control can allocate denoising budget, but it cannot fully replace model knowledge or AR-style reasoning strengths.

## 7. LoRA Gain Audit

| Task | Qwen Base | Qwen LoRA | AR Gain | LLaDA Base | LLaDA LoRA controlled | Controlled Gain | LLaDA LoRA original | Original Gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmlu_pro | 0.380 | 0.400 | +0.020 | 0.350 | 0.350 | +0.000 | 0.440 | +0.090 |
| pubmedqa | 0.510 | 0.770 | +0.260 | 0.750 | 0.750 | +0.000 | 0.720 | -0.030 |
| ceval_computer_network | 0.730 | 0.750 | +0.020 | 0.580 | 0.570 | -0.010 | 0.540 | -0.040 |
| sciq | 0.970 | 0.970 | +0.000 | 0.920 | 0.920 | +0.000 | 0.880 | -0.040 |
| winogrande | 0.630 | 0.720 | +0.090 | 0.630 | 0.630 | +0.000 | 0.780 | +0.150 |
| commonsenseqa | 0.820 | 0.860 | +0.040 | 0.870 | 0.870 | +0.000 | 0.800 | -0.070 |

**Interpretation.** LoRA is useful as an auxiliary adaptation baseline, but it is not the central contribution here. In the controlled small-scale LoRA audit, Qwen gains are often larger and more stable. LLaDA controlled LoRA gains are limited or mixed; the earlier original DDM LoRA can improve some tasks, but it is not consistently stronger than inference-loop control and is sensitive to protocol. This separates model adaptation from inference-loop control.

## 8. Trajectory / Error Analysis

| Task | n | Mean first-final step | Mean flip count | Late instability rate |
| --- | --- | --- | --- | --- |
| CommonsenseQA | 200 | 7.44 | 0.715 | 0.715 |
| WinoGrande | 200 | 15.89 | 0.345 | 0.325 |

### Probe Confidence vs Accuracy

| Task | Confidence bin | n | Accuracy |
| --- | --- | --- | --- |
| CommonsenseQA | [0.0,0.5) | 64 | 0.438 |
| CommonsenseQA | [0.5,0.6) | 57 | 0.579 |
| CommonsenseQA | [0.6,0.7) | 71 | 0.535 |
| CommonsenseQA | [0.7,0.8) | 73 | 0.644 |
| CommonsenseQA | [0.8,0.9) | 89 | 0.730 |
| CommonsenseQA | [0.9,1.0) | 646 | 0.938 |
| WinoGrande | [0.5,0.6) | 175 | 0.663 |
| WinoGrande | [0.6,0.7) | 163 | 0.650 |
| WinoGrande | [0.7,0.8) | 226 | 0.712 |
| WinoGrande | [0.8,0.9) | 221 | 0.778 |
| WinoGrande | [0.9,1.0) | 215 | 0.935 |

**Interpretation.** WinoGrande has a much later mean first-final step than CommonsenseQA, supporting the hitting-time narrative: some tasks require longer reverse diffusion to settle semantically. Confidence bins are monotonic enough to justify using probe uncertainty as a risk signal.

## 9. Boundary Negative Cases

| Task | Method | Acc | n | Avg calls | Seconds | Route rates |
| --- | --- | --- | --- | --- | --- | --- |
| PubMedQA | 8step | 0.557 | 300 | - | 225.9 | {"fixed": 1.0} |
| C-Eval CN | 8step | 0.585 | 171 | - | 66.6 | {"fixed": 1.0} |
| PubMedQA | 32step | 0.660 | 300 | 32.000 | 903.6 | {"fixed": 1.0} |
| C-Eval CN | 32step | 0.585 | 171 | 32.000 | 266.0 | {"fixed": 1.0} |
| PubMedQA | calibrated | 0.680 | 300 | 33.000 | 1012.4 | {"32": 1.0} |
| C-Eval CN | calibrated | 0.585 | 171 | 33.000 | 274.4 | {"32": 1.0} |

**Interpretation.** PubMedQA benefits from more budget, but also exposes label/calibration bias around `maybe`. C-Eval shows a knowledge boundary: more sampling does not fix missing domain knowledge. DROP is a formatting/metric and long-form generation boundary in the current setup.

## 10. Data Files

- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/tables/main_1000_summary.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/tables/paired_mcnemar.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/tables/threshold_sweep.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/tables/boundary_negative_cases.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/coverage_addendum/tables/coverage_summary.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/coverage_addendum/tables/coverage_deltas.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/tables/lora_control_v2.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/tables/trajectory_metrics.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/tables/confidence_accuracy.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/v3_choice_fast/tables/v3plus_macro_comparison_limit50_seed23.csv`
- `/Users/thomaswang/Documents/New project/results/domain_shift/task_aware/solid_v2/v3_choice_fast/tables/v3plus_combined_comparison_limit50_seed23.csv`

## 11. Report-ready Claim

We do not claim to modify the LLaDA backbone. The contribution is an inference-loop controller: a task-shape and sample-risk aware denoising-budget allocator. Empirically, the final v3plus controller preserves fixed-32 accuracy on the current 11-task limit=50 suite while cutting average forward calls by 45.1%. The exploratory step-budget analysis shows why fixed global schedules are suboptimal: some tasks tolerate 8-16 steps, some need 32 steps, and some are bounded by model knowledge rather than sampling budget.