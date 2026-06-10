# LoRA Optimization v1 Report

This report evaluates task-balanced and noise-schedule tuned LLaDA LoRA variants against the previous vanilla LoRA and external improved LoRA baselines.

## Macro Results

| Method | Macro Acc. | Correct / N | Delta vs old vanilla |
|---|---:|---:|---:|
| tardi_lora_balanced_r8_highnoise | 0.776 | 349 / 450 | +0.031 |
| tardi_lora_balanced_r16 | 0.776 | 349 / 450 | +0.031 |
| tasknara_r8_highnoise | 0.773 | 348 / 450 | +0.029 |
| tardi_lora_balanced_r16_highnoise | 0.769 | 346 / 450 | +0.024 |
| tardi_lora_balanced_loraplus_r16 | 0.762 | 343 / 450 | +0.018 |
| tardi_lora_balanced_r8 | 0.760 | 342 / 450 | +0.016 |
| tardi_lora_balanced_r8_lr5e5_s150 | 0.760 | 342 / 450 | +0.016 |
| tasknara_r16 | 0.758 | 341 / 450 | +0.013 |
| vanilla_lora_fixed32 | 0.744 | 335 / 450 | +0.000 |
| nara_vanilla_fixed32 | 0.744 | 335 / 450 | +0.000 |
| loraplus_vanilla_fixed32 | 0.742 | 334 / 450 | -0.002 |
| choice_noise_lora_fixed32 | 0.738 | 332 / 450 | -0.007 |
| rslora_vanilla_fixed32 | 0.736 | 331 / 450 | -0.009 |
| label_lora_fixed32 | 0.733 | 330 / 450 | -0.011 |
| dora_vanilla_fixed32 | 0.733 | 330 / 450 | -0.011 |
| base_fixed32 | 0.722 | 325 / 450 | -0.022 |

## Main Finding

Best method `tardi_lora_balanced_r8_highnoise` reaches `0.776` (349 / 450), improving over the previous vanilla LoRA `0.744` by `+0.031` and over base `0.722` by `+0.053`.

The gain comes from repairing the training/evaluation mismatch: the old train set covered only six tasks and omitted ARC-Challenge, HellaSwag, and BoolQ; the optimized train set is 9-task balanced and excludes the evaluation ids. High-noise denoising and r16 capacity both help, but their combination does not stack additively.

## Task-Level Table

| Task | tardi_lora_balanced_r8_highnoise | tardi_lora_balanced_r16 | tasknara_r8_highnoise | tardi_lora_balanced_r16_highnoise | tardi_lora_balanced_loraplus_r16 | tardi_lora_balanced_r8 |
|---|---:|---:|---:|---:|---:|---:|
| arc_challenge | 0.86 | 0.84 | 0.84 | 0.86 | 0.84 | 0.82 |
| boolq | 0.92 | 0.90 | 0.92 | 0.88 | 0.88 | 0.90 |
| ceval_computer_network | 0.64 | 0.70 | 0.68 | 0.66 | 0.56 | 0.64 |
| commonsenseqa | 0.92 | 0.90 | 0.90 | 0.84 | 0.86 | 0.90 |
| hellaswag | 0.78 | 0.80 | 0.78 | 0.78 | 0.82 | 0.76 |
| mmlu_pro | 0.44 | 0.42 | 0.42 | 0.44 | 0.42 | 0.42 |
| pubmedqa | 0.74 | 0.74 | 0.74 | 0.74 | 0.72 | 0.74 |
| sciq | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 |
| winogrande | 0.76 | 0.76 | 0.76 | 0.80 | 0.84 | 0.74 |

Outputs live in `results/domain_shift/task_aware/lora_opt_v1`.
