# Choice-aware Noise-aware LoRA 与选择性再掩码控制器实验报告
## 1. 实验目标
本实验检验 masked diffusion language model 在 fixed-label reasoning 上的两个改进方向：训练侧的 choice/noise-aware LoRA，以及推理侧的 selective re-masking controller。所有方法使用相同 held-out 样本、相同 prompt 格式和相同 seed。
## 2. 主结果
| Method | Macro Acc. | Micro Acc. | Total n | Seconds | Avg calls |
|---|---:|---:|---:|---:|---:|
| llada_vanilla_lora_fixed32 | 74.4% | 74.4% | 450 | 645.7 |  |
| llada_vanilla_lora_controller | 74.2% | 74.2% | 450 | 543.9 | 25.24 |
| llada_choice_noise_lora_fixed32 | 73.8% | 73.8% | 450 | 648.8 |  |
| llada_label_lora_fixed32 | 73.3% | 73.3% | 450 | 647.7 |  |
| llada_choice_noise_lora_controller | 73.3% | 73.3% | 450 | 613.9 | 25.30 |
| llada_base_fixed32 | 72.2% | 72.2% | 450 | 579.1 |  |

当前最佳方法为 `llada_vanilla_lora_fixed32`，相对 base fixed-32 的宏平均变化为 +2.2 个百分点。

## 3. 逐任务结果
| Task | llada_vanilla_lora_fixed32 | llada_vanilla_lora_controller | llada_choice_noise_lora_fixed32 | llada_label_lora_fixed32 | llada_choice_noise_lora_controller | llada_base_fixed32 |
|---|---:|---:|---:|---:|---:|---:|
| mmlu_pro | 38.0% | 38.0% | 42.0% | 38.0% | 42.0% | 44.0% |
| pubmedqa | 70.0% | 74.0% | 74.0% | 74.0% | 74.0% | 64.0% |
| ceval_computer_network | 60.0% | 60.0% | 58.0% | 60.0% | 58.0% | 56.0% |
| sciq | 90.0% | 90.0% | 88.0% | 86.0% | 88.0% | 86.0% |
| winogrande | 74.0% | 72.0% | 74.0% | 74.0% | 74.0% | 76.0% |
| commonsenseqa | 90.0% | 90.0% | 82.0% | 84.0% | 82.0% | 80.0% |
| arc_challenge | 88.0% | 88.0% | 88.0% | 88.0% | 88.0% | 86.0% |
| hellaswag | 72.0% | 72.0% | 68.0% | 68.0% | 68.0% | 74.0% |
| boolq | 88.0% | 84.0% | 90.0% | 88.0% | 86.0% | 84.0% |

## 4. 解释
- `vanilla LoRA` 用于判断普通 DDM LoRA 是否能自然转化为 final-label accuracy gain。
- `label-focused LoRA` 直接优化合法标签 posterior，用于验证 final-label objective mismatch。
- `choice-noise LoRA` 在 label objective 上加入多噪声阶段监督和一致性约束，用于降低单纯标签对齐带来的负迁移。
- `controller` 若保持接近 fixed-32 的准确率并降低 avg calls，则说明适配后的模型仍存在样本级 reverse budget heterogeneity。

## 5. 文件
原始 JSON、日志、表格均位于 `results/domain_shift/task_aware/choice_noise_v1`。
