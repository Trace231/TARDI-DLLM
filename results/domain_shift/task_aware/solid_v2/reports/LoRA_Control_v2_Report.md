# Controlled LoRA-v2: AR vs DDM Adaptation Gain

Matched protocol: same JSONL training set, LoRA r=8/alpha=16/dropout=0.05, 200 update steps, same 100-sample typed final-label evaluation.
AR uses native causal final-label SFT; DDM uses native LLaDA diffusion GRPO final-label reward. The original DDM LoRA checkpoint is retained as an additional reference, not as the controlled comparison.

| Method | Task | Value | n | Protocol |
|---|---|---:|---:|---|
| qwen_base_control | MMLU-Pro | 0.380 | 100 | base |
| qwen_base_control | PubMedQA | 0.510 | 100 | base |
| qwen_base_control | C-Eval CN | 0.730 | 100 | base |
| qwen_base_control | SciQ | 0.970 | 100 | base |
| qwen_base_control | WinoGrande | 0.630 | 100 | base |
| qwen_base_control | CommonsenseQA | 0.820 | 100 | base |
| qwen_lora_control_v2 | MMLU-Pro | 0.400 | 100 | controlled_v2 |
| qwen_lora_control_v2 | PubMedQA | 0.770 | 100 | controlled_v2 |
| qwen_lora_control_v2 | C-Eval CN | 0.750 | 100 | controlled_v2 |
| qwen_lora_control_v2 | SciQ | 0.970 | 100 | controlled_v2 |
| qwen_lora_control_v2 | WinoGrande | 0.720 | 100 | controlled_v2 |
| qwen_lora_control_v2 | CommonsenseQA | 0.860 | 100 | controlled_v2 |
| llada_base_control | MMLU-Pro | 0.350 | 100 | base |
| llada_base_control | PubMedQA | 0.750 | 100 | base |
| llada_base_control | C-Eval CN | 0.580 | 100 | base |
| llada_base_control | SciQ | 0.920 | 100 | base |
| llada_base_control | WinoGrande | 0.630 | 100 | base |
| llada_base_control | CommonsenseQA | 0.870 | 100 | base |
| llada_lora_control_v2 | MMLU-Pro | 0.350 | 100 | controlled_v2 |
| llada_lora_control_v2 | PubMedQA | 0.750 | 100 | controlled_v2 |
| llada_lora_control_v2 | C-Eval CN | 0.570 | 100 | controlled_v2 |
| llada_lora_control_v2 | SciQ | 0.920 | 100 | controlled_v2 |
| llada_lora_control_v2 | WinoGrande | 0.630 | 100 | controlled_v2 |
| llada_lora_control_v2 | CommonsenseQA | 0.870 | 100 | controlled_v2 |
| llada_lora_original | MMLU-Pro | 0.440 | 100 | original_ddm_lora |
| llada_lora_original | PubMedQA | 0.720 | 100 | original_ddm_lora |
| llada_lora_original | C-Eval CN | 0.540 | 100 | original_ddm_lora |
| llada_lora_original | SciQ | 0.880 | 100 | original_ddm_lora |
| llada_lora_original | WinoGrande | 0.780 | 100 | original_ddm_lora |
| llada_lora_original | CommonsenseQA | 0.800 | 100 | original_ddm_lora |
| ar_lora_control_gain | C-Eval CN | 0.020 | 100 | gain |
| ar_lora_control_gain | CommonsenseQA | 0.040 | 100 | gain |
| ar_lora_control_gain | MMLU-Pro | 0.020 | 100 | gain |
| ar_lora_control_gain | PubMedQA | 0.260 | 100 | gain |
| ar_lora_control_gain | SciQ | 0.000 | 100 | gain |
| ar_lora_control_gain | WinoGrande | 0.090 | 100 | gain |
| ddm_lora_control_gain | C-Eval CN | -0.010 | 100 | gain |
| ddm_lora_control_gain | CommonsenseQA | 0.000 | 100 | gain |
| ddm_lora_control_gain | MMLU-Pro | 0.000 | 100 | gain |
| ddm_lora_control_gain | PubMedQA | 0.000 | 100 | gain |
| ddm_lora_control_gain | SciQ | 0.000 | 100 | gain |
| ddm_lora_control_gain | WinoGrande | 0.000 | 100 | gain |
| ddm_lora_original_gain | C-Eval CN | -0.040 | 100 | gain |
| ddm_lora_original_gain | CommonsenseQA | -0.070 | 100 | gain |
| ddm_lora_original_gain | MMLU-Pro | 0.090 | 100 | gain |
| ddm_lora_original_gain | PubMedQA | -0.030 | 100 | gain |
| ddm_lora_original_gain | SciQ | -0.040 | 100 | gain |
| ddm_lora_original_gain | WinoGrande | 0.150 | 100 | gain |