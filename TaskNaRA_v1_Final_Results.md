# TaskNaRA v1 Final Results

## 1. 目的

上一轮最强训练侧结果来自 **TARDI-LoRA balanced r8 high-noise**：通过 9-task balanced、eval-disjoint 数据和 high-noise final-label denoising，把 LLaDA fixed-label reasoning 从 base `0.722` 提升到 `0.776`。本轮进一步回答一个更“架构味”的问题：

> 能不能让 LoRA adapter 本身感知任务类型和扩散噪声阶段，从而超过普通 TARDI-LoRA？

因此实现并测试了 **TaskNaRA**：一种 task-aware + noise-conditioned 的 LLaDA LoRA adapter。

## 2. 方法

TaskNaRA 不改变 LLaDA backbone，也不声称修改 LLaDA 架构。它只改 PEFT adapter 的条件化方式。

普通 LoRA 使用固定低秩更新：

```text
W' = W + BA
```

Noise-aware adapter 在低秩更新外加入噪声条件矩阵：

```text
W'(lambda) = W + B C(lambda) A
```

TaskNaRA 进一步把 task embedding 加入条件函数：

```text
C(task, lambda) = I + c_scale * MLP([Fourier(lambda); Emb(task)])
```

其中 `lambda` 表示训练时的 mask/noise ratio，`task` 来自当前样本所属任务。直觉是：不同任务的 final-label posterior 形状不同，医学 yes/no/maybe、中文知识题、多项常识推理不应共享完全相同的 adapter correction。

实现位置：

```text
scripts/nara_adapter.py
scripts/train_llada_choice_noise_lora.py
scripts/eval_domain_shift.py
scripts/run_tasknara_v1.sh
```

## 3. 实验设置

模型：

```text
/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
```

训练数据：

```text
results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl
```

评测协议：

```text
9 tasks
limit = 50 per task
seed = 23
fixed 32-step final-label evaluation
```

任务：

```text
mmlu_pro, pubmedqa, ceval_computer_network, sciq,
winogrande, commonsenseqa, arc_challenge, hellaswag, boolq
```

## 4. Macro Results

| Method | Macro Acc. | Correct / N | vs Base | vs Old Vanilla |
|---|---:|---:|---:|---:|
| TARDI-LoRA balanced r8 high-noise | 0.776 | 349 / 450 | +0.053 | +0.031 |
| TARDI-LoRA balanced r16 | 0.776 | 349 / 450 | +0.053 | +0.031 |
| TaskNaRA r8 high-noise | 0.773 | 348 / 450 | +0.051 | +0.029 |
| TARDI-LoRA balanced r16 high-noise | 0.769 | 346 / 450 | +0.047 | +0.024 |
| TARDI-LoRA balanced LoRA+ r16 | 0.762 | 343 / 450 | +0.040 | +0.018 |
| TARDI-LoRA balanced r8 | 0.760 | 342 / 450 | +0.038 | +0.016 |
| TaskNaRA r16 | 0.758 | 341 / 450 | +0.036 | +0.013 |
| Old vanilla LoRA fixed-32 | 0.744 | 335 / 450 | +0.022 | 0.000 |
| Base fixed-32 | 0.722 | 325 / 450 | 0.000 | -0.022 |

## 5. Task-Level Results

| Task | TARDI r8 high-noise | TaskNaRA r8 high-noise | TaskNaRA r16 | Old vanilla | Base |
|---|---:|---:|---:|---:|---:|
| MMLU-Pro | 0.44 | 0.42 | 0.40 | 0.38 | 0.44 |
| PubMedQA | 0.74 | 0.74 | 0.74 | 0.70 | 0.64 |
| C-Eval CN | 0.64 | 0.68 | 0.62 | 0.60 | 0.56 |
| SciQ | 0.92 | 0.92 | 0.92 | 0.90 | 0.86 |
| WinoGrande | 0.76 | 0.76 | 0.76 | 0.74 | 0.76 |
| CommonsenseQA | 0.92 | 0.90 | 0.90 | 0.90 | 0.80 |
| ARC-Challenge | 0.86 | 0.84 | 0.84 | 0.88 | 0.86 |
| HellaSwag | 0.78 | 0.78 | 0.74 | 0.72 | 0.74 |
| BoolQ | 0.92 | 0.92 | 0.90 | 0.88 | 0.84 |

## 6. 结论

TaskNaRA 是一个真实的非 toy 架构改进：adapter 同时感知 task id 和 diffusion noise ratio，并在训练、保存、加载、评测全链路接通。结果也确实超过 old vanilla LoRA：

```text
TaskNaRA r8 high-noise: 0.773
Old vanilla LoRA:       0.744
Base LLaDA:             0.722
```

但是它没有超过当前最强的 TARDI-LoRA：

```text
TARDI r8 high-noise:    0.776
TaskNaRA r8 high-noise: 0.773
```

差距只有 `1/450`，但这说明最终论文/报告中不应把 TaskNaRA 作为“显著优于 TARDI”的主结果。更稳的讲法是：

1. 复杂 task-conditioned hypernetwork 不是免费收益。
2. 在当前 100-step、每任务 100 条训练规模下，主增益仍来自 task-balanced coverage 与 high-noise final-label denoising。
3. TaskNaRA 在 C-Eval 上高于 TARDI r8 high-noise（`0.68` vs `0.64`），说明 task conditioning 会改变任务偏置；但它在 CommonsenseQA、ARC-Challenge、MMLU-Pro 上略有损失。
4. r16 TaskNaRA 明显低于 r8 TaskNaRA，提示更大条件化 adapter 可能放大任务间负迁移。

## 7. 推荐写法

不要写：

```text
We propose TaskNaRA and it outperforms all LoRA baselines.
```

可以写：

```text
We further explored a task-aware noise-conditioned adapter, TaskNaRA, where the LoRA correction is conditioned on both task identity and diffusion noise level. TaskNaRA improves over the old vanilla LLaDA LoRA baseline, but does not surpass the simpler task-balanced high-noise TARDI-LoRA protocol. This negative result suggests that, for fixed-label reasoning under limited data, aligning data coverage and denoising noise stage is more important than adding task-conditioned adapter capacity.
```

中文：

> 我们进一步实现了 task-aware noise-conditioned adapter，使 LoRA 更新同时依赖任务类型和扩散噪声阶段。该方法超过旧 vanilla LLaDA LoRA，但没有超过更简单的 task-balanced high-noise TARDI-LoRA。这说明在小规模 fixed-label adaptation 中，训练覆盖与噪声阶段对齐比引入更复杂的 task-conditioned adapter 容量更关键。

## 8. 文件位置

```text
results/domain_shift/task_aware/lora_tasknara_v1/raw/
results/domain_shift/task_aware/lora_opt_v1/tables/lora_opt_macro_summary.csv
results/domain_shift/task_aware/lora_opt_v1/tables/lora_opt_task_summary.csv
scripts/nara_adapter.py
scripts/train_llada_choice_noise_lora.py
scripts/eval_domain_shift.py
scripts/run_tasknara_v1.sh
```
