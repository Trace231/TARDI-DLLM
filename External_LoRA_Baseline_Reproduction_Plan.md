# External Improved LoRA Baseline Reproduction Plan

## 1. 为什么要补

当前 `choice_noise_v1` 已经完成了 LLaDA 内部 LoRA objective ablation：

```text
base fixed-32
vanilla denoising LoRA
label-focused LoRA
choice-noise LoRA
LoRA + selective re-masking controller
```

但这仍然主要是我们自己的消融实验。要回答“是否超过现有 improved LoRA 工作”，还需要把外部 LoRA baseline 放到同一协议下比较。

## 2. 要补的 baseline

本轮新增代码支持四个外部/改进 LoRA baseline：

| Method | 类型 | 说明 |
|---|---|---|
| rsLoRA | 通用 improved LoRA | rank-stabilized LoRA，测试缩放方式是否比 vanilla LoRA 稳 |
| DoRA | 通用 improved LoRA | weight-decomposed LoRA，测试方向/幅度分解是否提升 fixed-label reasoning |
| NaRA-style vanilla | dLLM-specific improved LoRA | mask-ratio-conditioned dynamic core `B C(lambda) A x`，复现 NaRA 的机制思想 |
| NaRA-style choice-noise | 我们的扩展 | 在 NaRA-style adapter 上加入 choice/noise objective |

注意：

```text
NaRA-style 是机制级复现，不声称调用了官方作者代码。
```

因为目前没有接入可验证的官方 NaRA repo。报告中应写成 `NaRA-style` 或 `mechanism-level reproduction`。

## 3. 已实现代码

核心动态 adapter：

```text
scripts/nara_adapter.py
```

训练入口：

```text
scripts/train_llada_choice_noise_lora.py
```

新增参数：

```bash
--peft-variant lora|rslora|dora|nara
--nara-buckets 4
```

评测自动加载：

```text
scripts/eval_subset.py
```

如果 adapter 目录下存在：

```text
nara_config.json
nara_adapter.pt
```

则自动安装 NaRA-style adapter；否则走 PEFT adapter。

一键队列：

```text
scripts/run_external_lora_baselines.sh
```

汇总脚本：

```text
scripts/analyze_external_lora_baselines.py
```

## 4. 一键运行

在 GPU 服务器：

```bash
cd /data/llada_eval
git pull
bash scripts/run_external_lora_baselines.sh
```

默认配置：

```text
MODEL=/data/hf/models/GSAI-ML/LLaDA-8B-Instruct
TRAIN=results/domain_shift/task_aware/solid_v2/lora_control_v2/train/domain_mix_final_typed_control_seed23.jsonl
ROOT=results/domain_shift/task_aware/lora_external_v1
TASKS=mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq
LIMIT=50
SEED=23
```

输出：

```text
results/domain_shift/task_aware/lora_external_v1/
  adapters/
  raw/
  logs/
  tables/
  reports/
```

## 5. 判定标准

当前已完成的强 baseline：

| Method | Macro Acc. | Avg calls |
|---|---:|---:|
| LLaDA base fixed-32 | 0.722 | 32.00 |
| LLaDA vanilla LoRA fixed-32 | 0.744 | 32.00 |
| LLaDA vanilla LoRA + controller | 0.742 | 25.24 |
| LLaDA choice-noise LoRA fixed-32 | 0.738 | 32.00 |

外部 baseline 跑完后，至少需要比较：

```text
rsLoRA fixed-32
DoRA fixed-32
NaRA-style fixed-32
NaRA-style choice-noise fixed-32
```

如果：

```text
NaRA-style choice-noise > NaRA-style vanilla
```

可以说我们的 fixed-label objective 对 dLLM-specific adapter 有增益。

如果：

```text
vanilla LoRA + controller >= best external LoRA fixed-32
```

并且 calls 更低，可以说我们的系统在 quality-cost trade-off 上超过外部 LoRA baseline。

如果外部 baseline fixed-32 accuracy 更高，则不能说超过；应改成：

```text
external LoRA has better full-budget quality, while our controller improves efficiency.
```

## 6. 当前状态

代码和汇总脚本已经完成并通过语法检查。

但当前服务器状态：

```text
223.109.239.30:14824 connection refused
223.109.239.36:22716 timed out
223.109.239.36:24010 connection refused
jq1.9gpu.com:14860 connection refused
```

因此 external LoRA baseline 还没有真实 GPU 结果。拿到新 GPU 后运行 `scripts/run_external_lora_baselines.sh` 即可继续。
