# External Improved LoRA Baseline Comparison

This report compares external/improved LoRA baselines under the same LLaDA fixed-label protocol when their raw outputs are available.

## Macro Results

| Method | Status | Macro Acc. | Correct / N |
|---|---|---:|---:|
| llada_vanilla_lora_fixed32 | complete | 0.744 | 335 / 450 |
| llada_vanilla_lora_controller | complete | 0.742 | 334 / 450 |
| llada_choice_noise_lora_fixed32 | complete | 0.738 | 332 / 450 |
| llada_base_fixed32 | complete | 0.722 | 325 / 450 |
| llada_rslora_vanilla_fixed32 | missing |  |  |
| llada_dora_vanilla_fixed32 | missing |  |  |
| llada_nara_vanilla_fixed32 | missing |  |  |
| llada_nara_choice_noise_fixed32 | missing |  |  |

Current best completed method: `llada_vanilla_lora_fixed32` with macro accuracy `0.744`.

Missing methods still need GPU runs:

- `llada_rslora_vanilla_fixed32`
- `llada_dora_vanilla_fixed32`
- `llada_nara_vanilla_fixed32`
- `llada_nara_choice_noise_fixed32`

## Interpretation Guardrail

NaRA-style here is a mechanism-level reproduction using a mask-ratio-conditioned dynamic core `B C(lambda) A x`; it is not claimed to be the official authors' code unless the official implementation is later plugged in.

Outputs live in `results/domain_shift/task_aware/lora_external_v1`.
