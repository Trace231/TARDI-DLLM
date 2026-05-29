# Coverage Addendum: Broader Downstream Tasks

This addendum extends the original closed-label suite with science reasoning, situation continuation, reading yes/no, and answer-only math.
The goal is coverage, not a new controller tuned to every task.

## Summary Table

| Method | Task | Acc | n | Avg Calls | Route |
|---|---:|---:|---:|---:|---|
| llada8b_32step | ARC-Challenge | 0.857 | 300 |  | `{"fixed": 1.0}` |
| llada8b_8step | ARC-Challenge | 0.823 | 300 |  | `{"fixed": 1.0}` |
| llada8b_calibrated | ARC-Challenge | 0.857 | 300 | 33.000 | `{"32": 1.0}` |
| qwen25_7b | ARC-Challenge | 0.890 | 300 |  | `{"fixed": 1.0}` |
| llada8b_32step | BoolQ | 0.867 | 300 |  | `{"fixed": 1.0}` |
| llada8b_8step | BoolQ | 0.843 | 300 |  | `{"fixed": 1.0}` |
| llada8b_calibrated | BoolQ | 0.853 | 300 | 12.867 | `{"32": 0.07666666666666666, "8": 0.86, "8->32": 0.06333333333333334}` |
| qwen25_7b | BoolQ | 0.823 | 300 |  | `{"fixed": 1.0}` |
| llada8b_32step | GSM8K | 0.580 | 100 |  | `{"fixed": 1.0}` |
| llada8b_8step | GSM8K | 0.320 | 100 |  | `{"fixed": 1.0}` |
| qwen25_7b | GSM8K | 0.770 | 100 |  | `{"fixed": 1.0}` |
| llada8b_32step | HellaSwag | 0.780 | 300 |  | `{"fixed": 1.0}` |
| llada8b_8step | HellaSwag | 0.760 | 300 |  | `{"fixed": 1.0}` |
| llada8b_calibrated | HellaSwag | 0.780 | 300 | 33.000 | `{"32": 1.0}` |
| qwen25_7b | HellaSwag | 0.850 | 300 |  | `{"fixed": 1.0}` |

## Deltas

| Task | Comparison | Delta |
|---|---:|---:|
| ARC-Challenge | 32step_minus_8step | 0.033 |
| ARC-Challenge | calibrated_minus_32step | 0.000 |
| ARC-Challenge | llada32_minus_qwen | -0.033 |
| BoolQ | 32step_minus_8step | 0.023 |
| BoolQ | calibrated_minus_32step | -0.013 |
| BoolQ | llada32_minus_qwen | 0.043 |
| GSM8K | 32step_minus_8step | 0.260 |
| GSM8K | llada32_minus_qwen | -0.190 |
| HellaSwag | 32step_minus_8step | 0.020 |
| HellaSwag | calibrated_minus_32step | 0.000 |
| HellaSwag | llada32_minus_qwen | -0.070 |

## Interpretation

- ARC-Challenge, HellaSwag, and BoolQ broaden the closed-label setting beyond the original commonsense/knowledge tasks.
- GSM8K is included as an answer-only long-chain reasoning boundary case; the controller is not claimed as a complete solution for open numeric reasoning.
- If 8-step and 32-step differ on a task, the task is reverse-budget sensitive. If both fail similarly, the bottleneck is more likely knowledge, reasoning format, or answer extraction.
