# TARDI-DLLM

Task-Aware Reverse Diffusion Inference for Diffusion Large Language Models.

This repository packages the course-project code, reports, and experiment
artifacts for exploring masked diffusion language models, centered on
LLaDA-8B-Instruct. The main contribution is not a backbone modification. It is
an inference-loop improvement: a task/sample-aware denoising budget controller
that allocates reverse diffusion computation according to estimated sample
risk, trajectory stability, and task shape.

## What Is Included

- `scripts/`: evaluation, controller, sampler-baseline, LoRA-audit, and analysis
  scripts used in the LLaDA experiments.
- `results/domain_shift/task_aware/solid_v2/`: raw JSON outputs, summary CSVs,
  figures, and reports for the solid experiment suite.
- `figures/`: main method illustrations.
- `LLaDA_Final_Experiment_Data_Report.md`: updated result report with final
  method comparisons and fixed-step sweep analysis.
- `LLaDA_Solid_Experiment_Report.md`: main Chinese report for the solid
  experiment suite.
- `external/Masked-Diffusion-Model-Reproduction-Experiment`: external masked
  diffusion model reproduction project included as a Git submodule.

## Main Result Snapshot

The final fast all-dataset comparison is stored at:

```text
results/domain_shift/task_aware/solid_v2/v3_choice_fast/tables/v3plus_macro_comparison_limit50_seed23.csv
```

The fine-grained fixed-step sweep is stored at:

```text
results/domain_shift/task_aware/solid_v2/step_sweep_limit20_4to32/tables/step_sweep_4to32_by_dataset_limit20_seed23.csv
```

The archived local data pack is:

```text
LLaDA_final_data_pack_20260529.tar.gz
```

## Submodule

This repository references the masked diffusion model reproduction baseline as
a submodule:

```bash
git submodule update --init --recursive
```

## Notes

The repository intentionally tracks the research artifacts needed to reproduce
and audit the reported conclusions. Large unrelated local files, temporary
exports, and other independent projects from the working directory are excluded
through `.gitignore`.
