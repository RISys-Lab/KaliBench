# Dataset and code reference

[README](../README.md) · [Reference](reference.md) · [Data construction](data-construction.md) · [Training](training.md) · [Evaluation](evaluation.md)

## Dataset and benchmark splits

The three released JSONL files are:

| File | Rows | Purpose |
| --- | ---: | --- |
| [KaliBench_data/kalibench_verified_train_3504.jsonl](../KaliBench_data/kalibench_verified_train_3504.jsonl) | 3,504 | SFT and GRPO/RLVR training split |
| [KaliBench_data/kalibench_verified_test_5000.jsonl](../KaliBench_data/kalibench_verified_test_5000.jsonl) | 5,000 | Held-out benchmark evaluation split |
| [KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl](../KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl) | 2,809 | Kali sub-tool names, parent tool titles, and usage documentation |

The train and test rows contain:

```text
custom_id, query, tool_name, ground_truth_command,
optional_args, positional_args, model_verification, terminal_verification
```

The usage-document rows contain:

```text
source_row, subtool, title, usage_code
```

The verified data is also available on Hugging Face: [RISys-Lab/KaliBench](https://huggingface.co/datasets/RISys-Lab/KaliBench).

## Samples and tool coverage

Each benchmark row pairs a natural-language query with a canonical command, decomposed arguments, and verification metadata.

<p align="center">
  <img src="../assets/samples.png" alt="KaliBench samples" width="80%">
</p>

The benchmark spans 23 Kali tool dimensions:

<p align="center">
  <img src="../assets/windmill.png" alt="KaliBench tool coverage" width="50%">
</p>

## Code reference

The artifact includes data construction, SFT, GRPO/RLVR, inference, checkpoint recovery, scoring, and result-table generation. Optional adjudication decisions are stored as auditable JSONL.

| Experimental stage | Main files |
| --- | --- |
| Constructed and verified data | [KaliBench_data/](../KaliBench_data/) |
| Kali documentation extraction and candidate generation | [src/data_creation/generate.py](../src/data_creation/generate.py) |
| Documentation-grounded model verification and regeneration | [src/data_creation/verify.py](../src/data_creation/verify.py) |
| Isolated terminal execution, rule filtering, and adjudication merge | [src/data_creation/terminal_verify.py](../src/data_creation/terminal_verify.py) |
| Kali Everything terminal-verification environment | [docker/kali-everything/Dockerfile](../docker/kali-everything/Dockerfile) |
| Exact deduplication and tool-stratified split finalization | [src/data_creation/finalize.py](../src/data_creation/finalize.py) |
| Data-construction package and shared utilities | [src/data_creation/__init__.py](../src/data_creation/__init__.py), [src/data_creation/common.py](../src/data_creation/common.py) |
| KaliBench SFT | [src/train/sft_kalibench.py](../src/train/sft_kalibench.py) |
| General chat SFT utility | [src/train/sft_general.py](../src/train/sft_general.py) |
| GRPO/RLVR with verifiable rewards | [src/train/grpo_kalibench.py](../src/train/grpo_kalibench.py) |
| LoRA merge/export | [src/train/merge_lora.py](../src/train/merge_lora.py) |
| Prompt construction for all three modes | [src/prompt.py](../src/prompt.py), [src/prompt_builder.py](../src/prompt_builder.py) |
| vLLM inference and recovery from interruption | [src/evaluate.py](../src/evaluate.py), [src/vllm_inference.py](../src/vllm_inference.py), [src/prediction_checkpoint.py](../src/prediction_checkpoint.py), [src/vllm_runtime.py](../src/vllm_runtime.py) |
| Tool/argument/exact-match scoring | [src/model_score.py](../src/model_score.py), [src/scoring_pipeline.py](../src/scoring_pipeline.py) |
| Dimension-level scoring | [Scoring/dim_score.py](../Scoring/dim_score.py) |
| CSV and LaTeX result tables | [src/format_table_results.py](../src/format_table_results.py) |
| Shared JSONL I/O | [src/io_utils.py](../src/io_utils.py) |
| Data-construction regression tests | [tests/test_data_creation.py](../tests/test_data_creation.py) |

Use `src/data_creation/` for data construction and `src/evaluate.py` for evaluation. The older [eval_request/](../eval_request/) and [Scoring/model_score.py](../Scoring/model_score.py) interfaces remain available for compatibility with previously generated files.

Run an entrypoint with `--help` for its full CLI configuration.
