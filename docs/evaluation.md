# Evaluation and scoring

[README](../README.md) · [Inference](inference.md) · [Evaluation](evaluation.md) · [Training](training.md) · [Data construction](data-construction.md) · [Reference](reference.md)

Evaluate a model under KaliBench's [three benchmark settings](../README.md#benchmark-settings). Run all commands from the repository root.

`src/evaluate.py` builds prompts and labels, runs vLLM inference, checkpoints predictions, and calculates tool, optional F1, positional F1, total, exact-match, and format-error metrics. Scoring across the 23 Kali tool dimensions is optional.

## Environment setup

Create a Python 3.10 environment and install the repository dependencies:

```bash
conda create -n kalibench python=3.10
conda activate kalibench
pip install -r requirements.txt
```

Evaluation requires GPU-compatible PyTorch, CUDA, vLLM, and FlashInfer/Triton builds. Model downloads require network access unless cached. For training, install the [additional training dependencies](training.md).

## Select a model

The examples use the released RedSage-K-SFT-GRPO model:

```bash
export MODEL="RISys-Lab/RedSage-K-SFT-GRPO"
```

To evaluate another model, set `MODEL` to its Hugging Face ID or local path. For your own training run, use the merged checkpoint (for example, `"$PWD/outputs/models/kalibench_grpo/merged"`).

Use a different output directory for every model/mode pair. This prevents a resumed run from mixing predictions generated with different prompts.

## Evaluate one mode

```bash
python src/evaluate.py \
  --mode hinted \
  --input "$PWD/KaliBench_data/kalibench_verified_test_5000.jsonl" \
  --subtools "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
  --model "$MODEL" \
  --output-dir "$PWD/outputs/evaluate/kalibench_grpo/hinted" \
  --candidate-seed 42 \
  --seed 3407 \
  --temperature 0.2 \
  --batch-size 128 \
  --tensor-parallel-size 1
```

In hinted mode, evaluation intentionally forces one candidate tool and includes its usage information. In restricted mode, `--candidate-tools 20` is the default. Unrestricted mode does not use a candidate-tool list.

## Evaluate all three modes

```bash
for MODE in unrestricted restricted hinted; do
  python src/evaluate.py \
    --mode "$MODE" \
    --input "$PWD/KaliBench_data/kalibench_verified_test_5000.jsonl" \
    --subtools "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
    --model "$MODEL" \
    --output-dir "$PWD/outputs/evaluate/kalibench_grpo/$MODE" \
    --candidate-tools 20 \
    --candidate-seed 42 \
    --seed 3407 \
    --temperature 0.2 \
    --batch-size 128 \
    --tensor-parallel-size 1
done
```

## Resume and runtime options

The default `--resume true` behavior reads existing `raw_predictions.jsonl` and `vllm_compat_predictions.jsonl` files, validates their IDs, and evaluates only missing examples. Use `--resume false` to start a clean run in an existing output directory.

Useful hardware/runtime flags include:

```text
--tensor-parallel-size
--gpu-memory-utilization
--max-model-len
--max-num-batched-tokens
--dtype
--enforce-eager
--language-model-only
--gdn-prefill-backend
--inference-api
```

Run `python src/evaluate.py --help` for their complete descriptions.

## Evaluation outputs

Each evaluation directory contains:

```text
run_config.json
labels.jsonl
raw_predictions.jsonl
vllm_compat_predictions.jsonl
<model>.<mode>.summary.json
model_scores/<model>.<mode>.scores.jsonl
```

`--output-raw` additionally stores the actual prompt messages and input/output token counts. `--wandb` enables optional run configuration and summary logging.

## Scoring and result generation

### Re-score saved predictions

Scoring runs automatically unless `--skip-scoring` is passed. To re-score an existing evaluation without repeating inference:

```bash
python src/scoring_pipeline.py \
  --folder "$PWD/outputs/evaluate/kalibench_grpo/hinted"
```

For explicit prediction and label files:

```bash
python src/model_score.py \
  --vllm_file "$PWD/outputs/evaluate/kalibench_grpo/hinted/vllm_compat_predictions.jsonl" \
  --label_file "$PWD/outputs/evaluate/kalibench_grpo/hinted/labels.jsonl" \
  --out "$PWD/outputs/evaluate/kalibench_grpo/hinted/model_scores/rescored.jsonl"
```

### Dimension-level scores

Dimension-level scoring uses the tool-to-title mapping in the released usage file and the anonymous Hugging Face metadata for the 23 metapackage labels. This lookup requires network access unless the metadata is already cached:

```bash
python Scoring/dim_score.py \
  --models-folder "$PWD/outputs/evaluate/kalibench_grpo/hinted/model_scores" \
  --output-folder "$PWD/outputs/evaluate/kalibench_grpo/hinted/dim_scores" \
  --subtools "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
  --use-hf \
  --hf-dataset "anonymous62567/kali-tools"
```

The same step can be appended to end-to-end evaluation with:

```text
--run-dim-score --dim-use-hf --dim-hf-dataset anonymous62567/kali-tools
```

### Paper-style result tables

`src/format_table_results.py` converts an aggregate run CSV into an organized CSV and a LaTeX table grouped by model size. Its input expects one row per model/mode configuration with `config_*` and `summary_*` columns.

```bash
python src/format_table_results.py aggregate_results.csv \
  --output-csv organized_scores.csv \
  --output-tex organized_scores.tex \
  --size-grouping coarse
```

Use `--override-sizes MODEL:7B` for model names that do not contain a recognizable parameter count.
