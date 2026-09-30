# Training

[README](../README.md) · [Inference](inference.md) · [Evaluation](evaluation.md) · [Training](training.md) · [Data construction](data-construction.md) · [Reference](reference.md)

Complete the [environment setup](evaluation.md#environment-setup), then install the training dependencies:

```bash
pip install unsloth trl wandb
```

Unsloth, TRL, PyTorch, and CUDA must be compatible with your GPU and driver. The examples use `--report-to none` to disable Weights & Biases logging. Run all commands from the repository root.

## Base model and training paths

Use [RISys-Lab/RedSage-Qwen3-8B-Ins](https://huggingface.co/RISys-Lab/RedSage-Qwen3-8B-Ins) as the starting checkpoint to reproduce the RedSage-K variants:

```bash
export BASE_MODEL="RISys-Lab/RedSage-Qwen3-8B-Ins"
```

This is also the default `--model-name` in the SFT and GRPO scripts. You can override it with another Hugging Face model ID or a local path for other experiments.

| Target variant | Starting checkpoint | Stages |
| --- | --- | --- |
| RedSage-K-SFT | `$BASE_MODEL` | [KaliBench SFT](#1-supervised-fine-tuning) |
| RedSage-K-GRPO | `$BASE_MODEL` | [GRPO-only](#grpo-only) |
| RedSage-K-SFT-GRPO | `$BASE_MODEL`, then the merged KaliBench SFT model | [SFT](#1-supervised-fine-tuning) → [GRPO](#sft-followed-by-grpo) |

The commands below use the released 3,504-example training split and tool documentation explicitly. Use the held-out 5,000-example test split only for [evaluation](evaluation.md).

## 1. Supervised fine-tuning

The KaliBench SFT implementation creates examples in all three modes, applies output-only supervision, trains LoRA adapters with Unsloth/TRL, and saves both the adapter and a merged 16-bit model.

```bash
python src/train/sft_kalibench.py \
  --model-name "$BASE_MODEL" \
  --dataset-path "$PWD/KaliBench_data/kalibench_verified_train_3504.jsonl" \
  --subtools-path "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
  --mode hinted restricted unrestricted \
  --candidate-tools 20 \
  --candidate-seed 42 \
  --seed 3407 \
  --output-dir "$PWD/outputs/models/kalibench_sft" \
  --adapter-output-dir "$PWD/outputs/adapters/kalibench_sft" \
  --report-to none
```

Outputs:

```text
outputs/adapters/kalibench_sft/       # LoRA adapter and tokenizer
outputs/models/kalibench_sft/merged/  # merged 16-bit model
```

The default optimizer, LoRA rank, sequence length, batch size, accumulation, epoch count, learning rate, warmup, and checkpoint cadence are exposed as CLI flags. Run `python src/train/sft_kalibench.py --help` for the complete configuration.

## 2. GRPO/RLVR

The training objective combines output-format rewards with the same tool, optional-argument, positional-argument, and exact-match signals used during evaluation.

Rewards are runtime-free: generated commands are scored against reference labels without executing them. Ground-truth commands are verified through execution during [data construction](data-construction.md#3-execute-and-triage-commands-in-kali).

### SFT followed by GRPO

To train RedSage-K-SFT-GRPO, first complete [SFT](#1-supervised-fine-tuning), then initialize GRPO from its merged model:

```bash
python src/train/grpo_kalibench.py \
  --model-name "$PWD/outputs/models/kalibench_sft/merged" \
  --dataset-path "$PWD/KaliBench_data/kalibench_verified_train_3504.jsonl" \
  --subtools-path "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
  --mode hinted:1.0 restricted:1.0 unrestricted:1.0 \
  --candidate-tools 20 \
  --candidate-seed 42 \
  --seed 3407 \
  --output-dir "$PWD/outputs/models/kalibench_grpo" \
  --adapter-output-dir "$PWD/outputs/adapters/kalibench_grpo" \
  --report-to none
```

The merged model is saved to `outputs/models/kalibench_grpo/merged/`, and the adapter to `outputs/adapters/kalibench_grpo/`.

### GRPO-only

To train RedSage-K-GRPO, initialize directly from `$BASE_MODEL` without the KaliBench SFT stage. Use separate output directories for this variant:

```bash
python src/train/grpo_kalibench.py \
  --model-name "$BASE_MODEL" \
  --dataset-path "$PWD/KaliBench_data/kalibench_verified_train_3504.jsonl" \
  --subtools-path "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
  --mode hinted:1.0 restricted:1.0 unrestricted:1.0 \
  --candidate-tools 20 \
  --candidate-seed 42 \
  --seed 3407 \
  --output-dir "$PWD/outputs/models/kalibench_grpo_only" \
  --adapter-output-dir "$PWD/outputs/adapters/kalibench_grpo_only" \
  --report-to none
```

The merged model is saved to `outputs/models/kalibench_grpo_only/merged/`, and the adapter to `outputs/adapters/kalibench_grpo_only/`.

### Reward configuration and dataset inspection

Each `--mode` value can include a sampling fraction in `[0,1]`, such as `restricted:0.5`. The default reward weights are:

| Reward component | Weight |
| --- | ---: |
| exact output format | 2.0 |
| approximate output format | 1.0 |
| tool score | 1.0 |
| optional-argument F1 | 1.5 |
| positional-argument F1 | 1.5 |
| exact match | 2.0 |

Before launching a long run, the prepared prompts and token-length statistics can be inspected without training. Set `--model-name` to the checkpoint for your chosen path (`"$BASE_MODEL"` for GRPO-only):

```bash
python src/train/grpo_kalibench.py \
  --model-name "$PWD/outputs/models/kalibench_sft/merged" \
  --dataset-path "$PWD/KaliBench_data/kalibench_verified_train_3504.jsonl" \
  --subtools-path "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
  --mode hinted:1.0 restricted:1.0 unrestricted:1.0 \
  --debug-dataset \
  --report-to none
```

## 3. Optional training utilities

`src/train/sft_general.py` trains on a generic chat JSONL file in which every row contains a `messages` list:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

```bash
python src/train/sft_general.py \
  --model-name "$BASE_MODEL" \
  --dataset-path "/absolute/path/to/messages.jsonl" \
  --output-dir "$PWD/outputs/models/general_sft" \
  --adapter-output-dir "$PWD/outputs/adapters/general_sft" \
  --report-to none
```

The SFT and GRPO scripts already save a merged model. To merge or export a separately saved adapter:

```bash
python src/train/merge_lora.py \
  --base-model "$BASE_MODEL" \
  --adapter "$PWD/outputs/adapters/kalibench_sft" \
  --output-dir "$PWD/outputs/models/kalibench_sft_merged"
```

Continue with [evaluation and scoring](evaluation.md) after training.
