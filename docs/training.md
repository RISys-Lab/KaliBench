# Training

[README](../README.md) · [Reference](reference.md) · [Data construction](data-construction.md) · [Training](training.md) · [Evaluation](evaluation.md)

Complete the [environment setup](../README.md#installation-and-evaluation), then install the training dependencies:

```bash
pip install unsloth trl wandb
```

Unsloth, TRL, PyTorch, and CUDA must be compatible with your GPU and driver. The examples use `--report-to none` to disable Weights & Biases logging. Run all commands from the repository root.

Set the base model to a local path or a Hugging Face model ID:

```bash
export BASE_MODEL="/path/to/base-model"
```

The main paper-reproduction path is:

```text
constructed and verified training split → KaliBench SFT → GRPO/RLVR → merged model → evaluation
```

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

For the SFT+GRPO experiment, initialize GRPO from the merged SFT model. The training objective combines output-format rewards with the same tool, optional-argument, positional-argument, and exact-match signals used during evaluation.

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

Each `--mode` value can include a sampling fraction in `[0,1]`, such as `restricted:0.5`. The default reward weights are:

| Reward component | Weight |
| --- | ---: |
| exact output format | 2.0 |
| approximate output format | 1.0 |
| tool score | 1.0 |
| optional-argument F1 | 1.5 |
| positional-argument F1 | 1.5 |
| exact match | 2.0 |

Before launching a long run, the prepared prompts and token-length statistics can be inspected without training:

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
