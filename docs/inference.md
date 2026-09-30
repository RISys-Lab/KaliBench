# Inference

[README](../README.md) · [Inference](inference.md) · [Evaluation](evaluation.md) · [Training](training.md) · [Data construction](data-construction.md) · [Reference](reference.md)

Generate commands with the released RedSage-K models or a local merged checkpoint. For full benchmark runs and scoring, see the [evaluation guide](evaluation.md).

## Examples

The [`demo/`](../demo/) directory contains command-line inference examples using Hugging Face Transformers:

| Example | Model |
| --- | --- |
| [SFT inference](../demo/sft_inference.py) | `RISys-Lab/RedSage-K-SFT` |
| [GRPO inference](../demo/grpo_inference.py) | `RISys-Lab/RedSage-K-SFT-GRPO` (default), or `RISys-Lab/RedSage-K-GRPO` |

With Python 3.10, install the demo dependencies and run from the repository root:

```bash
pip install torch transformers accelerate

python demo/sft_inference.py \
  --query "List the network interfaces using ifconfig."
python demo/grpo_inference.py \
  --query "Use sqlmap to test the URL 'http://192.168.1.250/?p=1&forumaction=search' and enumerate all available databases."

# Use the GRPO-only model:
python demo/grpo_inference.py \
  --model RISys-Lab/RedSage-K-GRPO \
  --query "Remove the IPv4 address 10.0.0.5 from the interface eth2 using ifconfig."
```

## Response format

The scripts print the model's response without executing the generated command. SFT is prompted to place the command inside an `<output>` block. An illustrative response to the first example is:

```text
<output>
ifconfig
</output>
```

GRPO is prompted to include a `<think>` reasoning block before `<output>`. Actual responses may vary by model and generation settings.

## Generation options

Use `--model` for a Hugging Face model ID or local merged checkpoint. Control generation with `--max-new-tokens`, `--temperature` (0 for greedy decoding), `--top-p`, `--top-k`, and `--seed`.

| Script | Default decoding | Maximum new tokens |
| --- | --- | ---: |
| SFT | Greedy | 256 |
| GRPO | Sampling at temperature 0.1 | 8,192 |

Both use `--dtype bfloat16` and `--device-map auto`; for CPU inference, use `--device-map cpu --dtype float32`. Run either script with `--help` to list all options.
