# <img src="assets/logo.png" alt="KaliBench" width="64"/> KaliBench

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-coming%20soon-B31B1B.svg" alt="arXiv: coming soon">
  <a href="https://huggingface.co/collections/RISys-Lab/kalibench-datasets-and-models"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KaliBench-orange" alt="Hugging Face: KaliBench datasets and models"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python: 3.10+">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c" alt="Framework: PyTorch">
</p>

<p align="center">
  🌐 <a href="https://risys-lab.github.io/KaliBench/">Project Page</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  📊 <a href="https://huggingface.co/collections/RISys-Lab/kalibench-datasets-and-models">Datasets &amp; Models</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  📄 Paper: arXiv coming soon
</p>

**Official repository for "KaliBench: A Fine-Grained Benchmark for Cybersecurity Tool Use on Kali Linux with Runtime-Free Verifiable Rewards" (NeurIPS 2026).**

**Authors:** Pengfei Li<sup>1,∗</sup>, Naufal Suryanto<sup>1,∗</sup>, Sicheng Zhang<sup>1</sup>, Muzammal Naseer<sup>1,2</sup>

<sup>1</sup> Khalifa University · <sup>2</sup> University of Western Australia · <sup>∗</sup> Equal contribution

---

## Overview

KaliBench evaluates natural-language-to-command translation on Kali Linux. It contains **8,504 verified query-command pairs** spanning **1,642 sub-tools** and **23 tool dimensions**. The repository includes data construction, supervised fine-tuning (SFT), reinforcement learning with verifiable rewards (GRPO/RLVR), and evaluation code.

## Evaluation settings

Models are evaluated under three tool-knowledge settings:

| Mode | Information available to the model |
| --- | --- |
| `unrestricted` | The query only; the model selects the tool. |
| `restricted` | The query and a candidate set of allowed tools. |
| `hinted` | The query, the target tool, and its usage documentation. |

Metrics include tool selection, optional-argument F1, positional-argument F1, total score, and exact command match, with aggregation across tool dimensions.

## Dataset

| Released data | Rows | Purpose |
| --- | ---: | --- |
| [Training split](KaliBench_data/kalibench_verified_train_3504.jsonl) | 3,504 | SFT and GRPO/RLVR |
| [Test split](KaliBench_data/kalibench_verified_test_5000.jsonl) | 5,000 | Held-out evaluation |
| [Tool documentation](KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl) | 2,809 | Sub-tool names and usage information |

See the [dataset reference](docs/reference.md#dataset-and-benchmark-splits) for schemas, examples, and coverage.

## Released models

Find the datasets and models in our [Hugging Face collection](https://huggingface.co/collections/RISys-Lab/kalibench-datasets-and-models).

| Model | Training |
| --- | --- |
| [RISys-Lab/RedSage-K-SFT](https://huggingface.co/RISys-Lab/RedSage-K-SFT) | Supervised fine-tuning (SFT) |
| [RISys-Lab/RedSage-K-GRPO](https://huggingface.co/RISys-Lab/RedSage-K-GRPO) | GRPO with verifiable rewards |
| [RISys-Lab/RedSage-K-SFT-GRPO](https://huggingface.co/RISys-Lab/RedSage-K-SFT-GRPO) | SFT followed by GRPO |

## Installation and evaluation

### Installation

Run from the repository root with Python 3.10:

```bash
conda create -n kalibench python=3.10
conda activate kalibench
pip install -r requirements.txt
```

Evaluation requires GPU-compatible PyTorch, CUDA, vLLM, and FlashInfer/Triton builds. Model downloads require network access unless cached. See the [training guide](docs/training.md) for additional dependencies.

### Inference examples

The [`demo/`](demo/) directory contains command-line inference examples using Hugging Face Transformers:

| Example | Model |
| --- | --- |
| [SFT inference](demo/sft_inference.py) | `RISys-Lab/RedSage-K-SFT` |
| [GRPO inference](demo/grpo_inference.py) | `RISys-Lab/RedSage-K-SFT-GRPO` (default), or `RISys-Lab/RedSage-K-GRPO` |

Install the demo dependencies, then run from the repository root:

```bash
pip install torch transformers accelerate

python demo/sft_inference.py \
  --query "In list mode, display the privileges of user 'eve' as they would apply to the command 'cat /etc/shadow', using non-interactive mode."
python demo/grpo_inference.py \
  --query "Use sqlmap to test the URL 'http://192.168.1.250/?p=1&forumaction=search' and enumerate all available databases."

# Use the GRPO-only model:
python demo/grpo_inference.py \
  --model RISys-Lab/RedSage-K-GRPO \
  --query "Remove the IPv4 address 10.0.0.5 from the interface eth2 using ifconfig."
```

Use `--model` for a Hugging Face model ID or local path. Generation options include `--max-new-tokens`, `--temperature` (0 for greedy decoding), `--top-p`, `--top-k`, and `--seed`. SFT defaults to greedy decoding with 256 new tokens; GRPO defaults to temperature 0.1 with 8,192 new tokens. Both use `--dtype bfloat16` and `--device-map auto`; for CPU inference, use `--device-map cpu --dtype float32`.

Each script prints its generated response. Run either script with `--help` to list all options.

### Evaluation

Set `MODEL` to a local path or Hugging Face model ID:

```bash
export MODEL="/path/to/model"

python src/evaluate.py \
  --mode hinted \
  --input "$PWD/KaliBench_data/kalibench_verified_test_5000.jsonl" \
  --subtools "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
  --model "$MODEL" \
  --output-dir "$PWD/outputs/evaluate/my-model/hinted" \
  --candidate-seed 42 \
  --seed 3407 \
  --temperature 0.2 \
  --batch-size 128 \
  --tensor-parallel-size 1
```

Outputs include predictions, per-example scores, and `<model>.hinted.summary.json`. Inference checkpoints each batch and resumes incomplete runs by default; use a separate output directory for each model/mode configuration. See the [evaluation guide](docs/evaluation.md) for runtime options, dimension scores, and result tables.

## Documentation

| Guide | Contents |
| --- | --- |
| [Data construction](docs/data-construction.md) | Candidate generation, verification, and split construction |
| [Training](docs/training.md) | SFT, GRPO/RLVR, reward weights, and LoRA export |
| [Evaluation and scoring](docs/evaluation.md) | Inference, recovery, saved-prediction scoring, and CSV/LaTeX tables |
| [Dataset and code reference](docs/reference.md) | Schemas, coverage, and source entrypoints |

## Citation

If you use KaliBench in your research, please cite:

```bibtex
@inproceedings{li2026kalibench,
  title={KaliBench: A Fine-Grained Benchmark for Cybersecurity Tool Use on Kali Linux with Runtime-Free Verifiable Rewards},
  author={Pengfei Li and Naufal Suryanto and Sicheng Zhang and Muzammal Naseer},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year={2026},
  url={https://github.com/RISys-Lab/KaliBench}
}
```
