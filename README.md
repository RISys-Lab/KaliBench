# <img src="assets/logo.png" alt="KaliBench" width="64"/> KaliBench

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python: 3.10+">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c" alt="Framework: PyTorch">
</p>

<p align="center">
  🌐 <a href="https://risys-lab.github.io/KaliBench/">Project Page</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  🤗 <a href="https://huggingface.co/collections/RISys-Lab/kalibench-datasets-and-models">Datasets &amp; Models</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  📄 <a href="https://arxiv.org/abs/2610.02206">arXiv Paper</a>
</p>

**Official repository for "KaliBench: A Fine-Grained Benchmark for Cybersecurity Tool Use on Kali Linux with Runtime-Free Verifiable Rewards" (NeurIPS 2026).**

**Authors:** Pengfei Li<sup>1,∗</sup>, Naufal Suryanto<sup>1,∗</sup>, Sicheng Zhang<sup>1</sup>, Muzammal Naseer<sup>1,2</sup>

<sup>1</sup> Khalifa University · <sup>2</sup> University of Western Australia · <sup>∗</sup> Equal contribution

<a id="overview"></a>

KaliBench evaluates natural-language-to-command translation on Kali Linux, with **8,504 verified query-command pairs** spanning **1,642 sub-tools** and **23 tool dimensions**.

We release RedSage-K models and reproducible training pipelines for supervised fine-tuning (SFT) and reinforcement learning with runtime-free verifiable rewards (GRPO/RLVR).

---

<a id="installation"></a>
<a id="installation-and-evaluation"></a>

## Start here

| I want to… | Go to |
| --- | --- |
| Try a released model | [Quick inference](#quick-inference) |
| Evaluate a model on KaliBench | [Evaluation guide](docs/evaluation.md) |
| Download the data or models | [Dataset](#dataset) · [Released models](#released-models) |
| Train models with SFT or GRPO/RLVR | [Training guide](docs/training.md) |
| Construct data or inspect its format | [Data construction](docs/data-construction.md) · [Reference](docs/reference.md) |

<a id="evaluation"></a>
<a id="benchmark-evaluation"></a>
<a id="evaluation-settings"></a>

## Benchmark settings

Models are evaluated under three tool-knowledge settings:

| Mode | Information available to the model |
| --- | --- |
| `unrestricted` | The query only; the model selects the tool. |
| `restricted` | The query and a candidate set of allowed tools. |
| `hinted` | The query, the target tool, and its usage documentation. |

## Dataset

| Released data | Rows | Purpose |
| --- | ---: | --- |
| [Training split](KaliBench_data/kalibench_verified_train_3504.jsonl) | 3,504 | SFT and GRPO/RLVR |
| [Test split](KaliBench_data/kalibench_verified_test_5000.jsonl) | 5,000 | Held-out evaluation |
| [Tool documentation](KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl) | 2,809 | Sub-tool names and usage information |

See the [dataset reference](docs/reference.md#dataset-and-benchmark-splits) for schemas, examples, and coverage.

## Released models

All three variants start from [RedSage-Qwen3-8B-Ins](https://huggingface.co/RISys-Lab/RedSage-Qwen3-8B-Ins). See the [training guide](docs/training.md) to reproduce them.

| Model | Training |
| --- | --- |
| [RedSage-K-SFT](https://huggingface.co/RISys-Lab/RedSage-K-SFT) | Supervised fine-tuning (SFT) |
| [RedSage-K-GRPO](https://huggingface.co/RISys-Lab/RedSage-K-GRPO) | GRPO with verifiable rewards |
| [RedSage-K-SFT-GRPO](https://huggingface.co/RISys-Lab/RedSage-K-SFT-GRPO) | SFT followed by GRPO |

<a id="inference-examples"></a>

## Quick inference

Try our best-performing released model, **RedSage-K-SFT-GRPO**, with Python 3.10 from the repository root:

```bash
pip install torch transformers accelerate
python demo/grpo_inference.py \
  --query "List the network interfaces using ifconfig."
```

See the [inference guide](docs/inference.md) for other model variants, local checkpoints, and generation options. For benchmark setup and scoring, follow the [evaluation guide](docs/evaluation.md).

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
