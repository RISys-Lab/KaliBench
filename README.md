# <img src="assets/logo.png" alt="KaliBench" width="100"/> KaliBench: A Fine-Grained Benchmark for Cybersecurity Tool Use on Kali Linux with Runtime-Free Verifiable Rewards

**Anonymous artifact for a paper under review.**

KaliBench evaluates query-to-CLI translation on Kali Linux under three tool-knowledge settings and 23 fine-grained tool dimensions. The verified release contains 8,504 query-command pairs covering 1,642 sub-tools: 3,504 training examples and 5,000 evaluation examples. This artifact includes the complete automated workflow for data generation and verification, supervised fine-tuning (SFT), reinforcement learning with verifiable rewards (GRPO/RLVR), inference, checkpoint recovery, scoring, and result-table generation.

---

## Table of Contents

- [Overview](#overview)
- [Artifact Coverage](#artifact-coverage)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [Dataset and Benchmark Splits](#dataset-and-benchmark-splits)
- [Data Construction](#data-construction)
- [Training](#training)
- [Evaluation](#evaluation)
- [Scoring and Result Generation](#scoring-and-result-generation)

## Overview

KaliBench measures whether a model can translate a natural-language cybersecurity request into a precise, executable Kali/Linux command. A prediction is evaluated at several levels:

- correct tool selection;
- correct optional flags and flag-value bindings;
- correct positional arguments;
- total score, computed from tool, optional-argument, and positional-argument scores;
- exact command match; and
- performance across 23 Kali tool dimensions.

The benchmark supports the following evaluation and training modes:

| Mode | Information available to the model |
| --- | --- |
| `unrestricted` | The query only; the model selects any appropriate Kali/Linux tool. |
| `restricted` | The query and a candidate set of allowed tools. |
| `hinted` | The query, the target tool, and its usage documentation. |

<p align="center">
  <img src="assets/windmill.png" alt="KaliBench tool coverage" width="50%">
</p>

### KaliBench samples

Each benchmark row contains a natural-language query, its canonical command, the expected tool, decomposed optional and positional arguments, and verification metadata.

<p align="center">
  <img src="assets/samples.png" alt="KaliBench samples" width="80%">
</p>

## Artifact Coverage

This revision provides refactored data-construction, training, and evaluation implementations under `src/`. Every automated experimental stage is directly inspectable and runnable; optional adjudication decisions are accepted as auditable JSONL rather than being embedded in source code.

| Experimental stage | Main files |
| --- | --- |
| Constructed and verified data | `KaliBench_data/*.jsonl` |
| Kali documentation extraction and candidate generation | `src/data_creation/generate.py` |
| Documentation-grounded model verification and regeneration | `src/data_creation/verify.py` |
| Isolated terminal execution, rule filtering, and adjudication merge | `src/data_creation/terminal_verify.py` |
| Kali Everything terminal-verification environment | `docker/kali-everything/Dockerfile` |
| Exact deduplication and tool-stratified split finalization | `src/data_creation/finalize.py` |
| Data-construction package and shared utilities | `src/data_creation/__init__.py`, `src/data_creation/common.py` |
| KaliBench SFT | `src/train/sft_kalibench.py` |
| General chat SFT utility | `src/train/sft_general.py` |
| GRPO/RLVR with verifiable rewards | `src/train/grpo_kalibench.py` |
| LoRA merge/export | `src/train/merge_lora.py` |
| Prompt construction for all three modes | `src/prompt.py`, `src/prompt_builder.py` |
| vLLM inference and recovery from interruption | `src/evaluate.py`, `src/vllm_inference.py`, `src/prediction_checkpoint.py`, `src/vllm_runtime.py` |
| Tool/argument/exact-match scoring | `src/model_score.py`, `src/scoring_pipeline.py` |
| Dimension-level scoring | `Scoring/dim_score.py` |
| CSV and LaTeX result tables | `src/format_table_results.py` |
| Shared JSONL I/O | `src/io_utils.py` |
| Data-construction regression tests | `tests/test_data_creation.py` |

The implementations under `src/data_creation/` avoid duplicated utilities and hard-coded paths, expose each stage separately, use resumable checkpoints, preserve audit outputs, and emit the same `model_verification` and `terminal_verification` field structure used by the released data.

## Repository Layout

```text
KaliBench/
├── KaliBench_data/
│   ├── kalibench_verified_train_3504.jsonl
│   ├── kalibench_verified_test_5000.jsonl
│   └── Kali_Tool_Subtools_UsageCode.jsonl
├── docker/
│   └── kali-everything/
│       └── Dockerfile              # full Kali toolset for terminal verification
├── src/
│   ├── data_creation/
│   │   ├── __init__.py             # data-construction package
│   │   ├── common.py               # shared JSONL and OpenAI-compatible API utilities
│   │   ├── generate.py             # documentation extraction, generation, and flattening
│   │   ├── verify.py               # model verification, filtering, and regeneration coverage
│   │   ├── terminal_verify.py      # isolated execution, rule triage, and adjudication merge
│   │   └── finalize.py             # deduplication, stable IDs, and stratified splits
│   ├── evaluate.py                 # end-to-end prompt building, vLLM inference, and scoring
│   ├── prompt.py                   # system and user prompt templates
│   ├── prompt_builder.py           # three-mode prompt and label construction
│   ├── vllm_inference.py           # vLLM chat/generate wrappers
│   ├── vllm_runtime.py             # runtime/cache configuration and safe fallbacks
│   ├── prediction_checkpoint.py    # batch checkpointing and custom_id-based resume
│   ├── model_score.py              # per-example and aggregate benchmark metrics
│   ├── scoring_pipeline.py         # scoring orchestration for saved evaluations
│   ├── format_table_results.py     # organized CSV and LaTeX table export
│   ├── io_utils.py                 # shared JSONL utilities
│   └── train/
│       ├── sft_kalibench.py        # KaliBench-specific SFT
│       ├── grpo_kalibench.py       # GRPO/RLVR and verifiable rewards
│       ├── sft_general.py          # generic messages-JSONL SFT
│       └── merge_lora.py           # merge a LoRA adapter into a 16-bit model
├── Scoring/
│   └── dim_score.py                # 23-dimension aggregation
├── eval_request/                   # legacy two-stage request-generation interface
├── tests/
│   └── test_data_creation.py       # data-construction unit/smoke tests
└── requirements.txt
```

The `src/data_creation/` and `src/evaluate.py` interfaces are the recommended construction and evaluation paths. The older `eval_request/` and `Scoring/model_score.py` entrypoints are retained for compatibility with previously generated files.

## Getting Started

### 1. Clone the anonymous artifact

```bash
git clone <ANONYMOUS_REPOSITORY_URL> KaliBench
cd KaliBench
```

### 2. Create the environment

Python 3.10 is recommended.

```bash
conda create -n kalibench python=3.10
conda activate kalibench
pip install -r requirements.txt
```

Evaluation uses vLLM. Training additionally uses Unsloth, TRL, and optionally Weights & Biases:

```bash
pip install unsloth trl wandb
```

> [!NOTE]
> PyTorch, CUDA, vLLM, FlashInfer/Triton, and Unsloth builds must be compatible with the local GPU and driver. If Weights & Biases is not desired, pass `--report-to none` to the training commands. Data generation/model verification call a hosted OpenAI-compatible endpoint; model downloads and dimension metadata lookup also require network access unless the resources are already cached.

### 3. Check the command-line entrypoints

```bash
python src/data_creation/generate.py --help
python src/data_creation/verify.py --help
python src/data_creation/terminal_verify.py --help
python src/data_creation/finalize.py --help
python src/evaluate.py --help
python src/train/sft_kalibench.py --help
python src/train/grpo_kalibench.py --help
python src/scoring_pipeline.py --help
```

## Dataset and Benchmark Splits

The three released JSONL files are:

| File | Rows | Purpose |
| --- | ---: | --- |
| `KaliBench_data/kalibench_verified_train_3504.jsonl` | 3,504 | SFT and GRPO/RLVR training split |
| `KaliBench_data/kalibench_verified_test_5000.jsonl` | 5,000 | Held-out benchmark evaluation split |
| `KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl` | 2,809 | Kali sub-tool names, parent tool titles, and usage documentation |

The train and test rows contain:

```text
custom_id, query, tool_name, ground_truth_command,
optional_args, positional_args, model_verification, terminal_verification
```

The usage-document rows contain:

```text
source_row, subtool, title, usage_code
```

The verified data can also be accessed from the anonymous Hugging Face dataset page: [anonymous62567/KaliBench-Verified](https://huggingface.co/datasets/anonymous62567/KaliBench-Verified).

## Data Construction

The complete automated construction path is:

```text
Kali tool manuscripts
  → sub-tool usage chunks
  → Qwen3-Max query-command candidates
  → documentation-grounded model verification
  → regeneration of uncovered sub-tools
  → isolated Kali terminal execution and rule-based triage
  → optional reviewed adjudications
  → exact query deduplication and tool-stratified train/test finalization
```

All commands below run from the repository root. The defaults use Qwen3-Max through DashScope's OpenAI-compatible API, matching the data-construction configuration used for the released artifact. The model, API base URL, and API-key environment variable are configurable.

### 1. Extract documentation and generate candidates

Set the API key without placing it in a command-line argument:

```bash
export DASHSCOPE_API_KEY="YOUR_KEY"
```

Run document extraction, request preparation, generation, and response flattening:

```bash
python src/data_creation/generate.py \
  --stage all \
  --source-dataset "anonymous62567/kali-tools" \
  --source-split train \
  --out-dir "$PWD/outputs/data_creation/generation" \
  --model qwen3-max \
  --items-per-tool 10 \
  --temperature 0.5 \
  --max-tokens 4096
```

This creates:

```text
outputs/data_creation/generation/
├── tool_subtools.jsonl             # extracted and deduplicated usage chunks
├── generation_requests.jsonl       # exact model requests
├── generation_responses.jsonl      # append-only API checkpoint
├── generated_labels.jsonl          # flattened query-command candidates
└── generation_parse_errors.jsonl   # responses that could not be parsed
```

Preparation can be reproduced without network access by using the released usage chunks:

```bash
python src/data_creation/generate.py \
  --stage prepare \
  --input-tool-subtools-jsonl "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
  --out-dir "$PWD/outputs/data_creation/generation"
```

The individual `prepare`, `generate`, and `flatten` stages can be rerun separately. API checkpoints are keyed by `custom_id`; completed IDs are skipped, while `--retry-errors` retries IDs whose latest checkpoint contains an API error.

### 2. Verify against the source documentation

The verifier receives each query-command pair together with its matched usage documentation and must return `ACCURATE` or `HALLUCINATED`.

```bash
python src/data_creation/verify.py \
  --stage all \
  --labels-jsonl "$PWD/outputs/data_creation/generation/generated_labels.jsonl" \
  --tool-subtools-jsonl "$PWD/outputs/data_creation/generation/tool_subtools.jsonl" \
  --out-dir "$PWD/outputs/data_creation/verification" \
  --model qwen3-max \
  --temperature 0 \
  --max-tokens 1024
```

This creates:

```text
outputs/data_creation/verification/
├── verification_prompts.jsonl      # exact messages presented to the verifier
├── verification_results.jsonl      # append-only API checkpoint and raw responses
├── model_verified.jsonl            # ACCURATE rows in released metadata format
├── model_rejected.jsonl            # HALLUCINATED rows
├── verification_errors.jsonl       # API, parse, or missing-documentation failures
└── remaining_subtools.jsonl         # usage chunks with no accepted candidate
```

`model_verified.jsonl` stores the verifier evidence as:

```json
{"model_verification":{"model":"qwen3-max","model_output":{"verdict":"ACCURATE","reasons":"..."}}}
```

To reproduce the released construction behavior, unmatched candidates are sent with an empty documentation block. Add `--require-usage` to route them to `verification_errors.jsonl` without an API call instead.

Uncovered sub-tools can be regenerated by passing `remaining_subtools.jsonl` back to the generation stage:

```bash
python src/data_creation/generate.py \
  --stage all \
  --input-tool-subtools-jsonl "$PWD/outputs/data_creation/verification/remaining_subtools.jsonl" \
  --out-dir "$PWD/outputs/data_creation/regeneration"
```

The verification CLI also accepts `--merge-clean` to combine clean files from multiple generation/verification rounds while retaining one latest row per `custom_id`.

### 3. Execute and triage commands in Kali

> [!CAUTION]
> This stage executes every `ground_truth_command`. Run it only inside an isolated Kali container or disposable VM with no sensitive files, credentials, or reachable production systems. The explicit acknowledgement flag is required.

The included image installs the `kali-linux-everything` toolset used for terminal verification. It is a large image, so the initial build requires substantial download time and disk space:

```bash
docker build \
  -t kalibench/kali-everything:latest \
  -f docker/kali-everything/Dockerfile .
```

Run the verifier from the repository root. The command below disables container networking, mounts the repository read-only, and makes only the terminal-output directory writable:

```bash
mkdir -p "$PWD/outputs/data_creation/terminal"

docker run --rm \
  --name kalibench-terminal-verification \
  --hostname kali-everything \
  --network none \
  --pids-limit 256 \
  --memory 8g \
  --cpus 4 \
  --security-opt no-new-privileges \
  --cap-add NET_ADMIN \
  --cap-add NET_RAW \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  --mount type=bind,src="$PWD/outputs/data_creation/terminal",dst=/workspace/outputs/data_creation/terminal \
  --workdir /workspace \
  kalibench/kali-everything:latest \
  python3 src/data_creation/terminal_verify.py \
    --input-jsonl /workspace/outputs/data_creation/verification/model_verified.jsonl \
    --out-dir /workspace/outputs/data_creation/terminal \
    --timeout-sec 20 \
    --max-stream-bytes 204800 \
    --acknowledge-command-execution
```

Networking is disabled by default so commands cannot reach external targets. Network-dependent commands may therefore be classified as runtime failures. If external connectivity is necessary for a specific verification experiment, attach the container only to a controlled, isolated test network and document that deviation.

For every row, the executor records timestamps, duration, timeout status, exit code, bounded stdout/stderr, truncation metadata, classifier label, category, reason, and matched error pattern. The rule-based categories are:

| Category | Meaning |
| --- | --- |
| `pass_review` | Executed successfully, timed out, or failed only because a runtime resource such as a file/device/service/target was unavailable. |
| `reject_hallucinated` | The shell/tool reported a missing command, unsupported flag, invalid option, or malformed required argument. |
| `undecided` | No reliable execution decision could be made. |

Outputs:

```text
outputs/data_creation/terminal/
├── terminal_results.jsonl          # append-only execution checkpoint
├── terminal_accepted.jsonl         # pass_review rows
├── terminal_rejected.jsonl         # hallucinated tool/parameter rows
├── terminal_undecided.jsonl        # rows requiring inspection
└── terminal_summary.json
```

Optional reviewed decisions can be supplied as an auditable JSONL file:

```json
{"custom_id":"candidate-000001-000","category":"pass_review","reason":"No unsupported capability was found during inspection."}
```

Apply those decisions by adding:

```text
--adjudications-jsonl /path/to/adjudications.jsonl
```

The merged rows record `ai_review_category` and `ai_review_reason` inside `terminal_verification`, matching the provenance structure in the released splits.

### 4. Deduplicate and create the final splits

The finalizer requires both an `ACCURATE` model verdict and terminal category `pass_review`, removes duplicate normalized queries, assigns stable `sample_` IDs, and creates a deterministic tool-stratified split.

```bash
python src/data_creation/finalize.py \
  --inputs "$PWD/outputs/data_creation/terminal/terminal_accepted.jsonl" \
  --out-dir "$PWD/outputs/data_creation/final" \
  --test-size 5000 \
  --min-test-per-tool 3 \
  --max-test-per-tool 4 \
  --seed 3407
```

For the released 8,504 accepted rows, these settings produce 3,504 training rows and 5,000 test rows while retaining all 1,642 tools in the test pool. The finalizer writes the combined data, train/test splits, rejected/duplicate audit rows, and a JSON summary with row and tool counts.

Because hosted-model outputs can change across model revisions or service updates, a fresh API run may not reproduce every generated sentence byte-for-byte. The released JSONL files are the canonical outputs of the documented construction and review process; the code reproduces the transformations, checkpoints, verification schema, filtering rules, and split policy.

### 5. Validate the construction implementation

The tests do not call external APIs and execute only a harmless local `printf` smoke command:

```bash
python -m unittest tests.test_data_creation -v
```

## Training

All examples below are run from the repository root. Explicit paths are used so that the commands refer to the filenames shipped in this artifact.

Set the base model to a local path or a Hugging Face model ID:

```bash
export BASE_MODEL="/path/to/base-model"
```

The main paper-reproduction path is:

```text
constructed and verified training split → KaliBench SFT → GRPO/RLVR → merged model → evaluation
```

### 1. Supervised fine-tuning

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

### 2. GRPO/RLVR

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

### 3. Optional training utilities

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

## Evaluation

`src/evaluate.py` is the end-to-end evaluation entrypoint. It:

1. constructs prompts and scorer labels for one benchmark mode;
2. runs local vLLM inference;
3. checkpoints predictions after every batch;
4. resumes missing predictions by `custom_id` after interruption;
5. writes raw and scorer-compatible predictions;
6. calculates tool, optional F1, positional F1, total, exact-match, and format-error metrics; and
7. optionally calculates scores for the 23 Kali tool dimensions.

Use a different output directory for every model/mode pair. This prevents a resumed run from mixing predictions generated with different prompts.

### Evaluate one mode

```bash
python src/evaluate.py \
  --mode hinted \
  --input "$PWD/KaliBench_data/kalibench_verified_test_5000.jsonl" \
  --subtools "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
  --model "$PWD/outputs/models/kalibench_grpo/merged" \
  --output-dir "$PWD/outputs/evaluate/kalibench_grpo/hinted" \
  --candidate-seed 42 \
  --seed 3407 \
  --temperature 0.2 \
  --batch-size 128 \
  --tensor-parallel-size 1
```

In hinted mode, evaluation intentionally forces one candidate tool and includes its usage information. In restricted mode, `--candidate-tools 20` is the default. Unrestricted mode does not use a candidate-tool list.

### Evaluate all three modes

```bash
for MODE in unrestricted restricted hinted; do
  python src/evaluate.py \
    --mode "$MODE" \
    --input "$PWD/KaliBench_data/kalibench_verified_test_5000.jsonl" \
    --subtools "$PWD/KaliBench_data/Kali_Tool_Subtools_UsageCode.jsonl" \
    --model "$PWD/outputs/models/kalibench_grpo/merged" \
    --output-dir "$PWD/outputs/evaluate/kalibench_grpo/$MODE" \
    --candidate-tools 20 \
    --candidate-seed 42 \
    --seed 3407 \
    --temperature 0.2 \
    --batch-size 128 \
    --tensor-parallel-size 1
done
```

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

### Evaluation outputs

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

## Scoring and Result Generation

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

Dimension-level scoring uses the tool-to-title mapping in the released usage file and the anonymous Hugging Face metadata for the 23 metapackage labels:

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
