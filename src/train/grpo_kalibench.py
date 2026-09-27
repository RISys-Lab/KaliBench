#!/usr/bin/env python3
"""GRPO training entrypoint for KaliBench tool-calling tasks.

This script mirrors the structure of notebooks/grpo_math.py while adapting the
dataset preprocessing and reward functions for KaliBench command generation.

Highlights:
- Unrestricted, restricted, and hinted prompt construction
- Verifiable rewards aligned with src/model_score.py metrics
- Fine-grained reward components: tool score, optional F1, positional F1,
  and exact match (with heavier default weight)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import re
import shlex
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Optional

import numpy as np

LOGGER = logging.getLogger("grpo_kalibench")

FORMAT_ERROR_TOKEN = "[FORMAT_ERROR]"


# -----------------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------------


KALIBENCH_PROMPT = """You are a cybersecurity function-calling AI model.
You have access to the following tool:
<tools>
[{'type':'function','function':{'name':'run_terminal','description':'Execute a shell command in a Kali/Linux terminal and return stdout, stderr, and exit code.'}}]
</tools>

TASK:
Given a USER QUERY, generate the single most accurate shell command using any appropriate Kali/Linux tool(s) to solve the query.

REQUIREMENTS:
1. Use the correct command-line tool(s) appropriate for the task.
2. Include all required optional arguments (flags beginning with '-' or '--') necessary to accomplish the task.
3. Correctly pair option keys and values (e.g., `--port 80`, `-A INPUT`, or `--flag=value`).
4. Preserve and include any positional arguments (e.g., IPs, filenames, interfaces).
5. Do NOT invent flags/options that do not exist for real Kali/Linux tools.
7. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

OUTPUT FORMAT:
<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""


KALIBENCH_USER_TEMPLATE = """USER QUERY: "{query}"

Generate the single most accurate shell command for the query.
Your response must follow the required structure:

<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""


KALIBENCH_RESTRICTED_PROMPT = """You are a cybersecurity function-calling AI model.

TASK:
Given a USER QUERY and a list of allowed tools provided in the user message, generate the single most accurate shell command that solves the query using ONLY the allowed tools.

REQUIREMENTS:
1. Use the correct command-line tool from the allowed list.
2. Include all required optional arguments (flags beginning with '-' or '--') necessary to accomplish the task.
3. Correctly pair option keys and values (e.g., `--port 80`, `-A INPUT`, or `--flag=value`).
4. Preserve and include any positional arguments (e.g., target IPs, filenames, interfaces).
5. Do NOT invent flags/options that do not exist for real Kali/Linux tools.
6. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

OUTPUT FORMAT:
<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""


KALIBENCH_USER_RESTRICTED_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: "{query}"

Generate the single most accurate shell command for the query using ONLY the allowed tools listed above.
Your response must follow the required structure:

<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""


KALIBENCH_USER_HINTED_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: "{query}"

Generate the single most accurate shell command for the query using ONLY the allowed tools listed above.
Use the provided usage hints to select the correct tool and arguments.
Your response must follow the required structure:

<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""


REASONING_START = "<think>"
REASONING_END = "</think>"
SOLUTION_START = "<output>"
SOLUTION_END = "</output>"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class RewardWeights:
    format_exact: float
    format_approx: float
    tool_score: float
    optional_f1: float
    positional_f1: float
    exact_match: float


@dataclass
class ModeSpec:
    name: str
    fraction: float


@dataclass
class TrainConfig:
    mode_specs: List[ModeSpec]
    model_name: str
    dataset_path: Path
    subtools_path: Path
    output_dir: Path
    adapter_output_dir: Path
    cache_root: Path | None
    candidate_tools: int
    candidate_seed: int
    include_usage_in_prompt: bool
    max_usage_tokens: int
    shuffle_dataset: bool
    seed: int
    max_seq_length: int
    lora_rank: int
    gpu_memory_utilization: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    optim: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    num_train_epochs: float
    logging_steps: int
    save_steps: float | None
    max_steps: int | None
    prompt_length_quantile: float
    attention_backend: str
    load_in_4bit: bool
    fast_inference: bool
    enforce_eager: bool
    report_to: str
    wandb_project: str | None
    wandb_entity: str | None
    wandb_mode: str | None
    debug_reward_print_every: int
    debug_dataset: bool
    reward_weights: RewardWeights


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"yes", "true", "t", "1", "y"}:
        return True
    if s in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def parse_optional_positive_float(value: str) -> float | None:
    s = str(value).strip().lower()
    if s in {"none", "no", "off", "false"}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0, or use 'none' to disable saving.")
    return parsed


VALID_MODES = {"unrestricted", "restricted", "hinted"}


def _flatten_mode_args(mode_args: Any) -> List[str]:
    if not mode_args:
        return []
    flattened: List[str] = []
    for entry in mode_args:
        if isinstance(entry, (list, tuple)):
            flattened.extend(entry)
        else:
            flattened.append(entry)
    return [str(item).strip() for item in flattened if str(item).strip()]


def _parse_mode_spec(raw: str) -> ModeSpec:
    name_part, sep, fraction_part = raw.partition(":")
    name = name_part.strip().lower()
    if name not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{name}'. Expected one of: {sorted(VALID_MODES)}")

    fraction = 1.0
    if sep:
        fraction_text = fraction_part.strip()
        if not fraction_text:
            raise ValueError(f"Missing fraction after ':' in mode '{raw}'")
        try:
            fraction = float(fraction_text)
        except ValueError as exc:
            raise ValueError(f"Mode fraction must be a float in [0, 1], got '{fraction_text}'") from exc

    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"Mode fraction must be within [0, 1], got {fraction}")

    return ModeSpec(name=name, fraction=fraction)


def _parse_mode_specs(mode_args: Any, default_modes: Sequence[str]) -> tuple[List[ModeSpec], List[str]]:
    raw_modes = _flatten_mode_args(mode_args)
    if not raw_modes:
        raw_modes = list(default_modes)

    specs: List[ModeSpec] = []
    seen: set[str] = set()
    duplicates: List[str] = []
    for raw in raw_modes:
        spec = _parse_mode_spec(raw)
        if spec.name in seen:
            duplicates.append(spec.name)
            continue
        seen.add(spec.name)
        specs.append(spec)

    if not specs:
        raise ValueError("At least one mode must be provided.")

    return specs, duplicates


def _format_mode_specs(mode_specs: Sequence[ModeSpec]) -> str:
    return ",".join(f"{spec.name}:{spec.fraction:g}" for spec in mode_specs)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a GRPO model for KaliBench tool calling")

    parser.add_argument(
        "--mode",
        action="append",
        nargs="+",
        default=None,
        help=(
            "Training modes (repeatable). Use mode or mode:fraction (0-1). "
            "Example: --mode hinted:1.0 --mode restricted:0.5 --mode unrestricted:0.5"
        ),
    )
    parser.add_argument("--model-name", default="RISys-Lab/RedSage-Qwen3-8B-Ins")
    parser.add_argument(
        "--dataset-path",
        default="KaliBench_data/verified/kalibench_verified_train_3504.jsonl",
        help="Training JSONL path for KaliBench.",
    )
    parser.add_argument(
        "--subtools-path",
        default="KaliBench_data/Kali_Tool_Subtools_UsageCode_Final.jsonl",
        help="Tool usage JSONL for hinted-mode prompt hints.",
    )

    parser.add_argument("--output-dir", default="outputs/models/RedSage-Qwen3-8B-Ins-kalibench_grpo_fixed")
    parser.add_argument("--adapter-output-dir", default="outputs/adapters/RedSage-Qwen3-8B-Ins-kalibench_grpo_fixed")

    parser.add_argument("--cache-root", default=os.environ.get("TRAIN_CACHE_ROOT"))
    parser.add_argument(
        "--candidate-tools",
        type=int,
        default=20,
        help="Number of candidate tools for restricted mode. Hinted mode always uses 1.",
    )
    parser.add_argument("--candidate-seed", type=int, default=42)
    parser.add_argument("--include-usage-in-prompt", type=str2bool, default=True)
    parser.add_argument("--max-usage-tokens", type=int, default=4096 * 2)
    parser.add_argument(
        "--shuffle-dataset",
        type=str2bool,
        default=True,
        help="Shuffle combined rows across all modes before tokenization.",
    )

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-seq-length", type=int, default=12288)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)

    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--optim", default="adamw_8bit")

    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument(
        "--save-steps",
        type=parse_optional_positive_float,
        default=0.5,
        help="Checkpoint cadence in epochs. Example: 1.0 = every epoch, 0.5 = every half epoch. Use 'none' to disable.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--prompt-length-quantile", type=float, default=1.0)

    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--disable-fast-inference", action="store_true")
    parser.add_argument("--disable-enforce-eager", action="store_true")

    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "kalibench_grpo"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))

    parser.add_argument("--debug-reward-print-every", type=int, default=20)

    parser.add_argument(
        "--debug-dataset",
        action="store_true",
        help="Print prompt token length stats after chat template and exit.",
    )

    parser.add_argument("--reward-weight-format-exact", type=float, default=2.0)
    parser.add_argument("--reward-weight-format-approx", type=float, default=1.0)
    parser.add_argument("--reward-weight-tool-score", type=float, default=1.0)
    parser.add_argument("--reward-weight-optional-f1", type=float, default=1.5)
    parser.add_argument("--reward-weight-positional-f1", type=float, default=1.5)
    parser.add_argument("--reward-weight-exact-match", type=float, default=2.0)

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    default_modes = ["hinted", "restricted", "unrestricted"]
    mode_specs, duplicate_modes = _parse_mode_specs(args.mode, default_modes)
    if duplicate_modes:
        LOGGER.warning("Duplicate modes ignored: %s", ", ".join(sorted(set(duplicate_modes))))

    repo_root = Path(__file__).resolve().parents[1]

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = (repo_root / dataset_path).resolve()

    subtools_path = Path(args.subtools_path)
    if not subtools_path.is_absolute():
        subtools_path = (repo_root / subtools_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / args.output_dir).resolve()

    adapter_output_dir = Path(args.adapter_output_dir)
    if not adapter_output_dir.is_absolute():
        adapter_output_dir = (repo_root / args.adapter_output_dir).resolve()

    return TrainConfig(
        mode_specs=mode_specs,
        model_name=args.model_name,
        dataset_path=dataset_path,
        subtools_path=subtools_path,
        output_dir=output_dir,
        adapter_output_dir=adapter_output_dir,
        cache_root=Path(args.cache_root).expanduser() if args.cache_root else None,
        candidate_tools=args.candidate_tools,
        candidate_seed=args.candidate_seed,
        include_usage_in_prompt=args.include_usage_in_prompt,
        max_usage_tokens=args.max_usage_tokens,
        shuffle_dataset=args.shuffle_dataset,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        prompt_length_quantile=args.prompt_length_quantile,
        attention_backend=args.attention_backend,
        load_in_4bit=args.load_in_4bit,
        fast_inference=not args.disable_fast_inference,
        enforce_eager=not args.disable_enforce_eager,
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        debug_reward_print_every=args.debug_reward_print_every,
        debug_dataset=args.debug_dataset,
        reward_weights=RewardWeights(
            format_exact=args.reward_weight_format_exact,
            format_approx=args.reward_weight_format_approx,
            tool_score=args.reward_weight_tool_score,
            optional_f1=args.reward_weight_optional_f1,
            positional_f1=args.reward_weight_positional_f1,
            exact_match=args.reward_weight_exact_match,
        ),
    )


# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        LOGGER.exception("Failed to fully seed torch")


def configure_environment(config: TrainConfig) -> None:
    """Prepare environment variables before importing GPU-heavy libraries."""
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    user = os.environ.get("USER", "user")

    os.environ.pop("TMPDIR", None)
    tempfile.tempdir = None

    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
    os.environ["FLASHINFER_JIT_VERBOSE"] = os.environ.get("FLASHINFER_JIT_VERBOSE", "0")

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/{user}/torchinductor_{job_id}"
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/{user}/triton_{job_id}"

    os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

    if config.cache_root is not None:
        config.cache_root.mkdir(parents=True, exist_ok=True)
        cache_paths = {
            "XDG_CACHE_HOME": config.cache_root / "xdg",
            "VLLM_CACHE_ROOT": config.cache_root / "vllm",
            "HF_HOME": config.cache_root / "hf",
            "UV_CACHE_DIR": config.cache_root / "uv",
        }
        for env_name, path in cache_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[env_name] = str(path)

    gcc = shutil.which("gcc")
    gxx = shutil.which("g++")
    if gcc:
        os.environ["CC"] = gcc
    if gxx:
        os.environ["CXX"] = gxx
        os.environ["CUDAHOSTCXX"] = gxx
        os.environ["NVCC_CCBIN"] = gxx

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    if config.wandb_entity:
        os.environ["WANDB_ENTITY"] = config.wandb_entity
    if config.wandb_mode:
        os.environ["WANDB_MODE"] = config.wandb_mode

    LOGGER.info("tempfile.gettempdir()=%s", tempfile.gettempdir())
    LOGGER.info("TORCHINDUCTOR_CACHE_DIR=%s", os.environ["TORCHINDUCTOR_CACHE_DIR"])
    LOGGER.info("TRITON_CACHE_DIR=%s", os.environ["TRITON_CACHE_DIR"])


def maybe_init_wandb(config: TrainConfig) -> None:
    if config.report_to != "wandb":
        return
    try:
        import wandb

        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"])
        else:
            LOGGER.info("WANDB_API_KEY not set; assuming existing auth or offline mode.")
    except Exception:
        LOGGER.exception("Failed to initialize Weights & Biases")
        raise


# -----------------------------------------------------------------------------
# Data loading and prompt construction
# -----------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{lineno}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{lineno}, got {type(row)}")
            rows.append(row)
    return rows


def parse_possible_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def load_tool_usages_jsonl(records: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    tool_map: Dict[str, str] = {}
    for rec in records:
        sub = rec.get("subtool") or rec.get("tool") or rec.get("title") or rec.get("name")
        usage = rec.get("usage_code") or rec.get("manuscript") or rec.get("usage") or rec.get("description")
        if not sub:
            continue
        tool_map[str(sub).strip().lower()] = str(usage or "")
    return tool_map


def make_allowed_str_with_usage(
    tools: Sequence[str],
    usage_for_tools: Dict[str, str],
    include_usage: bool,
    max_usage_tokens: int | None,
    tokenizer: Any | None = None,
) -> str:
    def truncate_by_tokens(text: str, max_tokens: int) -> str:
        if tokenizer is None:
            tokens = text.split()
            if len(tokens) <= max_tokens:
                return text
            return " ".join(tokens[:max_tokens]).rstrip() + "\n...(truncated)"

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        clipped_ids = token_ids[:max_tokens]
        clipped_text = tokenizer.decode(clipped_ids, skip_special_tokens=True)
        return clipped_text.rstrip() + "\n...(truncated)"

    lines: List[str] = []
    for tool_name in tools:
        lines.append(f"- {tool_name}")
        if not include_usage:
            continue
        usage = usage_for_tools.get(tool_name, "")
        if not usage:
            continue
        usage_snippet = usage
        if max_usage_tokens is not None and max_usage_tokens > 0:
            usage_snippet = truncate_by_tokens(usage, max_usage_tokens)
        for usage_line in usage_snippet.strip().splitlines():
            lines.append(f"  {usage_line}")
    return "\n".join(lines)


def extract_query(row: Dict[str, Any]) -> str:
    candidates = [row.get("query"), row.get("prompt")]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def to_json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_training_rows(
    records: Sequence[Dict[str, Any]],
    mode_specs: Sequence[ModeSpec],
    subtools_path: Path,
    candidate_tools: int,
    candidate_seed: int,
    include_usage_in_prompt: bool,
    max_usage_tokens: int,
    usage_tokenizer: Any | None = None,
) -> List[Dict[str, Any]]:
    if not mode_specs:
        raise ValueError("At least one mode must be provided.")

    valid_modes = {"unrestricted", "restricted", "hinted"}
    unknown_modes = [spec.name for spec in mode_specs if spec.name not in valid_modes]
    if unknown_modes:
        raise ValueError(f"Unsupported mode(s): {unknown_modes}")

    tool_usage_map: Dict[str, str] = {}
    if subtools_path.exists():
        subtool_records = read_jsonl(subtools_path)
        tool_usage_map = load_tool_usages_jsonl(subtool_records)
    elif any(spec.name == "hinted" for spec in mode_specs):
        LOGGER.warning("Subtools file not found at %s; hinted mode will run without usage snippets.", subtools_path)

    tool_pool = sorted(
        {
            str(r.get("tool_name") or r.get("tool") or "").strip()
            for r in records
            if str(r.get("tool_name") or r.get("tool") or "").strip()
        }
    )
    tool_pool_by_lower = {tool.lower(): tool for tool in tool_pool}

    if any(spec.name in {"restricted", "hinted"} for spec in mode_specs) and not tool_pool:
        raise ValueError("No tools found in records; cannot build restricted/hinted prompts.")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    per_mode_counts: Dict[str, int] = {}
    print_per_mode = {spec.name: False for spec in mode_specs}

    for mode_idx, mode_spec in enumerate(mode_specs):
        mode = mode_spec.name
        fraction = mode_spec.fraction
        rng = random.Random(candidate_seed + mode_idx)
        mode_rows: List[Dict[str, Any]] = []

        for idx, row in enumerate(records, start=1):
            custom_id_base = str(row.get("custom_id") or f"row-{idx}").strip()
            custom_id = f"{custom_id_base}::{mode}"
            query = extract_query(row)
            tool_name = str(row.get("tool_name") or row.get("tool") or "").strip()
            gt_cmd = str(row.get("ground_truth_command") or "").strip()

            if not query or not tool_name or not gt_cmd:
                skipped += 1
                continue

            optional_args = parse_possible_json_field(row.get("optional_args"), {})
            positional_args = parse_possible_json_field(row.get("positional_args"), [])

            if mode == "unrestricted":
                system_prompt = KALIBENCH_PROMPT
                user_prompt = KALIBENCH_USER_TEMPLATE.format(query=query)
            else:
                effective_candidate_tools = 1 if mode == "hinted" else candidate_tools
                if effective_candidate_tools <= 0:
                    raise ValueError("--candidate-tools must be > 0 for restricted mode")

                target_tool_lower = tool_name.lower()
                if target_tool_lower not in tool_pool_by_lower:
                    raise ValueError(
                        f"Gold tool '{tool_name}' for custom_id '{custom_id_base}' is missing from tool pool."
                    )

                candidate_count = min(effective_candidate_tools, len(tool_pool))
                target_tool = tool_pool_by_lower[target_tool_lower]
                non_target_tools = [t for t in tool_pool if t.lower() != target_tool_lower]
                sampled_tools = rng.sample(non_target_tools, k=max(candidate_count - 1, 0))
                chosen_chunk = sampled_tools + [target_tool]
                rng.shuffle(chosen_chunk)

                usage_for_tools = {t: tool_usage_map.get(t.lower(), "") for t in chosen_chunk}
                include_usage = mode == "hinted" and max_usage_tokens != 0
                allowed_tools_str = make_allowed_str_with_usage(
                    chosen_chunk,
                    usage_for_tools,
                    include_usage=include_usage,
                    max_usage_tokens=max_usage_tokens if max_usage_tokens > 0 else None,
                    tokenizer=usage_tokenizer,
                )

                system_prompt = KALIBENCH_RESTRICTED_PROMPT
                if mode == "restricted":
                    user_prompt = KALIBENCH_USER_RESTRICTED_TEMPLATE.format(
                        allowed_tools=allowed_tools_str,
                        query=query,
                    )
                else:
                    user_prompt = KALIBENCH_USER_HINTED_TEMPLATE.format(
                        allowed_tools=allowed_tools_str,
                        query=query,
                    )

            mode_rows.append(
                {
                    "custom_id": custom_id,
                    "mode": mode,
                    "query": query,
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "tool_name": tool_name,
                    "ground_truth_command": gt_cmd,
                    "optional_args": to_json_string(optional_args),
                    "positional_args": to_json_string(positional_args),
                }
            )
            if print_per_mode[mode] == False:
                LOGGER.info("Example prompt for mode '%s':\n%s", mode, user_prompt)
                print_per_mode[mode] = True
        selected_rows = mode_rows
        if 0.0 <= fraction < 1.0:
            rng.shuffle(mode_rows)
            keep_count = int(len(mode_rows) * fraction)
            keep_count = max(0, min(keep_count, len(mode_rows)))
            selected_rows = mode_rows[:keep_count]
            if fraction > 0.0 and mode_rows and keep_count == 0:
                LOGGER.warning(
                    "Mode '%s' fraction %.3f produced 0 rows after sampling.",
                    mode,
                    fraction,
                )

        rows.extend(selected_rows)
        mode_count = len(selected_rows)

        per_mode_counts[mode] = mode_count

    if skipped:
        LOGGER.warning("Skipped %d rows with missing query/tool_name/ground_truth_command", skipped)

    if not rows:
        raise ValueError("No valid rows were built for training.")

    LOGGER.info("Prepared %d training rows across modes=%s", len(rows), per_mode_counts)
    return rows


# -----------------------------------------------------------------------------
# model_score.py-compatible scoring helpers
# -----------------------------------------------------------------------------


def extract_command_from_output(text: str) -> str:
    """Extract the last command inside <output>...</output>."""
    if not isinstance(text, str):
        text = str(text)

    if "</think>" in text:
        text = text.split("</think>")[-1]

    matches = re.findall(r"<output>\s*(.*?)\s*</output>", text, re.S)
    if not matches:
        return ""
    return matches[-1].strip()


def parse_optional_args(raw: Any) -> Dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {item: None for item in raw}
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return loaded
            if isinstance(loaded, list):
                return {item: None for item in loaded}
            return {}
        except Exception:
            return {raw: None}
    return {}


def parse_positional_args(raw: Any) -> List[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return [str(x) for x in loaded]
            return []
        except Exception:
            return [raw]
    return []


def expand_alias_group(alias_str: str) -> List[str]:
    return alias_str.split("|")


def get_flags_with_values(gt_optional: Dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    if not gt_optional:
        return flags
    for key, val in gt_optional.items():
        if val is not None:
            flags.update(expand_alias_group(str(key)))
    return flags


def parse_command(cmd: str, flags_with_values: set[str] | None = None) -> tuple[Optional[str], Optional[List[str]], Optional[List[str]]]:
    if flags_with_values is None:
        flags_with_values = set()

    try:
        parts = shlex.split(cmd.strip())
    except ValueError:
        parts = cmd.strip().split()

    if not parts:
        return None, None, None

    tool = parts[0]
    optional: List[str] = []
    positional: List[str] = []

    i = 1
    while i < len(parts):
        token = parts[i]
        if token.startswith("-"):
            if "=" in token:
                flag_part, _value_part = token.split("=", 1)
                optional.append(flag_part)
            else:
                optional.append(token)
                if token in flags_with_values:
                    if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                        i += 1
        else:
            positional.append(token)
        i += 1

    return tool, optional, positional


def optional_f1(gt_alias_dict: Dict[str, Any], pred_optional: List[str]) -> float:
    if len(gt_alias_dict) == 0 and len(pred_optional) == 0:
        return 1.0
    if len(gt_alias_dict) == 0:
        return 0.0
    if len(pred_optional) == 0:
        return 0.0

    gt_groups = [expand_alias_group(group) for group in gt_alias_dict.keys()]
    pred_set = Counter(pred_optional)

    tp = 0
    for group in gt_groups:
        if any(alias in pred_set for alias in group):
            tp += 1

    gt_pos = len(gt_groups)
    pred_pos = len(pred_optional)

    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / gt_pos if gt_pos else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def positional_f1(gt_positional: List[str], pred_positional: List[str]) -> float:
    if len(gt_positional) == 0 and len(pred_positional) == 0:
        return 1.0
    if len(gt_positional) == 0:
        return 0.0
    if len(pred_positional) == 0:
        return 0.0

    gt_set = Counter(gt_positional)
    pred_set = Counter(pred_positional)

    tp = sum((gt_set & pred_set).values())
    fp = sum((pred_set - gt_set).values())
    fn = sum((gt_set - pred_set).values())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(gt_cmd: str, gt_alias_dict: Dict[str, Any], pred_cmd: str) -> bool:
    flags_with_values = get_flags_with_values(gt_alias_dict)

    pred_tool, pred_opt, pred_pos = parse_command(pred_cmd, flags_with_values)
    gt_tool, _gt_opt, gt_pos = parse_command(gt_cmd, flags_with_values)

    if pred_tool != gt_tool:
        return False
    if pred_pos != gt_pos:
        return False

    for alias_group in gt_alias_dict.keys():
        aliases = expand_alias_group(str(alias_group))
        if not any(alias in pred_opt for alias in aliases):
            return False

    allowed: set[str] = set()
    for group in gt_alias_dict.keys():
        allowed.update(expand_alias_group(str(group)))

    for pred_flag in pred_opt:
        if pred_flag not in allowed:
            return False

    return True


def score_components(
    gt_tool: str,
    gt_cmd: str,
    gt_optional_raw: Any,
    gt_positional_raw: Any,
    model_reply: str,
) -> Dict[str, float]:
    pred_cmd = extract_command_from_output(model_reply)

    gt_optional = parse_optional_args(gt_optional_raw)
    gt_positional = parse_positional_args(gt_positional_raw)

    flags_with_values = get_flags_with_values(gt_optional)
    pred_tool, pred_opt, pred_pos = parse_command(pred_cmd, flags_with_values)

    if pred_tool is None:
        tool_score = -1.0
    else:
        tool_score = 1.0 if pred_tool == gt_tool else 0.0

    if pred_opt is None:
        optional_score = -1.0
    else:
        optional_score = optional_f1(gt_optional, pred_opt)
    if pred_pos is None:
        positional_score = -1.0
    else:
        positional_score = positional_f1(gt_positional, pred_pos)

    if pred_tool is None or pred_opt is None or pred_pos is None:
        exact_score = -1.0
    else:
        exact_score = 1.0 if exact_match(gt_cmd, gt_optional, pred_cmd) else 0.0

    return {
        "tool_score": tool_score,
        "optional_f1": optional_score,
        "positional_f1": positional_score,
        "exact_match": exact_score,
    }


# -----------------------------------------------------------------------------
# Reward functions
# -----------------------------------------------------------------------------


class KaliBenchRewardFunctions:
    def __init__(self, weights: RewardWeights, tokenizer: Any, debug_print_every: int = 0) -> None:
        self.weights = weights
        self.tokenizer = tokenizer
        self.debug_print_every = max(0, debug_print_every)
        self.debug_counter = 0

        eos_token = getattr(tokenizer, "eos_token", None)
        eos_suffix = rf"(?:{re.escape(eos_token)})?" if eos_token else ""
        solution_end_regex = rf"{re.escape(SOLUTION_END)}[\s]{{0,}}{eos_suffix}"
        self.match_format = re.compile(
            rf"{re.escape(REASONING_END)}.*?"
            rf"{re.escape(SOLUTION_START)}(.+?){solution_end_regex}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL,
        )

    @staticmethod
    def _completion_text(completion: Any) -> str:
        if isinstance(completion, list) and completion:
            first = completion[0]
            if isinstance(first, dict):
                return str(first.get("content") or "")
            return str(first)
        if isinstance(completion, dict):
            return str(completion.get("content") or "")
        return str(completion)

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 3000) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head_len = max_chars // 2
        tail_len = max_chars - head_len
        head = text[:head_len].rstrip()
        tail = text[-tail_len:].lstrip()
        return f"{head}\n(truncated-for-vis)...\n{tail}"

    def _format_prompt(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    return self.tokenizer.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except Exception:
                    pass
            lines: List[str] = []
            for item in prompt:
                if isinstance(item, dict):
                    role = item.get("role") or ""
                    content = item.get("content") or ""
                    if role:
                        lines.append(f"{role}: {content}")
                    else:
                        lines.append(str(content))
                else:
                    lines.append(str(item))
            return "\n".join(lines)
        if isinstance(prompt, dict):
            for key in ("prompt", "messages", "text"):
                if key in prompt:
                    return self._format_prompt(prompt[key])
        return str(prompt)

    def _batch_scores(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        tool_name: Sequence[str],
        ground_truth_command: Sequence[str],
        optional_args: Sequence[Any],
        positional_args: Sequence[Any],
    ) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []

        for response_obj, gt_tool, gt_cmd, gt_opt, gt_pos in zip(
            completions,
            tool_name,
            ground_truth_command,
            optional_args,
            positional_args,
        ):
            response_text = self._completion_text(response_obj)
            comps = score_components(
                gt_tool=str(gt_tool),
                gt_cmd=str(gt_cmd),
                gt_optional_raw=gt_opt,
                gt_positional_raw=gt_pos,
                model_reply=response_text,
            )
            out.append(comps)

        if self.debug_print_every and out and self.debug_counter % self.debug_print_every == 0:
            sample_prompt = self._format_prompt(prompts[0]) if prompts else ""
            sample_reply = self._completion_text(completions[0]) if completions else ""
            if sample_prompt:
                LOGGER.info("Reward debug prompt:\n%s", self._truncate_text(sample_prompt))
            if sample_reply:
                LOGGER.info("Reward debug model reply:\n%s", self._truncate_text(sample_reply))
            LOGGER.info(
                "Reward debug | tool=%.3f optional=%.3f positional=%.3f exact=%.3f",
                out[0]["tool_score"],
                out[0]["optional_f1"],
                out[0]["positional_f1"],
                out[0]["exact_match"],
            )
        self.debug_counter += 1

        return out

    def reward_format_exact(self, completions: Sequence[Any], **_: Any) -> List[float]:
        scores: List[float] = []
        for completion in completions:
            response = self._completion_text(completion)
            score = 1.0 if self.match_format.search(response) is not None else 0.0
            scores.append(self.weights.format_exact * score)
        return scores

    def reward_format_approx(self, completions: Sequence[Any], **_: Any) -> List[float]:
        scores: List[float] = []
        for completion in completions:
            response = self._completion_text(completion)
            score = 0.0
            score += 0.5 if response.count(REASONING_END) == 1 else -1.0
            score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
            score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
            scores.append(self.weights.format_approx * score)
        return scores

    def reward_tool_score(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        tool_name: Sequence[str],
        ground_truth_command: Sequence[str],
        optional_args: Sequence[Any],
        positional_args: Sequence[Any],
        **_: Any,
    ) -> List[float]:
        scores = self._batch_scores(prompts, completions, tool_name, ground_truth_command, optional_args, positional_args)
        return [self.weights.tool_score * s["tool_score"] for s in scores]

    def reward_optional_f1(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        tool_name: Sequence[str],
        ground_truth_command: Sequence[str],
        optional_args: Sequence[Any],
        positional_args: Sequence[Any],
        **_: Any,
    ) -> List[float]:
        scores = self._batch_scores(prompts, completions, tool_name, ground_truth_command, optional_args, positional_args)
        return [self.weights.optional_f1 * s["optional_f1"] for s in scores]

    def reward_positional_f1(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        tool_name: Sequence[str],
        ground_truth_command: Sequence[str],
        optional_args: Sequence[Any],
        positional_args: Sequence[Any],
        **_: Any,
    ) -> List[float]:
        scores = self._batch_scores(prompts, completions, tool_name, ground_truth_command, optional_args, positional_args)
        return [self.weights.positional_f1 * s["positional_f1"] for s in scores]

    def reward_exact_match(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        tool_name: Sequence[str],
        ground_truth_command: Sequence[str],
        optional_args: Sequence[Any],
        positional_args: Sequence[Any],
        **_: Any,
    ) -> List[float]:
        scores = self._batch_scores(prompts, completions, tool_name, ground_truth_command, optional_args, positional_args)
        return [self.weights.exact_match * s["exact_match"] for s in scores]

    def as_list(self) -> List[Callable[..., List[float]]]:
        return [
            self.reward_format_exact,
            self.reward_format_approx,
            self.reward_tool_score,
            self.reward_optional_f1,
            self.reward_positional_f1,
            self.reward_exact_match,
        ]


# -----------------------------------------------------------------------------
# Model setup and training
# -----------------------------------------------------------------------------


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.exception("Unable to clear CUDA cache")
    gc.collect()


def patch_chat_template(tokenizer: Any) -> None:
    old = "'<|im_start|>assistant\\n\'"
    new = f"'<|im_start|>assistant\\n{REASONING_START}\'"

    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        raise ValueError("Tokenizer does not have a chat_template to patch")

    if old not in tokenizer.chat_template:
        LOGGER.warning("Expected assistant chat-template fragment was not found; leaving unchanged.")
        return

    if new in tokenizer.chat_template:
        LOGGER.warning("Chat template already appears to be patched; leaving unchanged.")
        return

    tokenizer.chat_template = tokenizer.chat_template.replace(old, new)
    LOGGER.info("Patched tokenizer chat template to prepend %s", REASONING_START)


def load_model_and_tokenizer(config: TrainConfig):
    from unsloth import FastLanguageModel

    LOGGER.info("Loading model %s", config.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        fast_inference=config.fast_inference,
        enforce_eager=config.enforce_eager,
        max_lora_rank=config.lora_rank,
        gpu_memory_utilization=config.gpu_memory_utilization,
        attention_backend=config.attention_backend,
    )

    patch_chat_template(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    return model, tokenizer


def load_tokenizer_only(config: TrainConfig):
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    except Exception as exc:
        LOGGER.warning("AutoTokenizer failed; falling back to model loader for tokenizer: %s", exc)

    if tokenizer is None:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            fast_inference=config.fast_inference,
            enforce_eager=config.enforce_eager,
            max_lora_rank=config.lora_rank,
            gpu_memory_utilization=config.gpu_memory_utilization,
            attention_backend=config.attention_backend,
        )
        del model
        clear_cuda_cache()

    patch_chat_template(tokenizer)
    return tokenizer


def _len_from_input_ids(input_ids: Any) -> int:
    if input_ids is None:
        return 0
    if hasattr(input_ids, "shape"):
        shape = input_ids.shape
        if len(shape) == 0:
            return 0
        if len(shape) == 1:
            return int(shape[0])
        return int(shape[-1])
    if isinstance(input_ids, (list, tuple)):
        if not input_ids:
            return 0
        first = input_ids[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(input_ids)
    try:
        return len(input_ids)
    except TypeError:
        return 0


def _count_prompt_tokens(tokenizer: Any, prompt: Sequence[Dict[str, Any]]) -> int:
    encoded = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
    )
    if isinstance(encoded, dict):
        return _len_from_input_ids(encoded.get("input_ids"))
    if hasattr(encoded, "input_ids"):
        return _len_from_input_ids(encoded.input_ids)
    return _len_from_input_ids(encoded)


def load_and_prepare_dataset(tokenizer: Any, config: TrainConfig):
    from datasets import Dataset

    records = read_jsonl(config.dataset_path)
    rows = build_training_rows(
        records=records,
        mode_specs=config.mode_specs,
        subtools_path=config.subtools_path,
        candidate_tools=config.candidate_tools,
        candidate_seed=config.candidate_seed,
        include_usage_in_prompt=config.include_usage_in_prompt,
        max_usage_tokens=config.max_usage_tokens,
        usage_tokenizer=tokenizer,
    )
    if config.shuffle_dataset:
        rng = random.Random(config.seed)
        rng.shuffle(rows)
        LOGGER.info("Shuffled %d rows with seed=%d", len(rows), config.seed)
    dataset = Dataset.from_list(rows)

    tokenized = dataset.map(
        lambda x: {"prompt_length": _count_prompt_tokens(tokenizer, x["prompt"])},
        batched=False,
    )

    prompt_lengths = np.array(tokenized["prompt_length"])
    max_prompt_length = int(np.quantile(prompt_lengths, config.prompt_length_quantile))
    keep_indices = np.where(prompt_lengths <= max_prompt_length)[0]
    dataset = dataset.select(keep_indices.tolist())

    LOGGER.info(
        "Loaded %d records, kept %d rows after %.2f quantile filter (max prompt length=%d)",
        len(rows),
        len(dataset),
        config.prompt_length_quantile,
        max_prompt_length,
    )

    return dataset, max_prompt_length


def _log_length_stats(title: str, lengths: Sequence[int]) -> None:
    if not lengths:
        LOGGER.warning("%s stats: no rows", title)
        return

    arr = np.asarray(lengths, dtype=np.int64)
    percentiles = np.percentile(arr, [50, 90, 95, 99])
    LOGGER.info(
        "%s stats | count=%d min=%d max=%d mean=%.2f median=%d p90=%d p95=%d p99=%d",
        title,
        arr.size,
        int(arr.min()),
        int(arr.max()),
        float(arr.mean()),
        int(percentiles[0]),
        int(percentiles[1]),
        int(percentiles[2]),
        int(percentiles[3]),
    )

    if arr.size < 2 or int(arr.min()) == int(arr.max()):
        LOGGER.info("%s histogram | all=%d (n=%d)", title, int(arr.min()), arr.size)
        return

    counts, edges = np.histogram(arr, bins=10)
    for left, right, count in zip(edges[:-1], edges[1:], counts):
        if count == 0:
            continue
        pct = (count / arr.size) * 100.0
        LOGGER.info(
            "%s histogram | %d-%d: %d (%.1f%%)",
            title,
            int(left),
            int(right),
            int(count),
            pct,
        )


def _log_sample_rows(
    title: str,
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_samples: int = 3,
) -> None:
    if not rows:
        LOGGER.info("%s samples | none", title)
        return

    sample_count = min(max_samples, len(rows))
    for idx, row in enumerate(rows[:sample_count], start=1):
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"],
            add_generation_prompt=True,
            tokenize=False,
        )
        LOGGER.info(
            "%s sample %d/%d | custom_id=%s tool=%s query=%s\n%s",
            title,
            idx,
            sample_count,
            row.get("custom_id"),
            row.get("tool_name"),
            row.get("query"),
            prompt_text,
        )


def debug_dataset(config: TrainConfig) -> None:
    configure_environment(config)

    tokenizer = load_tokenizer_only(config)

    from datasets import Dataset

    records = read_jsonl(config.dataset_path)
    rows = build_training_rows(
        records=records,
        mode_specs=config.mode_specs,
        subtools_path=config.subtools_path,
        candidate_tools=config.candidate_tools,
        candidate_seed=config.candidate_seed,
        include_usage_in_prompt=config.include_usage_in_prompt,
        max_usage_tokens=config.max_usage_tokens,
        usage_tokenizer=tokenizer,
    )
    if config.shuffle_dataset:
        rng = random.Random(config.seed)
        rng.shuffle(rows)
        LOGGER.info("Shuffled %d rows with seed=%d", len(rows), config.seed)

    rows_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        rows_by_mode.setdefault(mode, []).append(row)

    for mode in sorted(rows_by_mode.keys()):
        _log_sample_rows(f"mode={mode}", rows_by_mode[mode], tokenizer, max_samples=3)

    dataset = Dataset.from_list(rows)

    tokenized = dataset.map(
        lambda x: {"prompt_length": _count_prompt_tokens(tokenizer, x["prompt"])},
        batched=False,
    )

    prompt_lengths = tokenized["prompt_length"]
    _log_length_stats("all-rows", prompt_lengths)

    max_prompt_length = int(np.quantile(prompt_lengths, config.prompt_length_quantile))
    keep_indices = np.where(np.asarray(prompt_lengths) <= max_prompt_length)[0]
    filtered = tokenized.select(keep_indices.tolist())
    _log_length_stats("filtered", filtered["prompt_length"])

    LOGGER.info(
        "prompt_length filter | quantile=%.2f threshold=%d kept=%d/%d",
        config.prompt_length_quantile,
        max_prompt_length,
        len(filtered),
        len(tokenized),
    )


def resolve_save_steps(config: TrainConfig, dataset_len: int) -> int | None:
    if config.save_steps is None:
        return None
    if dataset_len <= 0:
        raise ValueError("Cannot resolve checkpoint cadence with an empty dataset.")

    global_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
    if global_batch_size <= 0:
        raise ValueError("per_device_train_batch_size * gradient_accumulation_steps must be > 0")

    # steps_per_epoch = dataset_len # For GRPO somehow steps_per_epochs = dataset_len
    steps_per_epoch = math.ceil(dataset_len / global_batch_size)
    resolved_steps = max(1, int(round(steps_per_epoch * config.save_steps)))

    LOGGER.info(
        "Resolved checkpoint cadence: save_steps=%s epoch(s) -> %d optimizer step(s) per save (steps_per_epoch=%d)",
        config.save_steps,
        resolved_steps,
        steps_per_epoch,
    )
    return resolved_steps


def build_training_args(
    config: TrainConfig,
    max_prompt_length: int,
    max_completion_length: int,
    dataset_len: int,
):
    from trl import GRPOConfig

    resolved_save_steps = resolve_save_steps(config, dataset_len)

    kwargs: Dict[str, Any] = {
        "temperature": 1.0,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optim,
        "logging_steps": config.logging_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_generations": config.num_generations,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "num_train_epochs": config.num_train_epochs,
        "report_to": config.report_to,
        "output_dir": str(config.output_dir),
    }

    if resolved_save_steps is None:
        kwargs["save_strategy"] = "no"
    else:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = resolved_save_steps

    if config.max_steps is not None:
        kwargs["max_steps"] = config.max_steps

    return GRPOConfig(**kwargs)


def train(config: TrainConfig) -> None:
    configure_environment(config)
    seed_everything(config.seed)
    maybe_init_wandb(config)

    import torch

    clear_cuda_cache()
    model, tokenizer = load_model_and_tokenizer(config)
    dataset, max_prompt_length = load_and_prepare_dataset(tokenizer, config)

    max_completion_length = config.max_seq_length - (max_prompt_length + 1)
    if max_completion_length <= 0:
        raise ValueError(
            f"Computed non-positive max_completion_length={max_completion_length}. "
            "Increase max_seq_length or reduce prompt_length_quantile."
        )

    reward_functions = KaliBenchRewardFunctions(
        weights=config.reward_weights,
        tokenizer=tokenizer,
        debug_print_every=config.debug_reward_print_every,
    )

    training_args = build_training_args(
        config=config,
        max_prompt_length=max_prompt_length + 1,
        max_completion_length=max_completion_length,
        dataset_len=len(dataset),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.adapter_output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Training rows=%d", len(dataset))
    LOGGER.info("Modes=%s", _format_mode_specs(config.mode_specs))
    LOGGER.info("max_prompt_length=%d max_completion_length=%d", max_prompt_length + 1, max_completion_length)
    LOGGER.info(
        "Reward weights | format_exact=%.2f format_approx=%.2f tool=%.2f optional=%.2f positional=%.2f exact=%.2f",
        config.reward_weights.format_exact,
        config.reward_weights.format_approx,
        config.reward_weights.tool_score,
        config.reward_weights.optional_f1,
        config.reward_weights.positional_f1,
        config.reward_weights.exact_match,
    )
    LOGGER.info(
        "Sample prompt:\n%s",
        tokenizer.apply_chat_template(dataset[0]["prompt"], add_generation_prompt=True, tokenize=False),
    )

    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions.as_list(),
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_lora(str(config.adapter_output_dir))
    LOGGER.info("Saved LoRA adapter to %s", config.adapter_output_dir)

    model.save_pretrained_merged(
        save_directory=str(config.output_dir / "merged"),
        tokenizer=tokenizer,
        save_method="merged_16bit",
    )
    LOGGER.info("Saved merged model to %s/merged", config.output_dir)
    LOGGER.info("Training complete.")

    if torch.cuda.is_available():
        LOGGER.info("Peak reserved CUDA memory: %.2f GB", torch.cuda.max_memory_reserved() / 1024**3)


def main() -> None:
    config = parse_args()
    LOGGER.info("Dataset path: %s", config.dataset_path)
    LOGGER.info("Subtools path: %s", config.subtools_path)
    LOGGER.info("Requested modes: %s", _format_mode_specs(config.mode_specs))
    if config.debug_dataset:
        debug_dataset(config)
        return
    train(config)


if __name__ == "__main__":
    main()
