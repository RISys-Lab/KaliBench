#!/usr/bin/env python3
"""General chat SFT training entrypoint.

Expected JSONL format:

Each line must be a JSON object with a `messages` field:

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

The script:
- Loads chat-style JSONL directly
- Applies the model tokenizer chat template
- Trains with Unsloth + TRL SFTTrainer
- Supports LoRA adapters
- Supports response-only masking through Unsloth
- Saves both LoRA adapter and merged 16-bit model
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

LOGGER = logging.getLogger("sft_general")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    model_name: str
    dataset_path: Path
    output_dir: Path
    adapter_output_dir: Path
    cache_root: Path | None

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

    train_on_responses_only: bool
    instruction_part: str
    response_part: str
    enable_thinking: bool | None


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

    raise argparse.ArgumentTypeError("Boolean value expected: true or false.")


def parse_optional_positive_float(value: str) -> float | None:
    s = str(value).strip().lower()
    if s in {"none", "no", "off", "false"}:
        return None

    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0, or use 'none' to disable saving.")

    return parsed


def resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a general chat SFT model from messages JSONL.")

    parser.add_argument("--model-name", default="RISys-Lab/RedSage-Qwen3-8B-Ins")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="JSONL file. Each line must contain a `messages` list.",
    )

    parser.add_argument("--output-dir", default="outputs/models/sft_general")
    parser.add_argument("--adapter-output-dir", default="outputs/adapters/sft_general")
    parser.add_argument("--cache-root", default=os.environ.get("TRAIN_CACHE_ROOT"))

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-seq-length", type=int, default=12288)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)

    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--optim", default="adamw_8bit")

    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--logging-steps", type=int, default=1)

    parser.add_argument(
        "--save-steps",
        type=parse_optional_positive_float,
        default=1.0,
        help=(
            "Checkpoint cadence in epochs. "
            "Example: 1.0 = every epoch, 0.5 = every half epoch. "
            "Use 'none' to disable checkpointing."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--prompt-length-quantile",
        type=float,
        default=0.99,
        help="Drop the longest prompts above this quantile before full sequence filtering.",
    )

    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--disable-fast-inference", action="store_true")
    parser.add_argument("--disable-enforce-eager", action="store_true")

    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "sft_general"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))

    parser.add_argument("--train-on-responses-only", type=str2bool, default=True)
    parser.add_argument("--instruction-part", default="<|im_start|>user\n")
    parser.add_argument("--response-part", default="<|im_start|>assistant\n")
    parser.add_argument("--enable-thinking", type=str2bool, default=None)

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]

    return TrainConfig(
        model_name=args.model_name,
        dataset_path=resolve_path(args.dataset_path, repo_root),
        output_dir=resolve_path(args.output_dir, repo_root),
        adapter_output_dir=resolve_path(args.adapter_output_dir, repo_root),
        cache_root=Path(args.cache_root).expanduser() if args.cache_root else None,

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

        train_on_responses_only=args.train_on_responses_only,
        instruction_part=args.instruction_part,
        response_part=args.response_part,
        enable_thinking=args.enable_thinking,
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
        LOGGER.exception("Failed to fully seed torch.")


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
    if str(config.report_to).lower() != "wandb":
        return

    try:
        import wandb

        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"])
        else:
            LOGGER.info("WANDB_API_KEY not set. Assuming existing auth or offline mode.")
    except Exception:
        LOGGER.exception("Failed to initialize Weights & Biases.")
        raise


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.exception("Unable to clear CUDA cache.")

    gc.collect()


# -----------------------------------------------------------------------------
# JSONL loading and chat formatting
# -----------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

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
                raise ValueError(f"Expected JSON object at {path}:{lineno}, got {type(row)}")

            rows.append(row)

    return rows


def normalize_messages(row: Dict[str, Any], row_index: int) -> List[Dict[str, str]] | None:
    messages = row.get("messages")

    if not isinstance(messages, list):
        LOGGER.warning("Skipping row %d because `messages` is missing or not a list.", row_index)
        return None

    normalized: List[Dict[str, str]] = []

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            LOGGER.warning(
                "Skipping row %d because message %d is not an object.",
                row_index,
                message_index,
            )
            return None

        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str) or not role.strip():
            LOGGER.warning(
                "Skipping row %d because message %d has invalid role.",
                row_index,
                message_index,
            )
            return None

        if content is None:
            content = ""

        if not isinstance(content, str):
            content = str(content)

        normalized.append(
            {
                "role": role.strip(),
                "content": content,
            }
        )

    if not normalized:
        LOGGER.warning("Skipping row %d because `messages` is empty.", row_index)
        return None

    if not any(message["role"] == "assistant" for message in normalized):
        LOGGER.warning("Skipping row %d because it contains no assistant message.", row_index)
        return None

    return normalized


def get_prompt_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return messages before the final assistant message.

    This is used only for prompt-length measurement and logging.
    The final training text still includes the full conversation.
    """
    final_assistant_index = None

    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            final_assistant_index = idx
            break

    if final_assistant_index is None:
        return list(messages)

    return list(messages[:final_assistant_index])


def apply_chat_template_safe(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    *,
    add_generation_prompt: bool,
    tokenize: bool,
    enable_thinking: bool | None,
) -> Any:
    kwargs: Dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": tokenize,
    }

    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_training_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skipped = 0

    for idx, row in enumerate(records, start=1):
        messages = normalize_messages(row, row_index=idx)

        if messages is None:
            skipped += 1
            continue

        custom_id = str(row.get("custom_id") or row.get("id") or f"row-{idx}")

        rows.append(
            {
                "custom_id": custom_id,
                "messages": messages,
                "prompt": get_prompt_messages(messages),
            }
        )

    if skipped:
        LOGGER.warning("Skipped %d invalid rows.", skipped)

    if not rows:
        raise ValueError("No valid rows were found. Expected JSONL rows with a `messages` list.")

    LOGGER.info("Prepared %d training rows.", len(rows))
    return rows


# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------


def load_model_and_tokenizer(config: TrainConfig):
    from unsloth import FastLanguageModel

    LOGGER.info("Loading model %s", config.model_name)

    from_pretrained_kwargs: Dict[str, Any] = {
        "model_name": config.model_name,
        "max_seq_length": config.max_seq_length,
        "load_in_4bit": config.load_in_4bit,
        "fast_inference": config.fast_inference,
        "enforce_eager": config.enforce_eager,
        "max_lora_rank": config.lora_rank,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "attention_backend": config.attention_backend,
    }

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)
    except TypeError:
        LOGGER.warning("Retrying model load without attention_backend.")
        from_pretrained_kwargs.pop("attention_backend", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

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


# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------


def load_and_prepare_dataset(tokenizer: Any, config: TrainConfig):
    from datasets import Dataset

    records = read_jsonl(config.dataset_path)
    rows = build_training_rows(records)

    dataset = Dataset.from_list(rows)

    def add_prompt_tokens(example: Dict[str, Any]) -> Dict[str, Any]:
        token_ids = apply_chat_template_safe(
            tokenizer,
            example["prompt"],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=config.enable_thinking,
        )
        return {
            "prompt_tokens": token_ids,
            "prompt_length": len(token_ids),
        }

    tokenized_prompts = dataset.map(add_prompt_tokens, batched=False)

    prompt_lengths = np.array(tokenized_prompts["prompt_length"])

    if len(prompt_lengths) == 0:
        raise ValueError("No prompt lengths were computed.")

    if not 0 < config.prompt_length_quantile <= 1:
        raise ValueError("--prompt-length-quantile must be in the range (0, 1].")

    max_prompt_length = int(np.quantile(prompt_lengths, config.prompt_length_quantile))
    keep_indices = np.where(prompt_lengths <= max_prompt_length)[0]
    dataset = dataset.select(keep_indices.tolist())

    def to_text_row(example: Dict[str, Any]) -> Dict[str, Any]:
        token_ids = apply_chat_template_safe(
            tokenizer,
            example["messages"],
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=config.enable_thinking,
        )

        text = apply_chat_template_safe(
            tokenizer,
            example["messages"],
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=config.enable_thinking,
        )

        return {
            "text": text,
            "text_length": len(token_ids),
        }

    dataset = dataset.map(to_text_row, batched=False)

    before_seq_filter = len(dataset)
    dataset = dataset.filter(lambda x: x["text_length"] <= config.max_seq_length)
    dropped_for_length = before_seq_filter - len(dataset)

    if len(dataset) == 0:
        raise ValueError(
            "No rows remain after max_seq_length filtering. "
            "Increase --max-seq-length or lower --prompt-length-quantile."
        )

    LOGGER.info(
        "Loaded %d valid rows, kept %d rows after %.4f prompt quantile filter. "
        "Max prompt length=%d.",
        len(rows),
        before_seq_filter,
        config.prompt_length_quantile,
        max_prompt_length,
    )

    if dropped_for_length:
        LOGGER.info(
            "Dropped %d rows with full sequence length > %d.",
            dropped_for_length,
            config.max_seq_length,
        )

    LOGGER.info("Final SFT rows=%d.", len(dataset))

    return dataset, max_prompt_length


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def resolve_save_steps(config: TrainConfig, dataset_len: int) -> int | None:
    if config.save_steps is None:
        return None

    if dataset_len <= 0:
        raise ValueError("Cannot resolve checkpoint cadence with an empty dataset.")

    global_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps

    if global_batch_size <= 0:
        raise ValueError("per_device_train_batch_size * gradient_accumulation_steps must be > 0.")

    steps_per_epoch = math.ceil(dataset_len / global_batch_size)
    resolved_steps = max(1, int(round(steps_per_epoch * config.save_steps)))

    LOGGER.info(
        "Resolved checkpoint cadence: save_steps=%s epoch(s) -> %d optimizer step(s) per save. "
        "steps_per_epoch=%d.",
        config.save_steps,
        resolved_steps,
        steps_per_epoch,
    )

    return resolved_steps


def build_training_args(config: TrainConfig, dataset_len: int):
    from trl import SFTConfig

    kwargs: Dict[str, Any] = {
        "dataset_text_field": "text",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optim,
        "logging_steps": config.logging_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs,
        "report_to": config.report_to,
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "packing": False,
    }

    if config.max_steps is not None:
        kwargs["max_steps"] = config.max_steps

    resolved_save_steps = resolve_save_steps(config, dataset_len)

    if resolved_save_steps is None:
        kwargs["save_strategy"] = "no"
    else:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = resolved_save_steps

    candidates = [
        {"max_seq_length": config.max_seq_length},
        {"max_length": config.max_seq_length},
        {},
    ]

    last_error: Exception | None = None

    for extra in candidates:
        try:
            return SFTConfig(**{**kwargs, **extra})
        except TypeError as exc:
            last_error = exc
            continue

    assert last_error is not None
    raise last_error


def build_sft_trainer(model: Any, tokenizer: Any, dataset: Any, training_args: Any):
    from trl import SFTTrainer

    try:
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=None,
            args=training_args,
        )
    except TypeError:
        return SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            eval_dataset=None,
            args=training_args,
        )


def maybe_apply_response_only_masking(trainer: Any, config: TrainConfig) -> Any:
    if not config.train_on_responses_only:
        return trainer

    from unsloth.chat_templates import train_on_responses_only

    return train_on_responses_only(
        trainer,
        instruction_part=config.instruction_part,
        response_part=config.response_part,
    )


def train(config: TrainConfig) -> None:
    configure_environment(config)
    seed_everything(config.seed)
    maybe_init_wandb(config)

    import torch

    clear_cuda_cache()

    model, tokenizer = load_model_and_tokenizer(config)
    dataset, max_prompt_length = load_and_prepare_dataset(tokenizer, config)

    training_args = build_training_args(config, dataset_len=len(dataset))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.adapter_output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Training rows=%d.", len(dataset))
    LOGGER.info("max_prompt_length after quantile=%d.", max_prompt_length)

    LOGGER.info(
        "Sample prompt:\n%s",
        apply_chat_template_safe(
            tokenizer,
            dataset[0]["prompt"],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=config.enable_thinking,
        ),
    )

    LOGGER.info("Sample full training text:\n%s", dataset[0]["text"])

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
    )

    trainer = maybe_apply_response_only_masking(trainer, config)

    trainer.train()

    model.save_lora(str(config.adapter_output_dir))
    tokenizer.save_pretrained(str(config.adapter_output_dir))

    LOGGER.info("Saved LoRA adapter and tokenizer to %s.", config.adapter_output_dir)

    model.save_pretrained_merged(
        save_directory=str(config.output_dir / "merged"),
        tokenizer=tokenizer,
        save_method="merged_16bit",
    )

    LOGGER.info("Saved merged model to %s.", config.output_dir / "merged")
    LOGGER.info("Training complete.")

    if torch.cuda.is_available():
        LOGGER.info(
            "Peak reserved CUDA memory: %.2f GB.",
            torch.cuda.max_memory_reserved() / 1024**3,
        )


def main() -> None:
    config = parse_args()

    LOGGER.info("Dataset path: %s", config.dataset_path)
    LOGGER.info("Output dir: %s", config.output_dir)
    LOGGER.info("Adapter output dir: %s", config.adapter_output_dir)

    train(config)


if __name__ == "__main__":
    main()