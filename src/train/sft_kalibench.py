#!/usr/bin/env python3
"""SFT training entrypoint for KaliBench tool-calling tasks.

This script mirrors the data construction in notebooks/grpo_kalibench.py so
comparisons between GRPO and SFT are fair:
- Same train JSONL fields and prompt templates
- Same unrestricted / restricted / hinted modes
- Same candidate tool sampling behavior
- Same core model, LoRA, and optimization defaults where applicable

Training is implemented with Unsloth + TRL SFTTrainer and supports
train-on-responses-only masking, following the Unsloth Qwen SFT tutorial.
SFT targets are output-only and do not include think tags.
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

LOGGER = logging.getLogger("sft_kalibench")


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
<output>
[command]
</output>"""


KALIBENCH_USER_TEMPLATE = """USER QUERY: \"{query}\"

Generate the single most accurate shell command for the query.
Your response must follow the required structure:

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
7. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

OUTPUT FORMAT:
<output>
[command]
</output>"""


KALIBENCH_USER_RESTRICTED_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: \"{query}\"

Generate the single most accurate shell command for the query using ONLY the allowed tools listed above.
Your response must follow the required structure:

<output>
[command]
</output>"""


KALIBENCH_USER_HINTED_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: \"{query}\"

Generate the single most accurate shell command for the query using ONLY the allowed tools listed above.
Use the provided usage hints to select the correct tool and arguments.
Your response must follow the required structure:

<output>
[command]
</output>"""


SOLUTION_START = "<output>"
SOLUTION_END = "</output>"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
	modes: List[str]
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
	enable_thinking: bool

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


def parse_args() -> TrainConfig:
	parser = argparse.ArgumentParser(description="Train an SFT model for KaliBench tool calling")

	parser.add_argument(
		"--mode",
		nargs="+",
		choices=["unrestricted", "restricted", "hinted"],
		default=["hinted", "restricted", "unrestricted"],
		help="One or more training modes. Multiple modes are concatenated into one train set.",
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
	parser.add_argument("--output-dir", default="outputs/models/RedSage-Qwen3-8B-Ins-kalibench_sft_fixed")
	parser.add_argument("--adapter-output-dir", default="outputs/adapters/RedSage-Qwen3-8B-Ins-kalibench_sft_fixed")
	parser.add_argument("--cache-root", default=os.environ.get("TRAIN_CACHE_ROOT"))
	parser.add_argument(
		"--candidate-tools",
		type=int,
		default=20,
		help="Number of candidate tools for restricted mode. Hinted mode always uses 1.",
	)
	parser.add_argument("--candidate-seed", type=int, default=42)
	parser.add_argument("--include-usage-in-prompt", type=str2bool, default=True)
	parser.add_argument("--max-usage-tokens", type=int, default=4096*2)

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
		help="Checkpoint cadence in epochs. Example: 1.0 = every epoch, 0.5 = every half epoch. Use 'none' to disable.",
	)
	parser.add_argument("--max-steps", type=int, default=None)
	parser.add_argument("--prompt-length-quantile", type=float, default=0.99)

	parser.add_argument("--attention-backend", default="TRITON_ATTN")
	parser.add_argument("--load-in-4bit", action="store_true")
	parser.add_argument("--disable-fast-inference", action="store_true")
	parser.add_argument("--disable-enforce-eager", action="store_true")

	parser.add_argument("--report-to", default="wandb")
	parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "kalibench_sft"))
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

	dataset_path = Path(args.dataset_path)
	if not dataset_path.is_absolute():
		dataset_path = (repo_root / dataset_path).resolve()

	subtools_path = Path(args.subtools_path)
	if not subtools_path.is_absolute():
		subtools_path = (repo_root / subtools_path).resolve()

	output_dir = Path(args.output_dir)
	if not output_dir.is_absolute():
		output_dir = (repo_root / output_dir).resolve()

	adapter_output_dir = Path(args.adapter_output_dir)
	if not adapter_output_dir.is_absolute():
		adapter_output_dir = (repo_root / adapter_output_dir).resolve()

	modes = [str(m).strip().lower() for m in args.mode if str(m).strip()]
	if not modes:
		raise ValueError("At least one mode must be provided.")
	modes = list(dict.fromkeys(modes))

	return TrainConfig(
		modes=modes,
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


def clear_cuda_cache() -> None:
	try:
		import torch

		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	except Exception:
		LOGGER.exception("Unable to clear CUDA cache")
	gc.collect()


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


def build_supervised_response(command: str) -> str:
	return f"{SOLUTION_START}\n{command}\n{SOLUTION_END}"


def build_training_rows(
	records: Sequence[Dict[str, Any]],
	modes: Sequence[str],
	subtools_path: Path,
	candidate_tools: int,
	candidate_seed: int,
	include_usage_in_prompt: bool,
	max_usage_tokens: int,
	usage_tokenizer: Any | None = None,
) -> List[Dict[str, Any]]:
	if not modes:
		raise ValueError("At least one mode must be provided.")

	valid_modes = {"unrestricted", "restricted", "hinted"}
	unknown_modes = [m for m in modes if m not in valid_modes]
	if unknown_modes:
		raise ValueError(f"Unsupported mode(s): {unknown_modes}")

	tool_usage_map: Dict[str, str] = {}
	if subtools_path.exists():
		subtool_records = read_jsonl(subtools_path)
		tool_usage_map = load_tool_usages_jsonl(subtool_records)
	elif "hinted" in modes:
		LOGGER.warning("Subtools file not found at %s; hinted mode will run without usage snippets.", subtools_path)

	tool_pool = sorted(
		{
			str(r.get("tool_name") or r.get("tool") or "").strip()
			for r in records
			if str(r.get("tool_name") or r.get("tool") or "").strip()
		}
	)
	tool_pool_by_lower = {tool.lower(): tool for tool in tool_pool}

	if any(mode in {"restricted", "hinted"} for mode in modes) and not tool_pool:
		raise ValueError("No tools found in records; cannot build restricted/hinted prompts.")

	rows: List[Dict[str, Any]] = []
	skipped = 0
	per_mode_counts: Dict[str, int] = {}
	print_per_mode = {mode: False for mode in modes}

	for mode_idx, mode in enumerate(modes):
		rng = random.Random(candidate_seed + mode_idx)
		mode_count = 0

		for idx, row in enumerate(records, start=1):
			custom_id_base = str(row.get("custom_id") or f"row-{idx}").strip()
			custom_id = f"{custom_id_base}::{mode}"
			query = extract_query(row)
			tool_name = str(row.get("tool_name") or row.get("tool") or "").strip()
			gt_cmd = str(row.get("ground_truth_command") or "").strip()

			if not query or not tool_name or not gt_cmd:
				skipped += 1
				continue

			if mode == "unrestricted":
				system_prompt = KALIBENCH_PROMPT
				user_prompt = KALIBENCH_USER_TEMPLATE.format(query=query)
			else:
				# Overwrite candidate_tools to 1 for hinted mode since we always include the target tool with usage hints
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

				# For hinted mode, we include usage snippets in the prompt to help the model identify the correct tool and its arguments.
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

			rows.append(
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
					"response": build_supervised_response(command=gt_cmd),
				}
			)
			if print_per_mode[mode] == False:
				LOGGER.info("Example prompt for mode '%s':\n%s", mode, user_prompt)
				print_per_mode[mode] = True
			mode_count += 1

		per_mode_counts[mode] = mode_count

	if skipped:
		LOGGER.warning("Skipped %d rows with missing query/tool_name/ground_truth_command", skipped)

	if not rows:
		raise ValueError("No valid rows were built for training.")

	LOGGER.info("Prepared %d training rows across modes=%s", len(rows), per_mode_counts)
	return rows


# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------


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
	rows = build_training_rows(
		records=records,
		modes=config.modes,
		subtools_path=config.subtools_path,
		candidate_tools=config.candidate_tools,
		candidate_seed=config.candidate_seed,
		include_usage_in_prompt=config.include_usage_in_prompt,
		max_usage_tokens=config.max_usage_tokens,
		usage_tokenizer=tokenizer,
	)
	dataset = Dataset.from_list(rows)

	tokenized_prompts = dataset.map(
		lambda x: {
			"prompt_tokens": tokenizer.apply_chat_template(
				x["prompt"],
				add_generation_prompt=True,
				tokenize=True,
				enable_thinking=config.enable_thinking if config.enable_thinking is not None else None,
			)
		},
		batched=False,
	)
	tokenized_prompts = tokenized_prompts.map(lambda x: {"prompt_length": len(x["prompt_tokens"])})

	prompt_lengths = np.array(tokenized_prompts["prompt_length"])
	max_prompt_length = int(np.quantile(prompt_lengths, config.prompt_length_quantile))
	keep_indices = np.where(prompt_lengths <= max_prompt_length)[0]
	dataset = dataset.select(keep_indices.tolist())

	def to_text_row(example: Dict[str, Any]) -> Dict[str, Any]:
		messages = list(example["prompt"]) + [{"role": "assistant", "content": example["response"]}]
		token_ids = tokenizer.apply_chat_template(
			messages,
			add_generation_prompt=False,
			tokenize=True,
			enable_thinking=config.enable_thinking if config.enable_thinking is not None else None,
		)
		text = tokenizer.apply_chat_template(
			messages,
			add_generation_prompt=False,
			tokenize=False,
			enable_thinking=config.enable_thinking if config.enable_thinking is not None else None,
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
			"No rows remain after max_seq_length filter. "
			"Increase --max-seq-length or lower --prompt-length-quantile."
		)

	LOGGER.info(
		"Loaded %d records, kept %d rows after %.2f prompt quantile filter (max prompt length=%d)",
		len(rows),
		before_seq_filter,
		config.prompt_length_quantile,
		max_prompt_length,
	)
	if dropped_for_length:
		LOGGER.info("Dropped %d rows with full sequence length > %d", dropped_for_length, config.max_seq_length)

	LOGGER.info("Final SFT rows=%d", len(dataset))
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
		raise ValueError("per_device_train_batch_size * gradient_accumulation_steps must be > 0")

	steps_per_epoch = math.ceil(dataset_len / global_batch_size)
	resolved_steps = max(1, int(round(steps_per_epoch * config.save_steps)))

	LOGGER.info(
		"Resolved checkpoint cadence: save_steps=%s epoch(s) -> %d optimizer step(s) per save (steps_per_epoch=%d)",
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

	LOGGER.info("Training rows=%d", len(dataset))
	LOGGER.info("Modes=%s", ",".join(config.modes))
	LOGGER.info("max_prompt_length(after quantile)=%d", max_prompt_length)
	LOGGER.info(
		"Sample prompt:\n%s",
		tokenizer.apply_chat_template(dataset[0]["prompt"], add_generation_prompt=True, tokenize=False, enable_thinking=config.enable_thinking if config.enable_thinking is not None else None),
	)
	LOGGER.info("Sample supervised response:\n%s", dataset[0]["response"])

	trainer = build_sft_trainer(model=model, tokenizer=tokenizer, dataset=dataset, training_args=training_args)
	trainer = maybe_apply_response_only_masking(trainer, config)

	trainer.train()

	model.save_lora(str(config.adapter_output_dir))
	tokenizer.save_pretrained(str(config.adapter_output_dir))
	LOGGER.info("Saved LoRA adapter and tokenizer to %s", config.adapter_output_dir)

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
	LOGGER.info("Requested modes: %s", ",".join(config.modes))
	train(config)


if __name__ == "__main__":
	main()
