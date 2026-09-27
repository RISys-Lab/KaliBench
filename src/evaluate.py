#!/usr/bin/env python3
"""End-to-end KaliBench inference and scoring pipeline.

This script unifies prompt construction and vLLM inference so evaluation can run
in one step instead of generating request JSONL first and running a separate
batch inference command.
"""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List


def _parse_wandb_tags(raw_tags: str) -> List[str]:
	if not raw_tags:
		return []
	return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def _init_wandb_if_enabled(args: argparse.Namespace, run_config: Dict[str, Any]) -> Any:
	if not args.wandb:
		print("[INFO] Weights & Biases logging is disabled.")
		return None
	try:
		wandb = importlib.import_module("wandb")
	except ImportError as exc:
		raise RuntimeError("--wandb was provided but wandb is not installed. Install with: pip install wandb") from exc

	run = wandb.init(
		project=args.wandb_project,
		entity=args.wandb_entity or None,
		name=args.wandb_run_name or None,
		group=args.wandb_group or None,
		job_type=args.wandb_job_type,
		tags=_parse_wandb_tags(args.wandb_tags),
		config=run_config,
	)
	print(f"[WANDB] initialized run: {run.id if run else 'unknown'}")
	return run


def _log_wandb_summary(run: Any, summary: Dict[str, Any]) -> None:
	if run is None:
		return
	run.log({"summary": summary})
	for key, value in summary.items():
		if isinstance(value, (int, float, bool, str)):
			run.summary[key] = value
	print("[WANDB] logged evaluation summary")


def _count_input_tokens(tokenizer: Any, messages: List[Dict[str, str]], inference_api: str) -> int:
	apply_chat_template_kwargs: Dict[str, Any] = dict(
		tokenize=True,
		add_generation_prompt=True,
	)
	if inference_api == "non-thinking":
		apply_chat_template_kwargs["enable_thinking"] = False
	encoded = tokenizer.apply_chat_template(messages, **apply_chat_template_kwargs)
	return len(encoded) if encoded is not None else 0


def _count_output_tokens(tokenizer: Any, text: str) -> int:
	return len(tokenizer.encode(text, add_special_tokens=False))


def _attach_raw_token_sizes(
	rows: List[Dict[str, Any]],
	prompt_rows: List[Any],
	tokenizer: Any,
	inference_api: str,
) -> None:
	for row, prompt_row in zip(rows, prompt_rows):
		row["input_tokens"] = _count_input_tokens(tokenizer, prompt_row.messages, inference_api)
		row["prediction_tokens"] = _count_output_tokens(tokenizer, str(row.get("model_reply") or ""))

from io_utils import read_jsonl, write_jsonl
from prediction_checkpoint import (
	append_jsonl,
	build_prediction_rows,
	extract_compat_prediction,
	read_jsonl_prefix,
)
from prompt_builder import build_prompts_and_labels, load_tool_usages_jsonl, str2bool
from scoring_pipeline import run_dim_score, run_model_score, sanitize_name
from vllm_inference import run_vllm


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="End-to-end vLLM inference + scoring for KaliBench.")
	parser.add_argument("--mode", choices=["unrestricted", "restricted", "hinted"], default="hinted")
	parser.add_argument("--input", required=True, help="Input JSONL with query/tool/ground-truth fields.")
	parser.add_argument("--subtools", default="KaliBench_data/Kali_Tool_Subtools_UsageCode_Final.jsonl")
	parser.add_argument("--model", required=True, help="Model path or HF model ID for vLLM.")
	parser.add_argument("--output-dir", default="outputs/evaluate")
	parser.add_argument("--temperature", type=float, default=0.2)
	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument(
		"--repetition-penalty",
		type=float,
		default=1.0,
		help="vLLM repetition_penalty sampling value; 1.0 disables the penalty.",
	)
	parser.add_argument("--max-tokens", type=int, default=4096) # Set max-tokens to 1/2 of max-model-len if specified, to allow for longer tool usage descriptions when using larger generation budgets.
	parser.add_argument(
		"--batch-size",
		type=int,
		default=1000, # Default to a large batch size to maximize throughput; users can override with smaller batches to reduce memory usage or checkpoint more frequently.
		help="Prompts per inference batch. Predictions are checkpointed after each batch.",
	)
	parser.add_argument(
		"--resume",
		type=str2bool,
		default=True,
		help="Resume from existing prediction files in output-dir and append new batches.",
	)
	parser.add_argument(
		"--candidate-tools",
		type=int,
		default=20,
		help="Number of candidate tools to include in restricted/hinted prompts.",
	)
	parser.add_argument(
		"--candidate-seed",
		type=int,
		default=42,
		help="Seed used for randomized candidate tool sampling in restricted prompts.",
	)
	parser.add_argument("--max-usage-tokens", type=int, default=4096*2) # Default to 2x max-tokens or 1/2 of max-model-len, to allow for longer tool usage descriptions when using smaller generation budgets.
	parser.add_argument("--include-usage-in-prompt", type=str2bool, default=False)
	parser.add_argument(
		"--output-policy",
		choices=["strict", "final-line", "thinking"],
		default="strict",
		help="strict forces only <output>...</output>; final-line allows earlier text but requires the response to end with the output block; thinking requires <think>...</think> followed by <output>...</output>.",
	)
	parser.add_argument("--encoding", default="utf-8")
	parser.add_argument("--tensor-parallel-size", type=int, default=1)
	parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
	parser.add_argument(
		"--language-model-only",
		type=str2bool,
		default=False,
		help="If true, skip vision encoder loading for text-only workloads.",
	)
	parser.add_argument(
		"--enforce-eager",
		type=str2bool,
		default=False,
		help="Disable torch.compile/CUDAGraphs for more stable startup.",
	)
	parser.add_argument(
		"--max-model-len",
		type=int,
		default=None,
		help="Override model context length to reduce KV/cache pressure.",
	)
	parser.add_argument(
		"--max-num-batched-tokens",
		type=int,
		default=None,
		help="Cap prefill batch token budget to reduce profiling memory.",
	)
	parser.add_argument(
		"--limit-mm-per-prompt-image",
		type=int,
		default=None,
		help="Max images per prompt; set 0 for text-only workloads.",
	)
	parser.add_argument(
		"--limit-mm-per-prompt-video",
		type=int,
		default=None,
		help="Max videos per prompt; set 0 for text-only workloads.",
	)
	parser.add_argument(
		"--gdn-prefill-backend",
		choices=["auto", "flashinfer", "triton"],
		default="auto",
		help="Backend for Qwen GDN prefill kernels. Use triton to avoid FlashInfer JIT issues.",
	)
	parser.add_argument("--dtype", default="auto")
	parser.add_argument("--trust-remote-code", type=str2bool, default=False)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument(
		"--inference-api",
		choices=["chat", "generate-template", "non-thinking"],
		default="chat",
		help="chat uses llm.chat(...), generate-template uses tokenizer.apply_chat_template + llm.generate(...), non-thinking uses tokenizer.apply_chat_template(..., enable_thinking=False) + llm.generate(...).",
	)
	parser.add_argument("--skip-scoring", action="store_true", help="Skip model_score and dim_score steps.")
	parser.add_argument("--run-dim-score", action="store_true", help="Run dim_score.py after model_score.py.")
	parser.add_argument("--dim-use-hf", action="store_true")
	parser.add_argument("--dim-hf-dataset", default="anonymous62567/kali-tools")
	parser.add_argument("--dim-labels-file", default=None)
	parser.add_argument("--dim-no-tqdm", action="store_true")
	parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging for config and summary.")
	parser.add_argument("--wandb-project", default="KaliBench-eval", help="wandb project name when --wandb is enabled.")
	parser.add_argument("--wandb-entity", default=None, help="wandb entity/team (optional).")
	parser.add_argument("--wandb-run-name", default=None, help="wandb run name (optional).")
	parser.add_argument("--wandb-group", default=None, help="wandb group (optional).")
	parser.add_argument("--wandb-job-type", default="evaluation", help="wandb job type.")
	parser.add_argument(
		"--wandb-tags",
		default="",
		help="Comma-separated wandb tags, for example: hinted,qwen,eval.",
	)
	parser.add_argument(
		"--output-raw",
		action="store_true",
		help="Include raw input messages in raw_predictions.jsonl when enabled.",
	)
	parser.add_argument("--verbose", action="store_true")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()

	repo_root = Path(__file__).resolve().parents[1]
	input_path = Path(args.input)
	if not input_path.is_absolute():
		input_path = (repo_root / input_path).resolve()

	subtools_path = Path(args.subtools)
	if not subtools_path.is_absolute():
		subtools_path = (repo_root / subtools_path).resolve()

	output_dir = Path(args.output_dir)
	if not output_dir.is_absolute():
		output_dir = (repo_root / output_dir).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	records = read_jsonl(input_path, encoding=args.encoding)
	if not records:
		raise ValueError(f"No valid records found in {input_path}")

	if subtools_path.exists():
		subtool_records = read_jsonl(subtools_path, encoding=args.encoding)
		usage_map = load_tool_usages_jsonl(subtool_records)
	else:
		usage_map = {}

	effective_candidate_tools = 1 if args.mode == "hinted" else args.candidate_tools
	if args.mode == "hinted":
		effective_candidate_tools = 1
		include_usage_in_prompt = True
		print("[INFO] For 'hinted' mode, candidate_tools is set to 1 and usage information will be included in prompts regardless of other settings.")
	else:
		include_usage_in_prompt = args.include_usage_in_prompt
		print(f"[INFO] For '{args.mode}' mode, candidate_tools is set to {effective_candidate_tools} and include_usage_in_prompt is {include_usage_in_prompt}.")

	usage_tokenizer = None
	if args.max_usage_tokens > 0:
		from transformers import AutoTokenizer

		usage_tokenizer = AutoTokenizer.from_pretrained(
			args.model,
			trust_remote_code=args.trust_remote_code,
		)
	raw_tokenizer = usage_tokenizer
	if args.output_raw and raw_tokenizer is None:
		from transformers import AutoTokenizer

		raw_tokenizer = AutoTokenizer.from_pretrained(
			args.model,
			trust_remote_code=args.trust_remote_code,
		)
	prompts, labels = build_prompts_and_labels(
		records=records,
		mode=args.mode,
		candidate_tools=effective_candidate_tools,
		tool_usage_map=usage_map,
		include_usage_in_prompt=include_usage_in_prompt,
		max_usage_tokens=args.max_usage_tokens,
		candidate_seed=args.candidate_seed,
		output_policy=args.output_policy,
		tokenizer=usage_tokenizer,
	)

	if args.batch_size <= 0:
		raise ValueError("--batch-size must be > 0")

	labels_file = output_dir / "labels.jsonl"
	raw_pred_file = output_dir / "raw_predictions.jsonl"
	vllm_compat_file = output_dir / "vllm_compat_predictions.jsonl"
	write_jsonl(labels_file, labels)

	messages_list = [row.messages for row in prompts]
	resume_index = 0
	predictions: List[Any] = [None] * len(prompts)
	missing_indices: List[int] = list(range(len(prompts)))

	# Override max_tokens to 1/2 of max_model_len if specified and max_tokens and max_usage_tokens are at their default values, to allow for longer tool usage descriptions when using larger generation budgets.
	if args.max_model_len is not None and args.max_tokens == 4096 and args.max_usage_tokens == 4096*2:
		args.max_tokens = max(args.max_tokens, args.max_model_len // 2)
		args.max_usage_tokens = max(args.max_usage_tokens, args.max_model_len - args.max_tokens)

	run_config: Dict[str, Any] = {
		"mode": args.mode,
		"output_policy": args.output_policy,
		"output_raw": args.output_raw,
		"input": str(input_path),
		"subtools": str(subtools_path),
		"model": args.model,
		"batch_size": args.batch_size,
		"resume": args.resume,
		"resume_index": resume_index,
		"temperature": args.temperature,
		"top_p": args.top_p,
		"repetition_penalty": args.repetition_penalty,
		"candidate_seed": args.candidate_seed,
		"candidate_tools": effective_candidate_tools,
		"max_tokens": args.max_tokens,
		"max_usage_tokens": args.max_usage_tokens,
		"gpu_memory_utilization": args.gpu_memory_utilization,
		"language_model_only": args.language_model_only,
		"enforce_eager": args.enforce_eager,
		"max_model_len": args.max_model_len,
		"max_num_batched_tokens": args.max_num_batched_tokens,
		"limit_mm_per_prompt_image": args.limit_mm_per_prompt_image,
		"limit_mm_per_prompt_video": args.limit_mm_per_prompt_video,
		"gdn_prefill_backend": args.gdn_prefill_backend,
		"inference_api": args.inference_api,
		"records": len(records),
	}

	print(f"[INFO] Starting evaluation with config: {json.dumps(run_config, indent=2)}")

	wandb_run = _init_wandb_if_enabled(args, run_config)

	config_file = output_dir / "run_config.json"
	config_file.write_text(
		json.dumps(
			run_config,
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)

	if args.resume:
		raw_existing = read_jsonl_prefix(raw_pred_file)
		compat_existing = read_jsonl_prefix(vllm_compat_file)

		prompt_index_by_id: Dict[str, int] = {}
		for idx, prompt in enumerate(prompts):
			cid = str(prompt.custom_id)
			if cid in prompt_index_by_id:
				raise ValueError(f"Duplicate custom_id '{cid}' in prompts; cannot resume safely.")
			prompt_index_by_id[cid] = idx

		def _build_prediction_map(
			rows: List[Dict[str, Any]],
			source_name: str,
			extractor: Any,
		) -> Dict[str, str]:
			predictions_by_id: Dict[str, str] = {}
			duplicates = 0
			for row in rows:
				cid = str(row.get("custom_id") or "").strip()
				if not cid:
					raise ValueError(f"{source_name} row missing custom_id; cannot resume.")
				if cid not in prompt_index_by_id:
					raise ValueError(
						f"{source_name} custom_id '{cid}' not found in prompts. "
						"Use --resume false or a new output directory."
					)
				prediction = str(extractor(row) or "")
				if cid in predictions_by_id:
					duplicates += 1
				predictions_by_id[cid] = prediction
			if duplicates:
				print(
					f"[WARN] {source_name} contains {duplicates} duplicate custom_id entries; "
					"keeping last occurrence."
				)
			return predictions_by_id

		raw_by_id = _build_prediction_map(
			raw_existing,
			"raw_predictions.jsonl",
			lambda row: row.get("model_reply"),
		)
		compat_by_id = _build_prediction_map(
			compat_existing,
			"vllm_compat_predictions.jsonl",
			extract_compat_prediction,
		)

		if raw_by_id and compat_by_id:
			mismatched = [
				cid
				for cid in (set(raw_by_id) & set(compat_by_id))
				if raw_by_id[cid] != compat_by_id[cid]
			]
			if mismatched:
				print(
					"[WARN] Existing raw/compat files disagree on "
					f"{len(mismatched)} shared custom_id values; preferring raw entries."
				)

		combined_by_id = dict(compat_by_id)
		combined_by_id.update(raw_by_id)

		for cid, pred in combined_by_id.items():
			predictions[prompt_index_by_id[cid]] = pred

		missing_indices = [idx for idx, pred in enumerate(predictions) if pred is None]
		existing_count = len(prompts) - len(missing_indices)
		resume_index = existing_count

		if existing_count > 0:
			existing_prompt_rows: List[Any] = []
			existing_pred_values: List[str] = []
			for idx, pred in enumerate(predictions):
				if pred is None:
					continue
				existing_prompt_rows.append(prompts[idx])
				existing_pred_values.append(pred)

			resume_raw_rows, resume_compat_rows = build_prediction_rows(
				existing_prompt_rows,
				existing_pred_values,
			)
			# Optionally include the original input messages in raw prediction rows.
			if args.output_raw:
				if raw_tokenizer is None:
					raise RuntimeError("--output-raw requires a tokenizer to compute token sizes.")
				_attach_raw_token_sizes(
					resume_raw_rows,
					existing_prompt_rows,
					raw_tokenizer,
					args.inference_api,
				)
				for i, r in enumerate(resume_raw_rows):
					# Attach the prompt messages that were sent to the model for this prediction.
					r["messages"] = existing_prompt_rows[i].messages
			write_jsonl(raw_pred_file, resume_raw_rows)
			write_jsonl(vllm_compat_file, resume_compat_rows)
			print(
				f"[INFO] Resuming with {existing_count} existing predictions; "
				f"{len(missing_indices)} missing."
			)
		else:
			write_jsonl(raw_pred_file, [])
			write_jsonl(vllm_compat_file, [])
	else:
		for file_path in (raw_pred_file, vllm_compat_file):
			if file_path.exists():
				file_path.unlink()

	if missing_indices:
		missing_messages_list = [messages_list[idx] for idx in missing_indices]

		def on_batch_complete(batch_start: int, batch_predictions: List[str]) -> None:
			batch_end = batch_start + len(batch_predictions)
			batch_indices = missing_indices[batch_start:batch_end]
			batch_prompts = [prompts[idx] for idx in batch_indices]
			batch_raw_rows, batch_compat_rows = build_prediction_rows(
				batch_prompts,
				batch_predictions,
			)
			# Optionally include the original input messages in raw prediction rows.
			if args.output_raw:
				if raw_tokenizer is None:
					raise RuntimeError("--output-raw requires a tokenizer to compute token sizes.")
				_attach_raw_token_sizes(
					batch_raw_rows,
					batch_prompts,
					raw_tokenizer,
					args.inference_api,
				)
				for i, r in enumerate(batch_raw_rows):
					# Attach the prompt messages that were sent to the model for this prediction.
					r["messages"] = batch_prompts[i].messages
			append_jsonl(raw_pred_file, batch_raw_rows)
			append_jsonl(vllm_compat_file, batch_compat_rows)
			for idx, pred in zip(batch_indices, batch_predictions):
				predictions[idx] = pred
			print(
				f"[SAVED] predictions for {batch_end}/{len(missing_indices)} missing prompts"
			)

		new_predictions = run_vllm(
			model=args.model,
			messages_list=missing_messages_list,
			temperature=args.temperature,
			top_p=args.top_p,
			repetition_penalty=args.repetition_penalty,
			max_tokens=args.max_tokens,
			tensor_parallel_size=args.tensor_parallel_size,
			gpu_memory_utilization=args.gpu_memory_utilization,
			language_model_only=args.language_model_only,
			enforce_eager=args.enforce_eager,
			max_model_len=args.max_model_len,
			max_num_batched_tokens=args.max_num_batched_tokens,
			limit_mm_per_prompt_image=args.limit_mm_per_prompt_image,
			limit_mm_per_prompt_video=args.limit_mm_per_prompt_video,
			gdn_prefill_backend=args.gdn_prefill_backend,
			dtype=args.dtype,
			trust_remote_code=args.trust_remote_code,
			seed=args.seed,
			inference_api=args.inference_api,
			batch_size=args.batch_size,
			on_batch_complete=on_batch_complete,
		)
		if len(new_predictions) != len(missing_indices):
			raise RuntimeError(
				"Prediction count mismatch while resuming missing custom_id entries: "
				f"got {len(new_predictions)}, expected {len(missing_indices)}"
			)
		for idx, pred in zip(missing_indices, new_predictions):
			predictions[idx] = pred
	else:
		print("[INFO] All predictions already exist; skipping inference.")

	if any(pred is None for pred in predictions):
		raise RuntimeError(
			f"Prediction count mismatch: got {len(predictions) - predictions.count(None)}, "
			f"expected {len(prompts)}"
		)

	print(f"[SAVED] labels: {labels_file}")
	print(f"[SAVED] raw predictions: {raw_pred_file}")
	print(f"[SAVED] scorer-compatible predictions: {vllm_compat_file}")

	if args.skip_scoring:
		print("[INFO] Scoring skipped via --skip-scoring")
		if wandb_run is not None:
			wandb_run.finish()
		return

	model_tag = sanitize_name(Path(args.model).name or args.model)
	mode_tag = sanitize_name(args.mode)
	model_scores_dir = output_dir / "model_scores"
	model_scores_dir.mkdir(parents=True, exist_ok=True)
	model_score_file = model_scores_dir / f"{model_tag}.{mode_tag}.scores.jsonl"

	summary = run_model_score(vllm_compat_file, labels_file, model_score_file, repo_root)
	print(f"[SAVED] model scores: {model_score_file}")
	print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))

	# Save the summary as a separate JSON file for easier parsing by scripts, and include the config for reference.
	summary_file = output_dir / f"{model_tag}.{mode_tag}.summary.json"
	summary_file.write_text(
		json.dumps(
			{
				"summary": summary,
				"config": {
					"model": args.model,
					"mode": args.mode,
					"output_policy": args.output_policy,
					"inference_api": args.inference_api,
				},
			},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)
	print(f"[SAVED] summary: {summary_file}")
	_log_wandb_summary(wandb_run, summary)

	if not args.run_dim_score:
		if wandb_run is not None:
			wandb_run.finish()
		return

	dim_output_dir = output_dir / "dim_scores"
	dim_output_dir.mkdir(parents=True, exist_ok=True)
	run_dim_score(
		repo_root=repo_root,
		models_folder=model_scores_dir,
		output_folder=dim_output_dir,
		subtools=subtools_path if subtools_path.exists() else None,
		use_hf=args.dim_use_hf,
		hf_dataset=args.dim_hf_dataset,
		labels_file=Path(args.dim_labels_file).resolve() if args.dim_labels_file else None,
		no_tqdm=args.dim_no_tqdm,
		verbose=args.verbose,
	)
	print(f"[SAVED] dimensional scores folder: {dim_output_dir}")

	if wandb_run is not None:
		wandb_run.finish()


if __name__ == "__main__":
	main()
