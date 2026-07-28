#!/usr/bin/env python3
"""
Export a trained Unsloth LoRA adapter into a merged FP16 model.

Examples
--------
# Case 1: adapter directory already contains enough info to load directly
python export_unsloth_lora_to_f16.py \
    --model path/to/adapter_or_checkpoint_dir \
    --output-dir ./merged_fp16_model

# Case 2: load base model first, then load adapter separately
python export_unsloth_lora_to_f16.py \
    --base-model unsloth/Meta-Llama-3.1-8B-Instruct \
    --adapter path/to/lora_adapter_dir \
    --output-dir ./merged_fp16_model

# Optional: push merged model to Hugging Face Hub
python export_unsloth_lora_to_f16.py \
    --model path/to/adapter_or_checkpoint_dir \
    --output-dir ./merged_fp16_model \
    --push-to-hub your-username/your-model-fp16 \
    --hf-token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained Unsloth LoRA adapter into a merged FP16 model."
    )

    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path or HF repo for a trained Unsloth checkpoint / adapter repo that can be "
            "loaded directly with FastLanguageModel.from_pretrained()."
        ),
    )
    source.add_argument(
        "--base-model",
        type=str,
        default=None,
        help=(
            "Base model name or local path. Use this together with --adapter when the "
            "adapter must be loaded separately."
        ),
    )

    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help=(
            "Path or HF repo for the LoRA adapter. Required with --base-model. "
            "Ignored when --model is used unless --force-load-adapter is set."
        ),
    )
    parser.add_argument(
        "--force-load-adapter",
        action="store_true",
        help=(
            "When --model is provided, also call model.load_adapter(--adapter). "
            "Useful only if your adapter is stored separately."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the merged FP16 model will be saved.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max sequence length used when loading with Unsloth.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Load dtype. Use float16 for an f16 export target.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help=(
            "Load model in 4-bit before exporting. This can reduce load-time memory usage, "
            "but merged export still writes a 16-bit model."
        ),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help='Optional device_map passed through internally if supported, for example "auto".',
    )
    parser.add_argument(
        "--save-tokenizer-only",
        action="store_true",
        help="Only save tokenizer after merge call. Usually not needed, since merge save already includes it.",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Request safetensors serialization if supported by your installed Unsloth version.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional Hugging Face Hub repo to push the merged model to.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF token for push_to_hub_merged. Can also come from HF_TOKEN env var.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model.",
    )

    args = parser.parse_args()

    if not args.model and not args.base_model:
        parser.error("You must provide either --model or --base-model.")

    if args.base_model and not args.adapter:
        parser.error("--adapter is required when using --base-model.")

    if args.force_load_adapter and not args.adapter:
        parser.error("--force-load-adapter requires --adapter.")

    return args


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype = resolve_dtype(args.dtype)

    load_target = args.model if args.model else args.base_model

    print(f"[INFO] Loading model from: {load_target}")
    print(f"[INFO] max_seq_length={args.max_seq_length}")
    print(f"[INFO] dtype={args.dtype}")
    print(f"[INFO] load_in_4bit={args.load_in_4bit}")

    # Keep kwargs conservative for compatibility across Unsloth versions.
    load_kwargs = {
        "model_name": load_target,
        "max_seq_length": args.max_seq_length,
        "dtype": dtype,
        "load_in_4bit": args.load_in_4bit,
    }

    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    should_load_adapter = False
    if args.base_model and args.adapter:
        should_load_adapter = True
    elif args.model and args.force_load_adapter and args.adapter:
        should_load_adapter = True

    if should_load_adapter:
        print(f"[INFO] Loading adapter from: {args.adapter}")
        model.load_adapter(args.adapter)

    return model, tokenizer


def save_merged_fp16(
    model,
    tokenizer,
    output_dir: str,
    safe_serialization: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Saving merged FP16 model to: {output_dir}")

    save_kwargs = {
        "save_directory": output_dir,
        "tokenizer": tokenizer,
        "save_method": "merged_16bit",
    }

    # Some Unsloth versions may accept additional save kwargs.
    if safe_serialization:
        save_kwargs["safe_serialization"] = True

    try:
        model.save_pretrained_merged(**save_kwargs)
    except TypeError:
        # Fallback for versions with a simpler signature.
        print("[WARN] Installed Unsloth version does not accept one or more optional save args. Retrying with minimal arguments.")
        model.save_pretrained_merged(
            output_dir,
            tokenizer,
            save_method="merged_16bit",
        )

    print("[INFO] Merged model export finished.")


def maybe_save_tokenizer(tokenizer, output_dir: str) -> None:
    print("[INFO] Saving tokenizer explicitly.")
    tokenizer.save_pretrained(output_dir)


def maybe_push_to_hub(
    model,
    tokenizer,
    repo_id: str | None,
    hf_token: str | None,
) -> None:
    if not repo_id:
        return

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "--push-to-hub was provided, but no token was found. "
            "Pass --hf-token or set HF_TOKEN."
        )

    print(f"[INFO] Pushing merged FP16 model to Hub: {repo_id}")
    model.push_to_hub_merged(
        repo_id,
        tokenizer,
        save_method="merged_16bit",
        token=token,
    )
    print("[INFO] Push to Hub finished.")


def main() -> int:
    args = parse_args()

    try:
        model, tokenizer = load_model_and_tokenizer(args)

        save_merged_fp16(
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            safe_serialization=args.safe_serialization,
        )

        if args.save_tokenizer_only:
            maybe_save_tokenizer(tokenizer, args.output_dir)

        maybe_push_to_hub(
            model=model,
            tokenizer=tokenizer,
            repo_id=args.push_to_hub,
            hf_token=args.hf_token,
        )

        print("[OK] Done.")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())