"""Shared CLI options and Transformers inference for the demo scripts."""

import argparse
import math


def build_parser(description, default_model, max_new_tokens, temperature):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", required=True, help="Request to translate into a Kali/Linux command.")
    parser.add_argument("--model", default=default_model, help="Hugging Face model ID or local model path.")
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=temperature, help="Sampling temperature; 0 uses greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling probability.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling cutoff; 0 disables it.")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16", help="Model weight dtype.")
    parser.add_argument("--device-map", default="auto", help="Model placement, e.g. auto, cpu, or cuda:0.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for generation.")
    return parser


def parse_args(parser, argv=None):
    args = parser.parse_args(argv)
    if not args.query.strip():
        parser.error("--query must not be empty")
    if not args.model.strip():
        parser.error("--model must not be empty")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be a finite number greater than or equal to 0")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")
    if args.top_k < 0:
        parser.error("--top-k must be greater than or equal to 0")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be greater than or equal to 0")
    return args


def run_inference(args, messages):
    # Keep --help and argument validation available without model dependencies.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if args.seed is not None:
        set_seed(args.seed)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device_map
    ).eval()
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "use_cache": True,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update(
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
        )
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    completion = outputs[0, inputs["input_ids"].shape[-1]:]
    print(tokenizer.decode(completion, skip_special_tokens=True))
