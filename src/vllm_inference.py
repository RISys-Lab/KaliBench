#!/usr/bin/env python3
"""vLLM inference wrappers."""

from typing import Any, Callable, Dict, List, Optional, Sequence

from vllm_runtime import (
    configure_triton_cache_dir,
    looks_like_flashinfer_gdn_build_failure,
    normalize_compiler_env,
)

def run_vllm(
    model: str,
    messages_list: Sequence[List[Dict[str, str]]],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    language_model_only: bool,
    enforce_eager: bool,
    max_model_len: Optional[int],
    max_num_batched_tokens: Optional[int],
    limit_mm_per_prompt_image: Optional[int],
    limit_mm_per_prompt_video: Optional[int],
    gdn_prefill_backend: str,
    dtype: str,
    trust_remote_code: bool,
    seed: Optional[int],
    inference_api: str,
    batch_size: Optional[int] = None,
    start_index: int = 0,
    on_batch_complete: Optional[Callable[[int, List[str]], None]] = None,
) -> List[str]:
    configure_triton_cache_dir()
    normalize_compiler_env()

    normalized_inference_api = "generate-template" if inference_api == "non-thinking" else inference_api

    from vllm import LLM, SamplingParams

    llm_kwargs: Dict[str, Any] = dict(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        language_model_only=language_model_only,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
    )

    backend = gdn_prefill_backend.strip().lower()
    if backend not in {"auto", "flashinfer", "triton"}:
        raise ValueError(
            "gdn_prefill_backend must be one of: auto, flashinfer, triton"
        )
    if backend != "auto":
        llm_kwargs["gdn_prefill_backend"] = backend

    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    mm_limits: Dict[str, int] = {}
    if limit_mm_per_prompt_image is not None:
        mm_limits["image"] = limit_mm_per_prompt_image
    if limit_mm_per_prompt_video is not None:
        mm_limits["video"] = limit_mm_per_prompt_video
    if mm_limits:
        llm_kwargs["limit_mm_per_prompt"] = mm_limits # type: ignore
    if seed is not None:
        llm_kwargs["seed"] = seed

    llm = None
    try:
        llm = LLM(**llm_kwargs)
    except RuntimeError as exc:
        message = str(exc)

        should_retry_triton = (
            backend == "auto" and looks_like_flashinfer_gdn_build_failure(message)
        )
        if should_retry_triton:
            print(
                "[WARN] FlashInfer GDN prefill JIT failed; retrying "
                "with gdn_prefill_backend=triton"
            )
            llm_kwargs["gdn_prefill_backend"] = "triton"
            try:
                llm = LLM(**llm_kwargs)
            except RuntimeError as retry_exc:
                exc = retry_exc
                message = str(retry_exc)

        if llm is None:
            # Some CUDA/Inductor stacks fail during vLLM engine initialization.
            # Retrying with eager mode avoids torch.compile/cudagraph initialization.
            should_retry_eager = (
                "Engine core initialization failed" in message
                or "CUBLAS_STATUS_INVALID_VALUE" in message
                or "Worker proc" in message
                or "cancelled" in message
            )
            if not should_retry_eager or llm_kwargs.get("enforce_eager", False):
                raise exc

            print("[WARN] vLLM engine init failed; retrying with enforce_eager=True")
            llm_kwargs["enforce_eager"] = True
            llm = LLM(**llm_kwargs)

    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if start_index > len(messages_list):
        raise ValueError(
            f"start_index ({start_index}) cannot exceed number of prompts ({len(messages_list)})"
        )

    effective_batch_size = batch_size or len(messages_list)
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    tokenizer = None
    if normalized_inference_api != "chat":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)

    predictions: List[str] = []
    for batch_start in range(start_index, len(messages_list), effective_batch_size):
        batch_end = min(batch_start + effective_batch_size, len(messages_list))
        batch_messages = list(messages_list[batch_start:batch_end])

        if normalized_inference_api == "chat":
            outputs = llm.chat(batch_messages, sampling_params)  # type: ignore[arg-type]
        else:
            assert tokenizer is not None
            apply_chat_template_kwargs = dict(
                tokenize=False,
                add_generation_prompt=True,
            )
            if inference_api == "non-thinking":
                apply_chat_template_kwargs["enable_thinking"] = False

            texts = tokenizer.apply_chat_template(batch_messages, **apply_chat_template_kwargs)
            outputs = llm.generate(texts, sampling_params)

        batch_predictions: List[str] = []
        for output in outputs:
            if output.outputs:
                batch_predictions.append(output.outputs[0].text)
            else:
                batch_predictions.append("")

        predictions.extend(batch_predictions)
        if on_batch_complete is not None:
            on_batch_complete(batch_start, batch_predictions)

    return predictions
