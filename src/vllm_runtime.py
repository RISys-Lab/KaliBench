#!/usr/bin/env python3
"""Runtime/environment helpers for vLLM startup stability."""

import os
from typing import Optional


def configure_triton_cache_dir() -> None:
    """Set a writable, job-local Triton cache directory when possible."""
    existing = os.environ.get("TRITON_CACHE_DIR")
    candidates = []
    if existing:
        candidates.append(existing)
    else:
        user = os.environ.get("USER", "user")
        job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        for root in (os.environ.get("SLURM_TMPDIR"), os.environ.get("TMPDIR"), "/tmp"):
            if root:
                candidates.append(os.path.join(root, f"triton_cache_{user}", f"job_{job_id}"))

    for cache_dir in candidates:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = cache_dir
            print(f"[INFO] TRITON_CACHE_DIR={cache_dir}")
            return
        except OSError:
            continue

    raise RuntimeError(
        "Unable to create a writable Triton cache directory. "
        "Set TRITON_CACHE_DIR to a writable local path."
    )


def normalize_compiler_env() -> None:
    """Ensure Triton uses a host C/C++ compiler, not CUDA compiler wrappers."""
    bad_compilers = {"nvcc", "nvc", "nvc++"}

    def is_bad(value: Optional[str]) -> bool:
        if not value:
            return False
        exe = os.path.basename(value.strip().split()[0])
        return exe in bad_compilers

    changed = False
    if is_bad(os.environ.get("CC")):
        os.environ["CC"] = "gcc"
        changed = True
    if is_bad(os.environ.get("CXX")):
        os.environ["CXX"] = "g++"
        changed = True

    if changed:
        print("[WARN] Reset CC/CXX to gcc/g++ for Triton host compilation")


def looks_like_flashinfer_gdn_build_failure(message: str) -> bool:
    """Detect FlashInfer GDN JIT failures that should fallback to Triton."""
    lower = message.lower()
    return (
        ("gdn" in lower or "gated_delta_rule" in lower)
        and (
            "flashinfer" in lower
            or "gdn_prefill_sm90" in lower
            or "flat_collective_store.hpp" in lower
        )
        and (
            "ninja build failed" in lower
            or "subcommand failed" in lower
            or "cuda::ptx" in lower
            or "tensormap_replace_global_dim" in lower
        )
    )
