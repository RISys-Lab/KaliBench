#!/usr/bin/env python3
"""Scoring orchestration helpers."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def _load_run_config(folder: Path) -> Dict[str, Any]:
    run_config_file = folder / "run_config.json"
    if not run_config_file.exists():
        return {}
    return json.loads(run_config_file.read_text(encoding="utf-8"))


def _resolve_inputs(
    folder: Optional[Path],
    vllm_file: Optional[Path],
    labels_file: Optional[Path],
) -> tuple[Path, Path, Path]:
    if folder is not None:
        eval_folder = folder.resolve()
        resolved_vllm = (eval_folder / "vllm_compat_predictions.jsonl").resolve()
        resolved_labels = (eval_folder / "labels.jsonl").resolve()
        return eval_folder, resolved_vllm, resolved_labels

    if vllm_file is None or labels_file is None:
        raise ValueError("Provide either --folder, or both --vllm-file and --labels-file")

    resolved_vllm = vllm_file.resolve()
    resolved_labels = labels_file.resolve()
    eval_folder = resolved_vllm.parent
    return eval_folder, resolved_vllm, resolved_labels


def _default_output_file(eval_folder: Path, run_config: Dict[str, Any]) -> Path:
    model_raw = str(run_config.get("model") or "unknown-model")
    mode_raw = str(run_config.get("mode") or "unknown-mode")
    model_tag = sanitize_name(Path(model_raw).name or model_raw)
    mode_tag = sanitize_name(mode_raw)
    return eval_folder / "model_scores" / f"{model_tag}.{mode_tag}.scores.jsonl"


def run_model_score(vllm_file: Path, labels_file: Path, out_file: Path, repo_root: Path) -> Dict[str, float]:
    try:
        from model_score import score_batch
    except ImportError:
        # If the import fails, it might be because the current working directory is not the repo root.
        # In that case, we add the repo root to sys.path and try again.
        print("[WARN] model_score import failed, trying again after adding repo root to sys.path")
        sys.path.insert(0, str(repo_root))
        from Scoring.model_score import score_batch

    return score_batch(str(vllm_file), str(labels_file), out_file=str(out_file))


def run_dim_score(
    repo_root: Path,
    models_folder: Path,
    output_folder: Path,
    subtools: Optional[Path],
    use_hf: bool,
    hf_dataset: str,
    labels_file: Optional[Path],
    no_tqdm: bool,
    verbose: bool,
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "Scoring" / "dim_score.py"),
        "--models-folder",
        str(models_folder),
        "--output-folder",
        str(output_folder),
        "--hf-dataset",
        hf_dataset,
    ]
    if subtools:
        cmd.extend(["--subtools", str(subtools)])
    if use_hf:
        cmd.append("--use-hf")
    if labels_file:
        cmd.extend(["--labels-file", str(labels_file)])
    if no_tqdm:
        cmd.append("--no-tqdm")
    if verbose:
        cmd.append("--verbose")
    subprocess.run(cmd, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run model_score on KaliBench predictions. "
            "Use --folder to auto-discover labels/predictions/run_config, "
            "or pass --vllm-file and --labels-file explicitly."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Evaluation output folder containing labels.jsonl, vllm_compat_predictions.jsonl, and run_config.json.",
    )
    parser.add_argument("--vllm-file", type=Path, default=None, help="Path to vllm_compat_predictions.jsonl")
    parser.add_argument("--labels-file", type=Path, default=None, help="Path to labels.jsonl")
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help=(
            "Output score JSONL path. Default: <folder>/model_scores/{model}.{mode}.scores.jsonl "
            "using run_config.json values."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used for fallback Scoring.model_score import.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.folder is None and (args.vllm_file is None or args.labels_file is None):
        raise ValueError("Provide either --folder, or both --vllm-file and --labels-file")

    eval_folder, resolved_vllm, resolved_labels = _resolve_inputs(
        folder=args.folder,
        vllm_file=args.vllm_file,
        labels_file=args.labels_file,
    )

    if not resolved_vllm.exists():
        raise FileNotFoundError(f"Predictions file not found: {resolved_vllm}")
    if not resolved_labels.exists():
        raise FileNotFoundError(f"Labels file not found: {resolved_labels}")

    run_config = _load_run_config(eval_folder)
    out_file = args.out_file.resolve() if args.out_file else _default_output_file(eval_folder, run_config)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_raw = str(run_config.get("model") or "unknown-model")
    mode_raw = str(run_config.get("mode") or "unknown-mode")
    model_tag = sanitize_name(Path(model_raw).name or model_raw)
    mode_tag = sanitize_name(mode_raw)

    summary = run_model_score(
        vllm_file=resolved_vllm,
        labels_file=resolved_labels,
        out_file=out_file,
        repo_root=args.repo_root.resolve(),
    )

    print(f"[INFO] evaluation folder: {eval_folder}")
    print(f"[INFO] run_config: {eval_folder / 'run_config.json'}")
    print(f"[INFO] model: {model_raw}")
    print(f"[INFO] mode: {mode_raw}")
    print(f"[SAVED] model scores: {out_file}")
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))

    # Save summary using the same structure as evaluate.py.
    summary_file = eval_folder / "model_scores" / f"{model_tag}.{mode_tag}.summary.json"
    summary_file.write_text(
        json.dumps(
            {
                "summary": summary,
                "config": {
                    "model": model_raw,
                    "mode": mode_raw,
                    "output_policy": run_config.get("output_policy"),
                    "inference_api": run_config.get("inference_api"),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[SAVED] summary: {summary_file}")


if __name__ == "__main__":
    main()
