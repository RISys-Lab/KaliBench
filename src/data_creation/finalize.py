#!/usr/bin/env python3
"""Finalize verified candidates into reproducible KaliBench train/test splits."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .common import iter_jsonl, write_jsonl
except ImportError:  # Support direct execution.
    from common import iter_jsonl, write_jsonl  # type: ignore


REQUIRED_FIELDS = (
    "query",
    "tool_name",
    "ground_truth_command",
)
PROVENANCE_FIELDS = (
    "source_request_id",
    "source_title",
    "source_subtool",
)


@dataclass(frozen=True)
class FinalizationPaths:
    out_dir: Path
    all_rows: Path
    train: Path
    test: Path
    rejected: Path
    summary: Path


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def model_verdict(row: dict[str, Any]) -> str:
    verification = row.get("model_verification")
    if not isinstance(verification, dict):
        return ""
    output = verification.get("model_output")
    if not isinstance(output, dict):
        return ""
    return str(output.get("verdict") or "").strip().upper()


def terminal_category(row: dict[str, Any]) -> str:
    verification = row.get("terminal_verification")
    if not isinstance(verification, dict):
        return ""
    return str(verification.get("category") or "").strip()


def validate_candidate(
    row: dict[str, Any],
    *,
    require_model_verification: bool,
    require_terminal_verification: bool,
) -> str | None:
    for field in REQUIRED_FIELDS:
        if not str(row.get(field) or "").strip():
            return f"missing_{field}"
    if require_model_verification and model_verdict(row) != "ACCURATE":
        return "model_verification_not_accurate"
    if require_terminal_verification and terminal_category(row) != "pass_review":
        return "terminal_verification_not_pass_review"
    return None


def parse_optional_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_positional_args(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    return []


def canonical_release_row(
    row: dict[str, Any],
    *,
    custom_id: str,
    keep_provenance: bool,
) -> dict[str, Any]:
    canonical: dict[str, Any] = {
        "custom_id": custom_id,
        "query": str(row.get("query") or "").strip(),
        "tool_name": str(row.get("tool_name") or "").strip(),
        "ground_truth_command": str(
            row.get("ground_truth_command") or ""
        ).strip(),
        "optional_args": parse_optional_args(row.get("optional_args")),
        "positional_args": parse_positional_args(row.get("positional_args")),
        "model_verification": row.get("model_verification"),
        "terminal_verification": row.get("terminal_verification"),
    }
    if keep_provenance:
        canonical["source_custom_id"] = row.get("custom_id")
        for field in PROVENANCE_FIELDS:
            if field in row:
                canonical[field] = row.get(field)
    return canonical


def load_and_filter_candidates(
    input_paths: Sequence[Path],
    *,
    require_model_verification: bool,
    require_terminal_verification: bool,
    id_prefix: str,
    keep_provenance: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    accepted_source_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    seen_queries: set[str] = set()

    for input_path in input_paths:
        for row in iter_jsonl(input_path):
            reason = validate_candidate(
                row,
                require_model_verification=require_model_verification,
                require_terminal_verification=require_terminal_verification,
            )
            if reason is None:
                query_key = normalize_text(row.get("query"))
                if query_key in seen_queries:
                    reason = "duplicate_normalized_query"
                else:
                    seen_queries.add(query_key)

            if reason is not None:
                rejected_row = dict(row)
                rejected_row["finalization_rejection"] = reason
                rejected.append(rejected_row)
                rejection_counts[reason] += 1
                continue
            accepted_source_rows.append(row)

    accepted = [
        canonical_release_row(
            row,
            custom_id=f"{id_prefix}{index}",
            keep_provenance=keep_provenance,
        )
        for index, row in enumerate(accepted_source_rows)
    ]
    return accepted, rejected, rejection_counts


def split_capacity(
    groups: dict[str, list[dict[str, Any]]],
    *,
    min_test_per_tool: int,
    max_test_per_tool: int,
) -> tuple[int, int]:
    minimum = sum(
        min(len(rows), min_test_per_tool)
        for rows in groups.values()
    )
    maximum = sum(
        min(len(rows), max_test_per_tool)
        for rows in groups.values()
    )
    return minimum, maximum


def stratified_split(
    rows: Sequence[dict[str, Any]],
    *,
    test_size: int,
    min_test_per_tool: int,
    max_test_per_tool: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a target-size test set with bounded representation per tool."""
    if min_test_per_tool < 0:
        raise ValueError("min_test_per_tool must be >= 0")
    if max_test_per_tool < min_test_per_tool:
        raise ValueError("max_test_per_tool must be >= min_test_per_tool")
    if test_size < 0 or test_size > len(rows):
        raise ValueError(f"test_size must be in [0, {len(rows)}]")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("tool_name") or "")].append(row)

    minimum, maximum = split_capacity(
        groups,
        min_test_per_tool=min_test_per_tool,
        max_test_per_tool=max_test_per_tool,
    )
    if not minimum <= test_size <= maximum:
        raise ValueError(
            f"Requested test_size={test_size} is infeasible for "
            f"{len(groups)} tools with min={min_test_per_tool}, "
            f"max={max_test_per_tool}; feasible range is [{minimum}, {maximum}]."
        )

    rng = random.Random(seed)
    selected_ids: set[str] = set()
    selected_per_tool: Counter[str] = Counter()
    shuffled_groups: dict[str, list[dict[str, Any]]] = {}

    for tool_name in sorted(groups):
        group_rows = sorted(groups[tool_name], key=lambda row: str(row["custom_id"]))
        rng.shuffle(group_rows)
        shuffled_groups[tool_name] = group_rows
        base_count = min(len(group_rows), min_test_per_tool)
        for row in group_rows[:base_count]:
            selected_ids.add(str(row["custom_id"]))
        selected_per_tool[tool_name] = base_count

    while len(selected_ids) < test_size:
        eligible = [
            tool_name
            for tool_name, group_rows in shuffled_groups.items()
            if (
                selected_per_tool[tool_name]
                < min(len(group_rows), max_test_per_tool)
            )
        ]
        if not eligible:
            raise RuntimeError("No remaining test capacity before reaching test_size.")
        rng.shuffle(eligible)
        for tool_name in eligible:
            if len(selected_ids) >= test_size:
                break
            index = selected_per_tool[tool_name]
            row = shuffled_groups[tool_name][index]
            selected_ids.add(str(row["custom_id"]))
            selected_per_tool[tool_name] += 1

    test = [row for row in rows if str(row["custom_id"]) in selected_ids]
    train = [row for row in rows if str(row["custom_id"]) not in selected_ids]
    return train, test


def tool_count(rows: Iterable[dict[str, Any]]) -> int:
    return len(
        {
            str(row.get("tool_name") or "")
            for row in rows
            if str(row.get("tool_name") or "")
        }
    )


def resolve_paths(args: argparse.Namespace) -> FinalizationPaths:
    out_dir = args.out_dir.resolve()
    return FinalizationPaths(
        out_dir=out_dir,
        all_rows=(args.all_jsonl or out_dir / "kalibench_verified_all.jsonl").resolve(),
        train=(args.train_jsonl or out_dir / "kalibench_verified_train.jsonl").resolve(),
        test=(args.test_jsonl or out_dir / "kalibench_verified_test.jsonl").resolve(),
        rejected=(
            args.rejected_jsonl or out_dir / "finalization_rejected.jsonl"
        ).resolve(),
        summary=(args.summary_json or out_dir / "finalization_summary.json").resolve(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter fully verified candidates, remove duplicate normalized queries, "
            "assign stable sample IDs, and create tool-stratified train/test splits."
        )
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/data_creation/final"),
    )
    parser.add_argument("--all-jsonl", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--test-jsonl", type=Path, default=None)
    parser.add_argument("--rejected-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--min-test-per-tool", type=int, default=3)
    parser.add_argument("--max-test-per-tool", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--id-prefix", default="sample_")
    parser.add_argument("--keep-provenance", action="store_true")
    parser.add_argument(
        "--allow-missing-model-verification",
        action="store_true",
    )
    parser.add_argument(
        "--allow-missing-terminal-verification",
        action="store_true",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_paths = [path.resolve() for path in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"Input file not found: {path}")

    paths = resolve_paths(args)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    accepted, rejected, rejection_counts = load_and_filter_candidates(
        input_paths,
        require_model_verification=not args.allow_missing_model_verification,
        require_terminal_verification=not args.allow_missing_terminal_verification,
        id_prefix=args.id_prefix,
        keep_provenance=args.keep_provenance,
    )
    train, test = stratified_split(
        accepted,
        test_size=args.test_size,
        min_test_per_tool=args.min_test_per_tool,
        max_test_per_tool=args.max_test_per_tool,
        seed=args.seed,
    )

    write_jsonl(paths.all_rows, accepted)
    write_jsonl(paths.train, train)
    write_jsonl(paths.test, test)
    write_jsonl(paths.rejected, rejected)
    summary = {
        "inputs": [str(path) for path in input_paths],
        "seed": args.seed,
        "split_policy": {
            "test_size": args.test_size,
            "min_test_per_tool": args.min_test_per_tool,
            "max_test_per_tool": args.max_test_per_tool,
        },
        "counts": {
            "accepted_all": len(accepted),
            "train": len(train),
            "test": len(test),
            "rejected": len(rejected),
            "all_tools": tool_count(accepted),
            "train_tools": tool_count(train),
            "test_tools": tool_count(test),
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "outputs": {
            "all": str(paths.all_rows),
            "train": str(paths.train),
            "test": str(paths.test),
            "rejected": str(paths.rejected),
        },
    }
    paths.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[finalized] all={len(accepted)} train={len(train)} test={len(test)} "
        f"rejected={len(rejected)}"
    )
    print(
        f"[coverage] all_tools={tool_count(accepted)} "
        f"train_tools={tool_count(train)} test_tools={tool_count(test)}"
    )
    print(f"[finalized] summary -> {paths.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
