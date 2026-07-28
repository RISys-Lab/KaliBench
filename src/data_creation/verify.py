#!/usr/bin/env python3
"""Documentation-grounded verification for generated KaliBench candidates.

This is the refactored form of ``legacy_data_creation/verify_qwen3max.py``.
It records the exact verifier prompt, supports checkpointed API calls, writes
the released ``model_verification`` schema, and produces regeneration inputs
for sub-tools that have no accepted examples.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .common import (
        append_jsonl,
        build_openai_client,
        iter_jsonl,
        load_latest_by_id,
        parse_json_container,
        request_chat_completion,
        write_jsonl,
    )
except ImportError:  # Support direct execution.
    from common import (  # type: ignore
        append_jsonl,
        build_openai_client,
        iter_jsonl,
        load_latest_by_id,
        parse_json_container,
        request_chat_completion,
        write_jsonl,
    )


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-max"

VERIFIER_SYSTEM_PROMPT = (
    "You are a meticulous CLI command verifier. "
    "Judge whether query-command pairs are accurate or hallucinated using only "
    "the provided tool usage documentation. Return strict JSON only."
)

RELEASE_FIELDS = (
    "custom_id",
    "query",
    "tool_name",
    "ground_truth_command",
    "optional_args",
    "positional_args",
)


@dataclass(frozen=True)
class VerificationPaths:
    out_dir: Path
    prompts: Path
    results: Path
    clean: Path
    rejected: Path
    errors: Path
    remaining_subtools: Path
    merged_clean: Path


def normalized_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def infer_tool_name(row: dict[str, Any]) -> str:
    tool_name = str(row.get("tool_name") or "").strip()
    if tool_name:
        return tool_name
    command = str(row.get("ground_truth_command") or "").strip()
    return command.split()[0] if command else ""


def load_usage_maps(
    tool_subtools_jsonl: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load case-insensitive sub-tool/title documentation mappings."""
    subtool_to_usage: dict[str, str] = {}
    title_to_usage: dict[str, str] = {}
    for row in iter_jsonl(tool_subtools_jsonl):
        subtool = normalized_key(row.get("subtool"))
        title = normalized_key(row.get("title"))
        usage_code = str(row.get("usage_code") or "")
        if subtool and subtool not in subtool_to_usage:
            subtool_to_usage[subtool] = usage_code
        if title and title not in title_to_usage:
            title_to_usage[title] = usage_code
    return subtool_to_usage, title_to_usage


def usage_candidates(row: dict[str, Any]) -> list[str]:
    candidates = [
        infer_tool_name(row),
        row.get("source_subtool"),
        row.get("source_title"),
    ]
    tool_name = infer_tool_name(row)
    if "/" in tool_name:
        candidates.extend(tool_name.split("/", 1))
    return [normalized_key(candidate) for candidate in candidates if candidate]


def find_usage_code(
    row: dict[str, Any],
    subtool_to_usage: dict[str, str],
    title_to_usage: dict[str, str],
) -> str:
    for candidate in usage_candidates(row):
        if candidate in subtool_to_usage:
            return subtool_to_usage[candidate]
        if candidate in title_to_usage:
            return title_to_usage[candidate]
    return ""


def make_verifier_prompt(row: dict[str, Any], usage_code: str) -> str:
    return f"""Verify whether the following query-ground truth pair is accurate or hallucinated.

Query:
{str(row.get("query") or "")}

Ground-truth command:
{str(row.get("ground_truth_command") or "")}

Tool name:
{infer_tool_name(row)}

Tool documentation:
<usage_code>
{usage_code}
</usage_code>

Verification criteria:
1. Every flag and argument in the ground-truth command must appear in the tool usage documentation.
2. Every flag-value binding must be used correctly.
3. The command must reasonably implement the requested query using only documented capabilities.
4. If any flag does not exist, any argument is unsupported, or the requested behavior is not documented, classify it as HALLUCINATED.
5. If the command is valid but the synthetic target/file/device may not exist at runtime, do not reject for that reason alone.

Return strict JSON only:
{{
  "verdict": "ACCURATE" or "HALLUCINATED",
  "reasons": "short explanation"
}}
""".strip()


def build_messages(row: dict[str, Any], usage_code: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": make_verifier_prompt(row, usage_code)},
    ]


def parse_verifier_output(text: str) -> dict[str, str]:
    try:
        parsed = parse_json_container(text, dict)
    except (ValueError, TypeError):
        return {
            "verdict": "PARSE_ERROR",
            "reasons": str(text or "")[:1000],
        }

    verdict = str(parsed.get("verdict") or "").strip().upper()
    if verdict not in {"ACCURATE", "HALLUCINATED"}:
        verdict = "PARSE_ERROR"
    return {
        "verdict": verdict,
        "reasons": str(parsed.get("reasons") or "").strip(),
    }


def build_verification_prompts(
    labels_jsonl: Path,
    tool_subtools_jsonl: Path,
    output_path: Path,
    *,
    limit: int | None = None,
) -> int:
    subtool_to_usage, title_to_usage = load_usage_maps(tool_subtools_jsonl)
    prompts: list[dict[str, Any]] = []
    for row_index, row in enumerate(iter_jsonl(labels_jsonl)):
        if limit is not None and row_index >= limit:
            break
        usage_code = find_usage_code(row, subtool_to_usage, title_to_usage)
        prompts.append(
            {
                "custom_id": row.get("custom_id"),
                "tool_name": infer_tool_name(row),
                "query": row.get("query", ""),
                "ground_truth_command": row.get("ground_truth_command", ""),
                "messages": build_messages(row, usage_code),
                "usage_found": bool(usage_code),
            }
        )
    return write_jsonl(output_path, prompts)


def checkpoint_has_error(row: dict[str, Any]) -> bool:
    if str(row.get("verify_error") or "").strip():
        return True
    verification = row.get("model_verification")
    if not isinstance(verification, dict):
        return True
    output = verification.get("model_output")
    return not isinstance(output, dict) or str(output.get("verdict")) in {
        "API_ERROR",
        "PARSE_ERROR",
        "",
    }


def run_verification_api(
    args: argparse.Namespace,
    paths: VerificationPaths,
) -> int:
    from tqdm import tqdm

    client = build_openai_client(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
    )
    subtool_to_usage, title_to_usage = load_usage_maps(args.tool_subtools_jsonl)
    checkpoints = load_latest_by_id(paths.results)
    labels = list(iter_jsonl(args.labels_jsonl))
    if args.limit is not None:
        labels = labels[: args.limit]

    completed = 0
    for row in tqdm(labels, desc=f"{args.model} verification"):
        custom_id = str(row.get("custom_id") or "").strip()
        previous = checkpoints.get(custom_id)
        if previous is not None and not (
            args.retry_errors and checkpoint_has_error(previous)
        ):
            continue

        usage_code = find_usage_code(row, subtool_to_usage, title_to_usage)
        messages = build_messages(row, usage_code)
        if not usage_code and args.require_usage:
            raw_text = ""
            model_output = {
                "verdict": "MISSING_USAGE",
                "reasons": "No documentation chunk matched this candidate.",
            }
            error = "missing_usage"
        else:
            try:
                raw_text = request_chat_completion(
                    client,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                model_output = parse_verifier_output(raw_text)
                error = ""
            except Exception as exc:
                raw_text = ""
                model_output = {
                    "verdict": "API_ERROR",
                    "reasons": f"{type(exc).__name__}: {exc}",
                }
                error = f"{type(exc).__name__}: {exc}"

        result = dict(row)
        result["tool_name"] = infer_tool_name(row)
        result["usage_found"] = bool(usage_code)
        result["model_verification"] = {
            "model": args.model,
            "model_output": model_output,
        }
        result["raw_verifier_response"] = raw_text
        result["verify_error"] = error
        append_jsonl(paths.results, result)
        checkpoints[custom_id] = result
        completed += 1
        if args.sleep > 0:
            time.sleep(args.sleep)
    return completed


def verifier_verdict(row: dict[str, Any]) -> str:
    verification = row.get("model_verification")
    if isinstance(verification, dict):
        output = verification.get("model_output")
        if isinstance(output, dict):
            return str(output.get("verdict") or "").upper()

    # Backward compatibility with the legacy result shape.
    output = row.get("model_output")
    if isinstance(output, dict):
        return str(output.get("verdict") or "").upper()
    return ""


def clean_accurate_row(row: dict[str, Any], model: str) -> dict[str, Any]:
    clean = {
        field: row.get(field)
        for field in RELEASE_FIELDS
    }
    for field in ("source_request_id", "source_title", "source_subtool"):
        if field in row:
            clean[field] = row.get(field)

    verification = row.get("model_verification")
    if not isinstance(verification, dict):
        legacy_output = row.get("model_output")
        verification = {
            "model": model,
            "model_output": (
                legacy_output if isinstance(legacy_output, dict) else {}
            ),
        }
    clean["model_verification"] = verification
    return clean


def filter_verification_results(
    *,
    results_path: Path,
    clean_path: Path,
    rejected_path: Path,
    errors_path: Path,
    model: str,
) -> tuple[int, int, int]:
    if not results_path.exists():
        raise FileNotFoundError(
            f"Verification result file not found: {results_path}. "
            "Run --stage verify first."
        )
    accurate: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in load_latest_by_id(results_path).values():
        verdict = verifier_verdict(row)
        if verdict == "ACCURATE":
            accurate.append(clean_accurate_row(row, model))
        elif verdict == "HALLUCINATED":
            rejected.append(row)
        else:
            errors.append(row)

    write_jsonl(clean_path, accurate)
    write_jsonl(rejected_path, rejected)
    write_jsonl(errors_path, errors)
    return len(accurate), len(rejected), len(errors)


def write_remaining_subtools(
    *,
    clean_jsonl: Path,
    tool_subtools_jsonl: Path,
    output_path: Path,
) -> tuple[int, int]:
    covered = {
        normalized_key(
            row.get("source_subtool") or infer_tool_name(row)
        )
        for row in iter_jsonl(clean_jsonl)
    }
    covered.discard("")

    remaining: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in iter_jsonl(tool_subtools_jsonl):
        subtool = normalized_key(row.get("subtool"))
        if not subtool or subtool in covered or subtool in seen:
            continue
        remaining.append(row)
        seen.add(subtool)
    write_jsonl(output_path, remaining)
    return len(covered), len(remaining)


def numeric_id(row: dict[str, Any]) -> tuple[int, str]:
    raw = str(row.get("custom_id") or "")
    match = re.search(r"\d+", raw)
    return (int(match.group()) if match else 10**18, raw)


def merge_clean_files(paths: Iterable[Path], output: Path) -> int:
    latest: dict[str, dict[str, Any]] = {}
    anonymous_rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            custom_id = str(row.get("custom_id") or "").strip()
            if custom_id:
                latest[custom_id] = row
            else:
                anonymous_rows.append(row)
    rows = [*latest.values(), *anonymous_rows]
    rows.sort(key=numeric_id)
    return write_jsonl(output, rows)


def resolve_paths(args: argparse.Namespace) -> VerificationPaths:
    out_dir = args.out_dir.resolve()
    return VerificationPaths(
        out_dir=out_dir,
        prompts=(
            args.verification_prompts_jsonl
            or out_dir / "verification_prompts.jsonl"
        ).resolve(),
        results=(
            args.verify_results_jsonl
            or out_dir / "verification_results.jsonl"
        ).resolve(),
        clean=(
            args.clean_jsonl or out_dir / "model_verified.jsonl"
        ).resolve(),
        rejected=(
            args.rejected_jsonl or out_dir / "model_rejected.jsonl"
        ).resolve(),
        errors=(
            args.verifier_errors_jsonl
            or out_dir / "verification_errors.jsonl"
        ).resolve(),
        remaining_subtools=(
            args.remaining_subtools_jsonl
            or out_dir / "remaining_subtools.jsonl"
        ).resolve(),
        merged_clean=(
            args.merged_clean_jsonl or out_dir / "model_verified_merged.jsonl"
        ).resolve(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare documentation-grounded verifier prompts, call an "
            "OpenAI-compatible model, and filter generated labels."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "verify", "filter"],
        default="all",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/data_creation/verification"),
    )
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--tool-subtools-jsonl", type=Path, required=True)
    parser.add_argument("--verification-prompts-jsonl", type=Path, default=None)
    parser.add_argument("--verify-results-jsonl", type=Path, default=None)
    parser.add_argument("--clean-jsonl", type=Path, default=None)
    parser.add_argument("--rejected-jsonl", type=Path, default=None)
    parser.add_argument("--verifier-errors-jsonl", type=Path, default=None)
    parser.add_argument("--remaining-subtools-jsonl", type=Path, default=None)
    parser.add_argument("--merge-clean", type=Path, nargs="*", default=None)
    parser.add_argument("--merged-clean-jsonl", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DASHSCOPE_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--require-usage",
        action="store_true",
        help=(
            "Do not call the verifier for candidates with no matched documentation. "
            "By default, empty documentation is preserved to match the legacy pipeline."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.labels_jsonl = args.labels_jsonl.resolve()
    args.tool_subtools_jsonl = args.tool_subtools_jsonl.resolve()
    paths = resolve_paths(args)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "prepare"}:
        count = build_verification_prompts(
            args.labels_jsonl,
            args.tool_subtools_jsonl,
            paths.prompts,
            limit=args.limit,
        )
        print(f"[prepared] verification prompts: {count} -> {paths.prompts}")

    if args.stage in {"all", "verify"}:
        completed = run_verification_api(args, paths)
        print(f"[verified] new API checkpoints: {completed} -> {paths.results}")

    if args.stage in {"all", "filter"}:
        accurate, rejected, errors = filter_verification_results(
            results_path=paths.results,
            clean_path=paths.clean,
            rejected_path=paths.rejected,
            errors_path=paths.errors,
            model=args.model,
        )
        covered, remaining = write_remaining_subtools(
            clean_jsonl=paths.clean,
            tool_subtools_jsonl=args.tool_subtools_jsonl,
            output_path=paths.remaining_subtools,
        )
        print(f"[filtered] accurate: {accurate} -> {paths.clean}")
        print(f"[filtered] hallucinated: {rejected} -> {paths.rejected}")
        print(f"[filtered] parse/API/missing-usage errors: {errors} -> {paths.errors}")
        print(f"[coverage] covered sub-tools: {covered}")
        print(f"[coverage] remaining sub-tools: {remaining} -> {paths.remaining_subtools}")

    if args.merge_clean:
        count = merge_clean_files(args.merge_clean, paths.merged_clean)
        print(f"[merged] clean labels: {count} -> {paths.merged_clean}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
