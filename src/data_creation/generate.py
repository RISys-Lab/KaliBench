#!/usr/bin/env python3
"""Create KaliBench query-command candidates from Kali tool documentation.

This is the refactored form of ``legacy_data_creation/generate_qwen3max.py``.
It separates preparation, API generation, and response flattening so every
stage can be audited or resumed independently.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

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
except ImportError:  # Support direct execution: python src/data_creation/generate.py
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
DEFAULT_SOURCE_DATASET = "anonymous62567/kali-tools"

PROMPT_RE = re.compile(r"^root@kali:[^\n]*#\s", re.MULTILINE)
SUBTOOL_RE = re.compile(r"^root@kali:[^\n]*#\s*([^\s;&|]+)", re.MULTILINE)

GENERATION_SYSTEM_PROMPT = (
    "You are an expert dataset generator for Kali/Linux tools. "
    "Follow all user rules exactly. Generate diverse yet unambiguous "
    "natural-language queries. Use only options from the seed documentation. "
    "Output strictly valid JSON arrays and no prose."
)

GENERATION_PROMPT_TEMPLATE = r"""
You are generating a dataset. Follow every instruction exactly.

Goal:
Convert the provided seed documentation about Kali/Linux tools into a JSON array.
Each item must contain:
- query
- ground_truth_command
- tool_name
- optional_args
- positional_args

Inputs:
- tool_name: {{TOOL_FQN}}
- number_of_items: {{NUM_ITEMS}}
- difficulty_mix: {{DIFFICULTY_DISTRIBUTION}}
- seed_documentation:
<usage_code>
{{SEED_DATA}}
</usage_code>

Hard constraints:
1. Use only tools, flags, aliases, arguments, and behavior explicitly documented in seed_documentation.
2. Do not invent flags, pipelines, subshells, or environment-dependent defaults.
3. The query must uniquely determine exactly one valid command string.
4. All required parameters must be explicit in the query.
5. Avoid ambiguous words such as quick, fast, recent, typical, default, simple scan, or verbose output without a level.
6. If multiple tools or aliases could apply, mention the intended tool or alias explicitly.
7. If aliases exist, represent them in optional_args as a merged alias key such as "--help|-h".
8. In the command line, use one canonical form consistently, preferably the long flag if available.

Canonical command formatting:
- command order: tool_name -> long boolean flags -> short boolean flags -> flags with values -> positional_args
- flag values are space-separated, for example "--output out.txt" or "-p 80"
- positional_args must be ordered and must not contain flags

Output format:
Return only a JSON array. No markdown, no comments, no explanation.
Each item must follow this schema:
{
  "query": "string",
  "ground_truth_command": "string",
  "tool_name": "string",
  "optional_args": { "flag[|alias]": "string or null" },
  "positional_args": ["arg1", "arg2"]
}

Self-check each item before output:
- every flag exists in seed_documentation
- query fixes all parameters needed to avoid multiple valid commands
- ground_truth_command matches the query
- optional_args and positional_args parse the command correctly
- exactly one command is a correct answer
""".strip()


@dataclass(frozen=True)
class GenerationPaths:
    out_dir: Path
    tool_subtools: Path
    requests: Path
    raw_responses: Path
    labels: Path
    parse_errors: Path


def split_into_subtools(text: str) -> list[tuple[str, str]]:
    """Split one Kali manuscript into shell-prompt-delimited usage chunks."""
    normalized = str(text or "").replace("\r\n", "\n")
    matches = list(PROMPT_RE.finditer(normalized))
    if not matches:
        return []

    segments: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        segment = normalized[match.start() : end].strip()
        subtool_match = SUBTOOL_RE.search(segment)
        if segment and subtool_match:
            segments.append((subtool_match.group(1).strip(), segment))
    return segments


def iter_source_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    """Read source manuscripts locally or from the anonymous HF dataset."""
    if args.source_jsonl is not None:
        yield from iter_jsonl(args.source_jsonl)
        return

    from datasets import load_dataset

    dataset = load_dataset(args.source_dataset, split=args.source_split)
    for row in dataset:
        if isinstance(row, dict):
            yield row
        else:
            yield dict(row)


def build_tool_subtools(
    source_rows: Iterable[dict[str, Any]],
    *,
    title_field: str = "title",
    content_field: str = "content",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Extract and exactly deduplicate sub-tool usage chunks."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for source_row, row in enumerate(source_rows):
        title = str(row.get(title_field, "") or "").strip()
        content = str(row.get(content_field, "") or "")
        for subtool, usage_code in split_into_subtools(content):
            candidate = {
                "source_row": source_row,
                "subtool": subtool,
                "title": title,
                "usage_code": usage_code,
            }
            key = (title, subtool, usage_code)
            if usage_code and key not in seen:
                seen.add(key)
                rows.append(candidate)
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def load_or_build_tool_subtools(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_tool_subtools_jsonl is not None:
        rows = list(iter_jsonl(args.input_tool_subtools_jsonl))
        return rows[: args.limit] if args.limit is not None else rows
    return build_tool_subtools(
        iter_source_rows(args),
        title_field=args.title_field,
        content_field=args.content_field,
        limit=args.limit,
    )


def make_generation_prompt(row: dict[str, Any], items_per_tool: int) -> str:
    tool_fqn = f"{row.get('title', '')}/{row.get('subtool', '')}".strip("/")
    return (
        GENERATION_PROMPT_TEMPLATE.replace("{{TOOL_FQN}}", tool_fqn)
        .replace("{{SEED_DATA}}", str(row.get("usage_code", "")))
        .replace("{{NUM_ITEMS}}", str(items_per_tool))
        .replace(
            "{{DIFFICULTY_DISTRIBUTION}}",
            "easy:40%, medium:40%, hard:20%",
        )
    )


def make_generation_request(
    row: dict[str, Any],
    index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "custom_id": f"generation-{index:06d}",
        "source_title": row.get("title", ""),
        "source_subtool": row.get("subtool", ""),
        "model": args.model,
        "messages": [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": make_generation_prompt(row, args.items_per_tool),
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }


def prepare_generation(args: argparse.Namespace, paths: GenerationPaths) -> tuple[int, int]:
    subtools = load_or_build_tool_subtools(args)
    subtool_count = write_jsonl(paths.tool_subtools, subtools)
    requests = [
        make_generation_request(row, index, args)
        for index, row in enumerate(subtools)
    ]
    request_count = write_jsonl(paths.requests, requests)
    return subtool_count, request_count


def should_retry_checkpoint(row: dict[str, Any], retry_errors: bool) -> bool:
    return bool(retry_errors and str(row.get("error") or "").strip())


def run_generation_api(args: argparse.Namespace, paths: GenerationPaths) -> int:
    from tqdm import tqdm

    if not paths.requests.exists():
        raise FileNotFoundError(
            f"Generation request file not found: {paths.requests}. "
            "Run --stage prepare first."
        )
    client = build_openai_client(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
    )
    checkpoints = load_latest_by_id(paths.raw_responses)
    requests = list(iter_jsonl(paths.requests))
    if args.limit is not None:
        requests = requests[: args.limit]

    completed = 0
    for request in tqdm(requests, desc=f"{args.model} generation"):
        request_id = str(request.get("custom_id") or "")
        previous = checkpoints.get(request_id)
        if previous is not None and not should_retry_checkpoint(previous, args.retry_errors):
            continue

        body = {
            "model": request.get("model") or args.model,
            "messages": request.get("messages") or [],
            "temperature": request.get("temperature", args.temperature),
            "max_tokens": request.get("max_tokens", args.max_tokens),
        }
        try:
            content = request_chat_completion(client, **body)
            error = ""
        except Exception as exc:  # Preserve failures for audit and optional retry.
            content = ""
            error = f"{type(exc).__name__}: {exc}"

        checkpoint = {
            "custom_id": request_id,
            "source_title": request.get("source_title"),
            "source_subtool": request.get("source_subtool"),
            "request": body,
            "response": content,
            "error": error,
        }
        append_jsonl(paths.raw_responses, checkpoint)
        checkpoints[request_id] = checkpoint
        completed += 1
        if args.sleep > 0:
            time.sleep(args.sleep)
    return completed


def parse_generated_array(text: str) -> list[dict[str, Any]]:
    parsed = parse_json_container(text, list)
    return [row for row in parsed if isinstance(row, dict)]


def normalize_label_item(
    item: dict[str, Any],
    *,
    custom_id: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    optional_args = item.get("optional_args")
    positional_args = item.get("positional_args")
    return {
        "custom_id": custom_id,
        "query": str(item.get("query") or "").strip(),
        "tool_name": str(item.get("tool_name") or "").strip(),
        "ground_truth_command": str(
            item.get("ground_truth_command") or ""
        ).strip(),
        "optional_args": optional_args if isinstance(optional_args, dict) else {},
        "positional_args": (
            [str(value) for value in positional_args]
            if isinstance(positional_args, list)
            else []
        ),
        "source_request_id": source.get("custom_id"),
        "source_title": source.get("source_title"),
        "source_subtool": source.get("source_subtool"),
    }


def flatten_responses(paths: GenerationPaths) -> tuple[int, int]:
    if not paths.raw_responses.exists():
        raise FileNotFoundError(
            f"Generation response file not found: {paths.raw_responses}. "
            "Run --stage generate first."
        )
    labels: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    checkpoints = load_latest_by_id(paths.raw_responses)

    for source_index, response_row in enumerate(checkpoints.values()):
        if str(response_row.get("error") or "").strip():
            continue
        try:
            items = parse_generated_array(str(response_row.get("response") or ""))
        except Exception as exc:
            errors.append(
                {
                    "custom_id": response_row.get("custom_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "response_excerpt": str(
                        response_row.get("response") or ""
                    )[:1000],
                }
            )
            continue

        for item_index, item in enumerate(items):
            label = normalize_label_item(
                item,
                custom_id=f"candidate-{source_index:06d}-{item_index:03d}",
                source=response_row,
            )
            if (
                label["query"]
                and label["tool_name"]
                and label["ground_truth_command"]
            ):
                labels.append(label)

    label_count = write_jsonl(paths.labels, labels)
    error_count = write_jsonl(paths.parse_errors, errors)
    return label_count, error_count


def resolve_paths(args: argparse.Namespace) -> GenerationPaths:
    out_dir = args.out_dir.resolve()
    return GenerationPaths(
        out_dir=out_dir,
        tool_subtools=(
            args.tool_subtools_jsonl or out_dir / "tool_subtools.jsonl"
        ).resolve(),
        requests=(
            args.requests_jsonl or out_dir / "generation_requests.jsonl"
        ).resolve(),
        raw_responses=(
            args.raw_responses or out_dir / "generation_responses.jsonl"
        ).resolve(),
        labels=(
            args.labels_jsonl or out_dir / "generated_labels.jsonl"
        ).resolve(),
        parse_errors=(
            args.parse_errors_jsonl or out_dir / "generation_parse_errors.jsonl"
        ).resolve(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Kali tool documentation, prepare generation requests, call "
            "an OpenAI-compatible model, and flatten query-command candidates."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "generate", "flatten"],
        default="all",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/data_creation/generation"),
    )
    parser.add_argument(
        "--source-dataset",
        default=DEFAULT_SOURCE_DATASET,
        help="Hugging Face source dataset used when --source-jsonl is omitted.",
    )
    parser.add_argument("--source-split", default="train")
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=None,
        help="Optional local JSONL with title/content fields for offline preparation.",
    )
    parser.add_argument("--title-field", default="title")
    parser.add_argument("--content-field", default="content")
    parser.add_argument(
        "--input-tool-subtools-jsonl",
        type=Path,
        default=None,
        help="Optional pre-extracted usage chunks, including remaining chunks from verification.",
    )
    parser.add_argument("--tool-subtools-jsonl", type=Path, default=None)
    parser.add_argument("--requests-jsonl", type=Path, default=None)
    parser.add_argument("--raw-responses", type=Path, default=None)
    parser.add_argument("--labels-jsonl", type=Path, default=None)
    parser.add_argument("--parse-errors-jsonl", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DASHSCOPE_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--items-per-tool", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry IDs whose latest checkpoint row contains an API error.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    paths = resolve_paths(args)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "prepare"}:
        subtool_count, request_count = prepare_generation(args, paths)
        print(f"[prepared] tool/subtool chunks: {subtool_count} -> {paths.tool_subtools}")
        print(f"[prepared] generation requests: {request_count} -> {paths.requests}")

    if args.stage in {"all", "generate"}:
        completed = run_generation_api(args, paths)
        print(f"[generated] new API checkpoints: {completed} -> {paths.raw_responses}")

    if args.stage in {"all", "flatten"}:
        label_count, error_count = flatten_responses(paths)
        print(f"[flattened] labels: {label_count} -> {paths.labels}")
        print(f"[flattened] parse errors: {error_count} -> {paths.parse_errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
