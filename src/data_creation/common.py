#!/usr/bin/env python3
"""Shared helpers for the KaliBench data-construction stages."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a UTF-8 JSONL file with useful error locations."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}, "
                    f"got {type(row).__name__}"
                )
            yield row


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write JSON objects to a JSONL file and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, row: Mapping[str, Any], *, sync: bool = True) -> None:
    """Append one checkpoint row, optionally forcing it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        if sync:
            os.fsync(handle.fileno())


def load_latest_by_id(
    path: Path,
    *,
    id_field: str = "custom_id",
) -> dict[str, dict[str, Any]]:
    """Load the last checkpoint row for each non-empty ID."""
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        row_id = str(row.get(id_field) or "").strip()
        if row_id:
            rows[row_id] = row
    return rows


def clean_json_text(text: str) -> str:
    """Remove common prose/fence wrappers around model-produced JSON."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(
        r"(?i)^here\s+is\s+the\s+json\s*[:：]?\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)^json\s*[:：]?\s*", "", cleaned)
    return cleaned.strip()


def parse_json_container(text: str, expected_type: type) -> Any:
    """Parse a model response as a JSON object/array, tolerating outer prose."""
    cleaned = clean_json_text(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        opening, closing = ("[", "]") if expected_type is list else ("{", "}")
        start = cleaned.find(opening)
        end = cleaned.rfind(closing)
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, expected_type):
        raise ValueError(
            f"Expected model response to contain {expected_type.__name__}, "
            f"got {type(parsed).__name__}"
        )
    return parsed


def build_openai_client(
    *,
    api_key: str | None,
    api_key_env: str,
    base_url: str,
) -> Any:
    """Build an OpenAI-compatible client without logging its credential."""
    from openai import OpenAI

    resolved_key = api_key or os.environ.get(api_key_env)
    if not resolved_key:
        raise SystemExit(
            f"Missing API key. Set ${api_key_env} or pass --api-key."
        )
    return OpenAI(api_key=resolved_key, base_url=base_url)


def request_chat_completion(
    client: Any,
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint and return text."""
    response = client.chat.completions.create(
        model=model,
        messages=[dict(message) for message in messages],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return str(response.choices[0].message.content or "").strip()
