#!/usr/bin/env python3
"""Checkpoint and resume helpers for evaluation prediction JSONL files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from prompt_builder import PromptRow


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Append JSONL rows to path, writing one object per line."""
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_prefix(path: Path, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """Read the longest valid JSONL prefix from path.

    If a run is interrupted during write, the final line may be truncated.
    This helper keeps valid records up to the first malformed line.
    """
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding=encoding) as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                print(f"[WARN] Truncated JSON at {path}:{lineno}; using valid prefix for resume.")
                break
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object in {path}:{lineno}, got {type(obj)}")
            records.append(obj)
    return records


def extract_compat_prediction(row: Dict[str, Any]) -> str:
    """Extract content text from scorer-compatible prediction row."""
    response = row.get("response")
    if not isinstance(response, dict):
        return ""
    body = response.get("body")
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def build_prediction_rows(
    prompt_rows: Sequence[PromptRow],
    predictions: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build raw and scorer-compatible JSONL rows from prompts and predictions."""
    if len(prompt_rows) != len(predictions):
        raise ValueError("prompt/prediction length mismatch while building output rows")

    raw_rows: List[Dict[str, Any]] = []
    compat_rows: List[Dict[str, Any]] = []
    for prompt_row, pred in zip(prompt_rows, predictions):
        raw_rows.append(
            {
                "custom_id": prompt_row.custom_id,
                "query": prompt_row.query,
                "tool_name": prompt_row.tool_name,
                "model_reply": pred,
            }
        )
        compat_rows.append(
            {
                "custom_id": prompt_row.custom_id,
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": pred,
                                }
                            }
                        ]
                    }
                },
            }
        )
    return raw_rows, compat_rows


def validate_resume_rows(
    rows: Sequence[Dict[str, Any]],
    prompts: Sequence[PromptRow],
    source_name: str,
) -> None:
    """Validate existing checkpoint rows can be safely resumed against prompts."""
    if len(rows) > len(prompts):
        raise ValueError(
            f"{source_name} has {len(rows)} rows but current run has only {len(prompts)} prompts. "
            "Use --resume false or a new output directory."
        )

    for idx, row in enumerate(rows):
        expected_id = str(prompts[idx].custom_id)
        actual_id = str(row.get("custom_id") or "")
        if actual_id != expected_id:
            raise ValueError(
                f"{source_name} custom_id mismatch at index {idx}: "
                f"expected '{expected_id}', got '{actual_id}'. "
                "Use --resume false or a new output directory."
            )
