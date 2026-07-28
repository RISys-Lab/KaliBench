#!/usr/bin/env python3
"""I/O helpers for JSONL and lightweight parsing utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding=encoding) as handle:
        for lineno, line in enumerate(handle, start=1):
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object in {path}:{lineno}, got {type(obj)}")
            records.append(obj)
    return records


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_possible_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value
