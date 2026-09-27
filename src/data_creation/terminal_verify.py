#!/usr/bin/env python3
"""Execute model-verified commands in Kali and triage terminal evidence.

Run this stage only inside an isolated Kali container or disposable VM. The
input commands are data, but executing them has the same effect as typing them
into a shell. The CLI therefore requires an explicit acknowledgement flag.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, BinaryIO, Iterable

try:
    from .common import append_jsonl, iter_jsonl, load_latest_by_id, write_jsonl
except ImportError:  # Support direct execution.
    from common import append_jsonl, iter_jsonl, load_latest_by_id, write_jsonl  # type: ignore


DEFAULT_TIMEOUT_SEC = 20
DEFAULT_MAX_STREAM_BYTES = 200 * 1024

TOOL_ERROR_PATTERNS = (
    re.compile(r"\bcommand not found\b", re.IGNORECASE),
    re.compile(
        r"\bis not recognized as an internal or external command\b",
        re.IGNORECASE,
    ),
)

PARAMETER_ERROR_PATTERNS = (
    re.compile(r"\bunrecognized option\b", re.IGNORECASE),
    re.compile(r"\bunrecognised option\b", re.IGNORECASE),
    re.compile(r"\bunknown option\b", re.IGNORECASE),
    re.compile(r"\binvalid option\b", re.IGNORECASE),
    re.compile(r"\billegal option\b", re.IGNORECASE),
    re.compile(r"\bno such option\b", re.IGNORECASE),
    re.compile(r"\bunsupported option\b", re.IGNORECASE),
    re.compile(r"\bis not a supported option\b", re.IGNORECASE),
    re.compile(r"\bflag provided but not defined\b", re.IGNORECASE),
    re.compile(r"\bunrecognized arguments?\b", re.IGNORECASE),
    re.compile(r"\bunrecognised arguments?\b", re.IGNORECASE),
    re.compile(r"\bunexpected arguments?\b", re.IGNORECASE),
    re.compile(r"\bambiguous option\b", re.IGNORECASE),
    re.compile(r"\binvalid choice\b", re.IGNORECASE),
    re.compile(
        r"\boption\b.*\brequires (?:an? )?(?:argument|parameter)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bthe following arguments are required\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class TerminalPaths:
    out_dir: Path
    results: Path
    accepted: Path
    rejected: Path
    undecided: Path
    summary: Path


@dataclass
class StreamCapture:
    max_bytes: int
    total_bytes: int = 0
    truncated: bool = False

    def __post_init__(self) -> None:
        self._buffer = bytearray()

    def consume(self, stream: BinaryIO) -> None:
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                self.total_bytes += len(chunk)
                remaining = self.max_bytes - len(self._buffer)
                if remaining > 0:
                    self._buffer.extend(chunk[:remaining])
                if self.total_bytes > self.max_bytes:
                    self.truncated = True
        finally:
            stream.close()

    @property
    def text(self) -> str:
        return self._buffer.decode("utf-8", errors="replace")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def category_for_label(label: str) -> str:
    if label.startswith("REJECT_HALLUCINATED"):
        return "reject_hallucinated"
    if label in {"PASS_EXECUTED", "PASS_RUNTIME_ERROR", "TIMEOUT"}:
        return "pass_review"
    return "undecided"


def find_pattern(
    stderr: str,
    stdout: str,
    patterns: Iterable[re.Pattern[str]],
) -> tuple[str, str] | None:
    for stream_name, text in (("stderr", stderr), ("stdout", stdout)):
        for pattern in patterns:
            if pattern.search(text):
                return stream_name, pattern.pattern
    return None


def classify_execution(
    *,
    exit_code: int | None,
    timed_out: bool,
    stdout: str,
    stderr: str,
) -> dict[str, str | None]:
    """Classify whether runtime evidence indicates a hallucinated CLI."""
    if timed_out:
        label = "TIMEOUT"
        return {
            "label": label,
            "category": category_for_label(label),
            "reason": "Command exceeded the timeout and was terminated.",
            "matched_pattern": None,
            "matched_stream": None,
        }

    if exit_code is None:
        label = "UNDECIDED"
        return {
            "label": label,
            "category": category_for_label(label),
            "reason": "No exit code was available.",
            "matched_pattern": None,
            "matched_stream": None,
        }

    if exit_code == 127:
        label = "REJECT_HALLUCINATED_TOOL"
        return {
            "label": label,
            "category": category_for_label(label),
            "reason": "Exit code 127 indicates an unresolved command name.",
            "matched_pattern": "exit_code_127",
            "matched_stream": "process",
        }

    parameter_match = find_pattern(
        stderr,
        stdout,
        PARAMETER_ERROR_PATTERNS,
    )
    if parameter_match is not None:
        stream_name, pattern = parameter_match
        label = "REJECT_HALLUCINATED_PARAMETER"
        return {
            "label": label,
            "category": category_for_label(label),
            "reason": "The tool reported an unsupported or malformed option/argument.",
            "matched_pattern": pattern,
            "matched_stream": stream_name,
        }

    if exit_code != 0:
        tool_match = find_pattern(stderr, stdout, TOOL_ERROR_PATTERNS)
        if tool_match is not None:
            stream_name, pattern = tool_match
            label = "REJECT_HALLUCINATED_TOOL"
            return {
                "label": label,
                "category": category_for_label(label),
                "reason": "The shell reported that the requested tool was unavailable.",
                "matched_pattern": pattern,
                "matched_stream": stream_name,
            }

    if exit_code == 0:
        label = "PASS_EXECUTED"
        return {
            "label": label,
            "category": category_for_label(label),
            "reason": "Command executed successfully (exit code 0).",
            "matched_pattern": None,
            "matched_stream": None,
        }

    label = "PASS_RUNTIME_ERROR"
    return {
        "label": label,
        "category": category_for_label(label),
        "reason": (
            "Non-zero exit code without a tool/parameter hallucination indicator; "
            "the failure may depend on unavailable files, devices, services, or targets."
        ),
        "matched_pattern": None,
        "matched_stream": None,
    }


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute_command(
    command: str,
    *,
    shell: Path,
    timeout_sec: int,
    max_stream_bytes: int,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    started_clock = monotonic()
    timed_out = False
    exit_code: int | None = None
    stdout_capture = StreamCapture(max_stream_bytes)
    stderr_capture = StreamCapture(max_stream_bytes)

    process = subprocess.Popen(
        [str(shell), "-o", "pipefail", "-c", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_thread = threading.Thread(
        target=stdout_capture.consume,
        args=(process.stdout,),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stderr_capture.consume,
        args=(process.stderr,),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        exit_code = process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(process)
        exit_code = process.returncode
    finally:
        stdout_thread.join()
        stderr_thread.join()

    ended_at = utc_now_iso()
    duration_sec = monotonic() - started_clock
    classification = classify_execution(
        exit_code=exit_code,
        timed_out=timed_out,
        stdout=stdout_capture.text,
        stderr=stderr_capture.text,
    )
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": round(duration_sec, 6),
        "timeout_sec": timeout_sec,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stdout": stdout_capture.text,
        "stderr": stderr_capture.text,
        "stdout_bytes": stdout_capture.total_bytes,
        "stderr_bytes": stderr_capture.total_bytes,
        "stdout_truncated": stdout_capture.truncated,
        "stderr_truncated": stderr_capture.truncated,
        **classification,
    }


def undecided_verification(reason: str, timeout_sec: int) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "started_at": now,
        "ended_at": now,
        "duration_sec": 0.0,
        "timeout_sec": timeout_sec,
        "timed_out": False,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "label": "UNDECIDED",
        "category": "undecided",
        "reason": reason,
        "matched_pattern": None,
        "matched_stream": None,
    }


def run_terminal_verification(
    args: argparse.Namespace,
    paths: TerminalPaths,
) -> int:
    checkpoints = load_latest_by_id(paths.results)
    rows = list(iter_jsonl(args.input_jsonl))
    rows = rows[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]

    completed = 0
    for row_index, row in enumerate(rows, start=args.start_index):
        custom_id = str(row.get("custom_id") or f"row-{row_index}").strip()
        previous = checkpoints.get(custom_id)
        if previous is not None:
            category = str(
                (previous.get("terminal_verification") or {}).get("category")
                if isinstance(previous.get("terminal_verification"), dict)
                else ""
            )
            if not (args.retry_undecided and category == "undecided"):
                continue

        command = str(row.get("ground_truth_command") or "").strip()
        if not command:
            verification = undecided_verification(
                "Missing ground_truth_command.",
                args.timeout_sec,
            )
        else:
            try:
                verification = execute_command(
                    command,
                    shell=args.shell,
                    timeout_sec=args.timeout_sec,
                    max_stream_bytes=args.max_stream_bytes,
                )
            except Exception as exc:
                verification = undecided_verification(
                    f"Execution failed: {type(exc).__name__}: {exc}",
                    args.timeout_sec,
                )

        output = dict(row)
        output["custom_id"] = custom_id
        output["terminal_verification"] = verification
        append_jsonl(paths.results, output)
        checkpoints[custom_id] = output
        completed += 1
        print(
            f"[terminal] {custom_id}: "
            f"{verification['label']} ({verification['duration_sec']:.3f}s)"
        )
    return completed


def load_adjudications(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    allowed_categories = {
        "pass_review",
        "reject_hallucinated",
        "undecided",
    }
    adjudications: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        custom_id = str(row.get("custom_id") or "").strip()
        category = str(row.get("category") or "").strip()
        if not custom_id:
            raise ValueError(f"Adjudication row is missing custom_id: {row}")
        if category not in allowed_categories:
            raise ValueError(
                f"Invalid adjudication category {category!r} for {custom_id}; "
                f"expected one of {sorted(allowed_categories)}"
            )
        adjudications[custom_id] = {
            "category": category,
            "reason": str(row.get("reason") or "").strip(),
        }
    return adjudications


def apply_adjudication(
    row: dict[str, Any],
    adjudication: dict[str, Any] | None,
) -> dict[str, Any]:
    if adjudication is None:
        return row
    updated = dict(row)
    verification = (
        dict(row.get("terminal_verification"))
        if isinstance(row.get("terminal_verification"), dict)
        else {}
    )
    category = str(adjudication["category"])
    verification["category"] = category
    verification["ai_review_category"] = f"ai_{category}"
    verification["ai_review_reason"] = str(adjudication.get("reason") or "")
    updated["terminal_verification"] = verification
    return updated


def materialize_terminal_buckets(
    paths: TerminalPaths,
    *,
    adjudications_path: Path | None = None,
) -> dict[str, Any]:
    adjudications = load_adjudications(adjudications_path)
    buckets: dict[str, list[dict[str, Any]]] = {
        "pass_review": [],
        "reject_hallucinated": [],
        "undecided": [],
    }
    labels: Counter[str] = Counter()
    for row in load_latest_by_id(paths.results).values():
        custom_id = str(row.get("custom_id") or "")
        row = apply_adjudication(row, adjudications.get(custom_id))
        verification = row.get("terminal_verification")
        category = (
            str(verification.get("category") or "")
            if isinstance(verification, dict)
            else ""
        )
        if category not in buckets:
            category = "undecided"
        buckets[category].append(row)
        label = (
            str(verification.get("label") or "UNDECIDED")
            if isinstance(verification, dict)
            else "UNDECIDED"
        )
        labels[label] += 1

    write_jsonl(paths.accepted, buckets["pass_review"])
    write_jsonl(paths.rejected, buckets["reject_hallucinated"])
    write_jsonl(paths.undecided, buckets["undecided"])
    summary = {
        "total_rows": sum(len(rows) for rows in buckets.values()),
        "bucket_counts": {
            name: len(rows)
            for name, rows in buckets.items()
        },
        "label_counts": dict(sorted(labels.items())),
        "outputs": {
            "results": str(paths.results),
            "accepted": str(paths.accepted),
            "rejected": str(paths.rejected),
            "undecided": str(paths.undecided),
        },
        "adjudications": {
            "input": str(adjudications_path) if adjudications_path else None,
            "applied": len(adjudications),
        },
    }
    paths.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def resolve_paths(args: argparse.Namespace) -> TerminalPaths:
    out_dir = args.out_dir.resolve()
    return TerminalPaths(
        out_dir=out_dir,
        results=(
            args.results_jsonl or out_dir / "terminal_results.jsonl"
        ).resolve(),
        accepted=(
            args.accepted_jsonl or out_dir / "terminal_accepted.jsonl"
        ).resolve(),
        rejected=(
            args.rejected_jsonl or out_dir / "terminal_rejected.jsonl"
        ).resolve(),
        undecided=(
            args.undecided_jsonl or out_dir / "terminal_undecided.jsonl"
        ).resolve(),
        summary=(args.summary_json or out_dir / "terminal_summary.json").resolve(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute model-verified ground-truth commands in an isolated Kali "
            "environment and triage runtime evidence."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/data_creation/terminal"),
    )
    parser.add_argument("--results-jsonl", type=Path, default=None)
    parser.add_argument("--accepted-jsonl", type=Path, default=None)
    parser.add_argument("--rejected-jsonl", type=Path, default=None)
    parser.add_argument("--undecided-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--adjudications-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional reviewed decisions with custom_id, category "
            "(pass_review/reject_hallucinated/undecided), and reason."
        ),
    )
    parser.add_argument("--shell", type=Path, default=Path("/bin/bash"))
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument(
        "--max-stream-bytes",
        type=int,
        default=DEFAULT_MAX_STREAM_BYTES,
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-undecided", action="store_true")
    parser.add_argument(
        "--acknowledge-command-execution",
        action="store_true",
        help="Required acknowledgement that input commands will be executed.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.acknowledge_command_execution:
        raise SystemExit(
            "Refusing to execute commands without --acknowledge-command-execution. "
            "Run this stage only inside an isolated Kali container or disposable VM."
        )
    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be > 0")
    if args.max_stream_bytes < 0:
        raise SystemExit("--max-stream-bytes must be >= 0")
    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0")
    if not args.shell.exists():
        raise SystemExit(f"Shell not found: {args.shell}")


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    args.input_jsonl = args.input_jsonl.resolve()
    if args.adjudications_jsonl is not None:
        args.adjudications_jsonl = args.adjudications_jsonl.resolve()
    paths = resolve_paths(args)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    completed = run_terminal_verification(args, paths)
    summary = materialize_terminal_buckets(
        paths,
        adjudications_path=args.adjudications_jsonl,
    )
    print(f"[terminal] new checkpoints: {completed} -> {paths.results}")
    print(
        "[terminal] "
        f"accepted={summary['bucket_counts']['pass_review']} "
        f"rejected={summary['bucket_counts']['reject_hallucinated']} "
        f"undecided={summary['bucket_counts']['undecided']}"
    )
    print(f"[terminal] summary -> {paths.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
