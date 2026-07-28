from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data_creation.common import iter_jsonl, write_jsonl
from src.data_creation.finalize import (
    load_and_filter_candidates,
    stratified_split,
)
from src.data_creation.generate import (
    build_tool_subtools,
    parse_generated_array,
)
from src.data_creation.terminal_verify import (
    apply_adjudication,
    classify_execution,
    execute_command,
)
from src.data_creation.verify import (
    filter_verification_results,
    find_usage_code,
    load_usage_maps,
    parse_verifier_output,
)


def accurate_model_verification() -> dict:
    return {
        "model": "qwen3-max",
        "model_output": {
            "verdict": "ACCURATE",
            "reasons": "Supported by the documentation.",
        },
    }


def accepted_terminal_verification() -> dict:
    return {
        "label": "PASS_EXECUTED",
        "category": "pass_review",
        "reason": "Command executed successfully.",
    }


class GenerationTests(unittest.TestCase):
    def test_document_chunking_and_exact_deduplication(self) -> None:
        content = (
            "root@kali:~# alpha --help\nalpha usage\n"
            "root@kali:~# beta -h\nbeta usage\n"
        )
        rows = build_tool_subtools(
            [
                {"title": "demo", "content": content},
                {"title": "demo", "content": content},
            ]
        )
        self.assertEqual([row["subtool"] for row in rows], ["alpha", "beta"])
        self.assertEqual(rows[0]["source_row"], 0)

    def test_generation_response_parser_accepts_fenced_array(self) -> None:
        rows = parse_generated_array(
            '```json\n[{"query":"q","ground_truth_command":"echo ok"}]\n```'
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "q")


class ModelVerificationTests(unittest.TestCase):
    def test_usage_lookup_and_verifier_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage.jsonl"
            write_jsonl(
                usage_path,
                [
                    {
                        "title": "Demo",
                        "subtool": "Echo",
                        "usage_code": "echo [STRING]",
                    }
                ],
            )
            subtools, titles = load_usage_maps(usage_path)
            usage = find_usage_code(
                {"tool_name": "echo"},
                subtools,
                titles,
            )
            self.assertEqual(usage, "echo [STRING]")

        parsed = parse_verifier_output(
            'prefix {"verdict":"accurate","reasons":"documented"} suffix'
        )
        self.assertEqual(parsed["verdict"], "ACCURATE")

    def test_filter_writes_released_model_verification_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results.jsonl"
            clean = root / "clean.jsonl"
            rejected = root / "rejected.jsonl"
            errors = root / "errors.jsonl"
            write_jsonl(
                results,
                [
                    {
                        "custom_id": "candidate-1",
                        "query": "Print ok.",
                        "tool_name": "echo",
                        "ground_truth_command": "echo ok",
                        "optional_args": {},
                        "positional_args": ["ok"],
                        "model_verification": accurate_model_verification(),
                    }
                ],
            )
            counts = filter_verification_results(
                results_path=results,
                clean_path=clean,
                rejected_path=rejected,
                errors_path=errors,
                model="qwen3-max",
            )
            self.assertEqual(counts, (1, 0, 0))
            row = next(iter_jsonl(clean))
            self.assertEqual(
                row["model_verification"]["model_output"]["verdict"],
                "ACCURATE",
            )


class TerminalVerificationTests(unittest.TestCase):
    def test_classifier_labels_tool_parameter_runtime_and_timeout(self) -> None:
        tool = classify_execution(
            exit_code=127,
            timed_out=False,
            stdout="",
            stderr="missing: command not found",
        )
        parameter = classify_execution(
            exit_code=2,
            timed_out=False,
            stdout="",
            stderr="demo: unrecognized option '--invented'",
        )
        runtime = classify_execution(
            exit_code=1,
            timed_out=False,
            stdout="",
            stderr="cat: /missing: No such file or directory",
        )
        timeout = classify_execution(
            exit_code=-15,
            timed_out=True,
            stdout="",
            stderr="",
        )
        self.assertEqual(tool["label"], "REJECT_HALLUCINATED_TOOL")
        self.assertEqual(parameter["label"], "REJECT_HALLUCINATED_PARAMETER")
        self.assertEqual(runtime["label"], "PASS_RUNTIME_ERROR")
        self.assertEqual(timeout["label"], "TIMEOUT")

    def test_bounded_command_execution(self) -> None:
        result = execute_command(
            "printf 'ok'",
            shell=Path("/bin/bash"),
            timeout_sec=2,
            max_stream_bytes=1024,
        )
        self.assertEqual(result["label"], "PASS_EXECUTED")
        self.assertEqual(result["stdout"], "ok")

    def test_review_adjudication_is_recorded_in_release_metadata(self) -> None:
        row = {
            "custom_id": "candidate-1",
            "terminal_verification": {
                "label": "PASS_RUNTIME_ERROR",
                "category": "pass_review",
            },
        }
        updated = apply_adjudication(
            row,
            {
                "category": "reject_hallucinated",
                "reason": "Manual inspection found an unsupported capability.",
            },
        )
        verification = updated["terminal_verification"]
        self.assertEqual(verification["category"], "reject_hallucinated")
        self.assertEqual(
            verification["ai_review_category"],
            "ai_reject_hallucinated",
        )


class FinalizationTests(unittest.TestCase):
    def test_filter_deduplicate_and_stratify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "accepted.jsonl"
            rows = []
            for tool_index, tool in enumerate(("alpha", "beta", "gamma")):
                for item_index in range(3):
                    rows.append(
                        {
                            "custom_id": f"source-{tool_index}-{item_index}",
                            "query": f"Run {tool} example {item_index}.",
                            "tool_name": tool,
                            "ground_truth_command": f"{tool} {item_index}",
                            "optional_args": {},
                            "positional_args": [str(item_index)],
                            "model_verification": accurate_model_verification(),
                            "terminal_verification": accepted_terminal_verification(),
                        }
                    )
            duplicate = dict(rows[0])
            duplicate["custom_id"] = "duplicate"
            duplicate["query"] = "  RUN alpha  example 0. "
            rows.append(duplicate)
            write_jsonl(input_path, rows)

            accepted, rejected, rejection_counts = load_and_filter_candidates(
                [input_path],
                require_model_verification=True,
                require_terminal_verification=True,
                id_prefix="sample_",
                keep_provenance=False,
            )
            train, test = stratified_split(
                accepted,
                test_size=6,
                min_test_per_tool=1,
                max_test_per_tool=2,
                seed=42,
            )

            self.assertEqual(len(accepted), 9)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejection_counts["duplicate_normalized_query"], 1)
            self.assertEqual(len(train), 3)
            self.assertEqual(len(test), 6)
            self.assertEqual(
                {row["tool_name"] for row in test},
                {"alpha", "beta", "gamma"},
            )
            self.assertEqual(
                [row["custom_id"] for row in accepted],
                [f"sample_{index}" for index in range(9)],
            )


if __name__ == "__main__":
    unittest.main()
