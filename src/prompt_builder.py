#!/usr/bin/env python3
"""Prompt construction logic for KaliBench inference modes."""

import argparse
import random
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from io_utils import parse_possible_json_field
from prompt import (
    SYSTEM_PROMPT_RESTRICTED_THINKING,
    SYSTEM_PROMPT_RESTRICTED_FINAL_LINE,
    SYSTEM_PROMPT_RESTRICTED,
    SYSTEM_PROMPT_UNRESTRICTED_THINKING,
    SYSTEM_PROMPT_UNRESTRICTED_FINAL_LINE,
    SYSTEM_PROMPT_UNRESTRICTED,
    USER_PROMPT_HINTED_THINKING_TEMPLATE,
    USER_PROMPT_HINTED_FINAL_LINE_TEMPLATE,
    USER_PROMPT_HINTED_TEMPLATE,
    USER_PROMPT_RESTRICTED_THINKING_TEMPLATE,
    USER_PROMPT_RESTRICTED_FINAL_LINE_TEMPLATE,
    USER_PROMPT_RESTRICTED_TEMPLATE,
    USER_PROMPT_UNRESTRICTED_THINKING_TEMPLATE,
    USER_PROMPT_UNRESTRICTED_FINAL_LINE_TEMPLATE,
    USER_PROMPT_UNRESTRICTED_TEMPLATE,
)


class PromptRow(NamedTuple):
    custom_id: str
    query: str
    tool_name: str
    messages: List[Dict[str, str]]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"yes", "true", "t", "1", "y"}:
        return True
    if s in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def chunk_list(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def load_tool_usages_jsonl(records: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    tool_map: Dict[str, str] = {}
    for rec in records:
        sub = rec.get("subtool") or rec.get("tool") or rec.get("title") or rec.get("name")
        usage = rec.get("usage_code") or rec.get("manuscript") or rec.get("usage") or rec.get("description")
        if not sub:
            continue
        tool_map[str(sub).strip().lower()] = str(usage or "")
    return tool_map


def make_allowed_str_with_usage(
    tools: Sequence[str],
    usage_for_tools: Dict[str, str],
    include_usage: bool,
    max_usage_tokens: Optional[int],
    tokenizer: Optional[Any] = None,
) -> str:
    def truncate_by_tokens(text: str, max_tokens: int) -> str:
        if tokenizer is None:
            tokens = text.split()
            if len(tokens) <= max_tokens:
                return text
            return " ".join(tokens[:max_tokens]).rstrip() + "\n...(truncated)"

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        clipped_ids = token_ids[:max_tokens]
        clipped_text = tokenizer.decode(clipped_ids, skip_special_tokens=True)
        return clipped_text.rstrip() + "\n...(truncated)"

    lines: List[str] = []
    for tool_name in tools:
        lines.append(f"- {tool_name}")
        if not include_usage:
            continue
        usage = usage_for_tools.get(tool_name, "")
        if not usage:
            continue
        usage_snippet = usage
        if max_usage_tokens is not None and max_usage_tokens > 0:
            usage_snippet = truncate_by_tokens(usage, max_usage_tokens)
        for usage_line in usage_snippet.strip().splitlines():
            lines.append(f"  {usage_line}")
    return "\n".join(lines)


def extract_query(row: Dict[str, Any]) -> str:
    candidates = [row.get("query"), row.get("prompt")]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_prompts_and_labels(
    records: Sequence[Dict[str, Any]],
    mode: str,
    candidate_tools: int,
    tool_usage_map: Dict[str, str],
    include_usage_in_prompt: bool,
    max_usage_tokens: int,
    candidate_seed: int = 42,
    output_policy: str = "strict",
    tokenizer: Optional[Any] = None,
) -> Tuple[List[PromptRow], List[Dict[str, Any]]]:
    if output_policy not in {"strict", "final-line", "thinking"}:
        raise ValueError("output_policy must be one of: strict, final-line, thinking")

    if output_policy == "strict":
        unrestricted_system_prompt = SYSTEM_PROMPT_UNRESTRICTED
        restricted_system_prompt = SYSTEM_PROMPT_RESTRICTED
        unrestricted_user_template = USER_PROMPT_UNRESTRICTED_TEMPLATE
        restricted_user_template = USER_PROMPT_RESTRICTED_TEMPLATE
        hinted_user_template = USER_PROMPT_HINTED_TEMPLATE
    elif output_policy == "final-line":
        unrestricted_system_prompt = SYSTEM_PROMPT_UNRESTRICTED_FINAL_LINE
        restricted_system_prompt = SYSTEM_PROMPT_RESTRICTED_FINAL_LINE
        unrestricted_user_template = USER_PROMPT_UNRESTRICTED_FINAL_LINE_TEMPLATE
        restricted_user_template = USER_PROMPT_RESTRICTED_FINAL_LINE_TEMPLATE
        hinted_user_template = USER_PROMPT_HINTED_FINAL_LINE_TEMPLATE
    else:
        unrestricted_system_prompt = SYSTEM_PROMPT_UNRESTRICTED_THINKING
        restricted_system_prompt = SYSTEM_PROMPT_RESTRICTED_THINKING
        unrestricted_user_template = USER_PROMPT_UNRESTRICTED_THINKING_TEMPLATE
        restricted_user_template = USER_PROMPT_RESTRICTED_THINKING_TEMPLATE
        hinted_user_template = USER_PROMPT_HINTED_THINKING_TEMPLATE

    if mode == "unrestricted":

        prompts: List[PromptRow] = []
        labels: List[Dict[str, Any]] = []
        for idx, row in enumerate(records, start=1):
            custom_id = str(row.get("custom_id") or f"row-{idx}").strip()
            query = extract_query(row)
            messages = [
                {"role": "system", "content": unrestricted_system_prompt},
                {
                    "role": "user",
                    "content": unrestricted_user_template.format(query=query),
                },
            ]
            prompts.append(
                PromptRow(
                    custom_id=custom_id,
                    query=query,
                    tool_name=str(row.get("tool_name") or row.get("tool") or ""),
                    messages=messages,
                )
            )
            labels.append(
                {
                    "custom_id": custom_id,
                    "query": query,
                    "tool_name": str(row.get("tool_name") or row.get("tool") or ""),
                    "ground_truth_command": str(row.get("ground_truth_command") or ""),
                    "optional_args": parse_possible_json_field(row.get("optional_args"), []),
                    "positional_args": parse_possible_json_field(row.get("positional_args"), []),
                    "option_kv": parse_possible_json_field(row.get("option_kv"), {}),
                }
            )
        return prompts, labels

    if candidate_tools <= 0:
        raise ValueError("candidate_tools must be > 0")

    tools = sorted(
        {
            str(r.get("tool_name") or r.get("tool") or "").strip()
            for r in records
            if str(r.get("tool_name") or r.get("tool") or "").strip()
        }
    )
    if not tools:
        raise ValueError("No tools found in input records; cannot build restricted/hinted candidates.")

    tools_lower = [t.strip().lower() for t in tools]
    rng = random.Random(candidate_seed)

    prompts = []
    labels = []
    for idx, row in enumerate(records, start=1):
        custom_id = str(row.get("custom_id") or f"row-{idx}").strip()
        query = extract_query(row)
        tool_name = str(row.get("tool_name") or row.get("tool") or "").strip()
        target_tool_lower = tool_name.lower()
        candidate_count = min(candidate_tools, len(tools))

        chosen_chunk: List[str]
        if candidate_count == 0:
            chosen_chunk = []
        elif target_tool_lower in tools_lower:
            non_target_tools = [t for t in tools if t.strip().lower() != target_tool_lower]
            sampled = rng.sample(non_target_tools, k=max(candidate_count - 1, 0))
            chosen_chunk = sampled + [next(t for t in tools if t.strip().lower() == target_tool_lower)]
            rng.shuffle(chosen_chunk)
        else:
            raise ValueError(
                f"Gold tool '{tool_name}' for custom_id '{custom_id}' is missing from the tool pool. "
                "This is unexpected and indicates inconsistent input data."
            )

        usage_map = {t: tool_usage_map.get(t.strip().lower(), "") for t in chosen_chunk}
        include_usage = mode == "hinted" and include_usage_in_prompt and max_usage_tokens != 0
        allowed_str = make_allowed_str_with_usage(
            chosen_chunk,
            usage_map,
            include_usage=include_usage,
            max_usage_tokens=max_usage_tokens if max_usage_tokens > 0 else None,
            tokenizer=tokenizer,
        )

        if mode == "restricted":
            user_prompt = restricted_user_template.format(allowed_tools=allowed_str, query=query)
        else:
            user_prompt = hinted_user_template.format(allowed_tools=allowed_str, query=query)

        messages = [
            {"role": "system", "content": restricted_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompts.append(PromptRow(custom_id=custom_id, query=query, tool_name=tool_name, messages=messages))
        labels.append(
            {
                "custom_id": custom_id,
                "query": query,
                "tool_name": tool_name,
                "ground_truth_command": str(row.get("ground_truth_command") or ""),
                "optional_args": parse_possible_json_field(row.get("optional_args"), []),
                "positional_args": parse_possible_json_field(row.get("positional_args"), []),
                "option_kv": parse_possible_json_field(row.get("option_kv"), {}),
            }
        )

    return prompts, labels
