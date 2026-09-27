#!/usr/bin/env python3
"""
model_score.py

usage example:

python model_score.py --vllm_file "./unres_llama_8b-out.jsonl" --label_file "./labels.jsonl" --out "./unres_llama_8b-scores.jsonl"



"""
#!/usr/bin/env python3
# score_cmd_strict.py

import json
import re
import shlex
import argparse
from collections import Counter

FORMAT_ERROR_TOKEN = "[FORMAT_ERROR]"

# =================================================
# Utility: Extract model output
# =================================================
def extract_command_from_output(text):
    """Extract content inside the last <output>...</output>. If not found, return empty string (严格与原代码一致)."""
    # Manually trim texts before </think> if present, to handle cases where model includes reasoning before final answer.
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # Then extract the last <output>...</output> block
    matches = re.findall(r"<output>\s*(.*?)\s*</output>", text, re.S)

    if not matches: # Foundsec instruct output
        # Retry with the last ```output ... ``` block if no <output> tags found
        matches = re.findall(r"```output\s*(.*?)\s*```", text, re.S)

    if "</output>" in text and "<output>" not in text:
        output_block = text.split("</output>")[0].strip()

    elif not matches:
        return FORMAT_ERROR_TOKEN

    else:
        output_block = matches[-1].strip()

    # Remove [command] if present in the output block
    output_block = output_block.replace("[command]", "").strip()
    # If the output block starts with "[" and ends with "]", then remove the brackets
    if output_block.startswith("[") and output_block.endswith("]"):
        output_block = output_block[1:-1].strip()
    return output_block

def parse_optional_args(raw):
    """
    Normalize optional_args into a dict:
        {"--start|-s": None, ... }
    Handles cases:
        - dict
        - list
        - string containing JSON
        - empty or null
    (严格按你原代码逻辑)
    """
    # None or empty ⇒ return empty dict
    if raw is None:
        return {}
    if raw == "":
        return {}

    # Already dict
    if isinstance(raw, dict):
        return raw

    # List → convert to {"arg": None}
    if isinstance(raw, list):
        return {item: None for item in raw}

    # String case → try json.loads
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return loaded
            if isinstance(loaded, list):
                return {item: None for item in loaded}
            return {}
        except:
            # fallback: treat as single key
            return {raw: None}

    # Unknown type
    return {}


def parse_positional_args(raw):
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return loaded
            return []
        except:
            return [raw]
    return []


# =================================================
# Alias handling
# =================================================
def expand_alias_group(alias_str):
    return alias_str.split("|")


# =================================================
# Helper: Identify flags that expect values
# =================================================
def get_flags_with_values(gt_optional):
    """
    Return a set of flags (including aliases) that expect a value,
    based on the ground truth optional args dictionary.
    """
    flags = set()
    if not gt_optional:
        return flags

    for key, val in gt_optional.items():
        # If val is not None, it implies this flag expects a value
        if val is not None:
            aliases = expand_alias_group(key)
            flags.update(aliases)
    return flags


# =================================================
# Parse predicted command
# =================================================
def parse_command(cmd, flags_with_values=None):
    if flags_with_values is None:
        flags_with_values = set()

    # Use shlex to handle quotes properly
    try:
        parts = shlex.split(cmd.strip())
    except ValueError:
        # Fallback for unbalanced quotes or other shlex errors
        parts = cmd.strip().split()

    if not parts:
        return "", [], []

    tool = parts[0]
    optional = []
    positional = []

    i = 1
    while i < len(parts):
        p = parts[i]
        if p.startswith("-"):
            # Handle --flag=value case
            if "=" in p:
                flag_part, value_part = p.split("=", 1)
                optional.append(flag_part)
                # Value is embedded, so we don't consume another token
            else:
                optional.append(p)
                # ONLY consume next token if this flag is KNOWN to take a value (from GT schema)
                if p in flags_with_values:
                    if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                        i += 1  # consume the value
        else:
            positional.append(p)
        i += 1

    return tool, optional, positional


# =================================================
# Optional F1 scoring with alias support
# =================================================
def optional_f1(gt_alias_dict, pred_optional):
    if len(gt_alias_dict) == 0 and len(pred_optional) == 0:
        return 1.0

    if len(gt_alias_dict) == 0:
        return 0.0

    if len(pred_optional) == 0:
        return 0.0

    gt_groups = [expand_alias_group(g) for g in gt_alias_dict.keys()]
    pred_set = Counter(pred_optional)

    tp = 0
    for group in gt_groups:
        if any(a in pred_set for a in group):
            tp += 1

    gt_pos = len(gt_groups)
    pred_pos = len(pred_optional)

    precision = tp / pred_pos if pred_pos else 0
    recall = tp / gt_pos if gt_pos else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# =================================================
# Positional F1
# =================================================
def positional_f1(gt_positional, pred_positional):
    if len(gt_positional) == 0 and len(pred_positional) == 0:
        return 1.0

    if len(gt_positional) == 0:
        return 0.0

    if len(pred_positional) == 0:
        return 0.0

    gt_set = Counter(gt_positional)
    pred_set = Counter(pred_positional)

    tp = sum((gt_set & pred_set).values())
    fp = sum((pred_set - gt_set).values())
    fn = sum((gt_set - pred_set).values())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# =================================================
# Exact match with alias support
# =================================================
def exact_match(gt_cmd, gt_alias_dict, pred_cmd):
    # Identify flags that take values to parse correctly
    flags_with_values = get_flags_with_values(gt_alias_dict)

    pred_tool, pred_opt, pred_pos = parse_command(pred_cmd, flags_with_values)
    gt_tool, _, gt_pos = parse_command(gt_cmd, flags_with_values)

    if pred_tool != gt_tool:
        return False
    if pred_pos != gt_pos:
        return False

    for alias_group in gt_alias_dict.keys():
        aliases = expand_alias_group(alias_group)
        if not any(a in pred_opt for a in aliases):
            return False

    allowed = set()
    for grp in gt_alias_dict.keys():
        allowed.update(expand_alias_group(grp))

    for p in pred_opt:
        if p not in allowed:
            return False

    return True


# ===========================================================
# Compute scores for one sample
# ===========================================================
def score_one(label, model_reply):
    pred_cmd = extract_command_from_output(model_reply)
    format_error = 1.0 if pred_cmd == FORMAT_ERROR_TOKEN else 0.0

    # Parse GT optional args first to know which flags take values
    gt_optional = parse_optional_args(label.get("optional_args"))
    flags_with_values = get_flags_with_values(gt_optional)

    # Parse predicted command using the schema knowledge
    pred_tool, pred_opt, pred_pos = parse_command(pred_cmd, flags_with_values)

    gt_cmd = label["ground_truth_command"]
    #gt_cmd = label["gt_cmd"]
    gt_tool = label["tool_name"]
    gt_pos = parse_positional_args(label.get("positional_args"))

    tool_score = 1.0 if pred_tool == gt_tool else 0.0
    optional_score = optional_f1(gt_optional, pred_opt)
    positional_score = positional_f1(gt_pos, pred_pos)
    exact_score = 1.0 if exact_match(gt_cmd, gt_optional, pred_cmd) else 0.0

    # === 新增：total_score ===
    total_score = (tool_score + optional_score + positional_score) / 3.0

    return {
        "pred_cmd": pred_cmd,
        "gt_cmd": gt_cmd,
        "tool_score": tool_score,
        "optional_f1": optional_score,
        "positional_f1": positional_score,
        "total_score": total_score,
        "exact_match": exact_score,
        "format_error": format_error,
    }


# ===========================================================
# Main: batch scoring
# ===========================================================
def score_batch(vllm_file, label_file, out_file="scores.jsonl"):
    labels = {}
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels[obj["custom_id"]] = obj

    out_f = open(out_file, "w", encoding="utf-8")

    total = 0
    agg = {
        "tool_score": 0,
        "optional_f1": 0,
        "positional_f1": 0,
        "total_score": 0,   # === 新增：聚合 total_score ===
        "exact_match": 0,
        "format_error": 0,
    }

    with open(vllm_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj["custom_id"]

            if cid not in labels:
                continue

            model_reply = obj["response"]["body"]["choices"][0]["message"]["content"]
            label = labels[cid]

            scores = score_one(label, model_reply)
            scores["custom_id"] = cid

            # Append query, tool_name, positional_args, optional_args
            scores["query"] = label.get("query")
            scores["tool_name"] = label.get("tool_name")
            scores["positional_args"] = label.get("positional_args")
            scores["optional_args"] = label.get("optional_args")

            total += 1
            for k in agg.keys():
                agg[k] += scores[k]

            out_f.write(json.dumps(scores, ensure_ascii=False) + "\n")

    out_f.close()

    summary = {k: v / total for k, v in agg.items()}
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Processed {total} samples")
    print(f"Scores saved to: {out_file}")

    return summary

# -------------------------
# CLI wrapper
# -------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Strict scoring (behavior matches original code).")
    p.add_argument("--vllm_file", "-i", required=True, help="Path to vllm output JSONL")
    p.add_argument("--label_file", "-l", required=True, help="Path to label JSONL")
    p.add_argument("--out", "-o", default="scores.jsonl", help="Output JSONL path")
    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    score_batch(args.vllm_file, args.label_file, out_file=args.out)

if __name__ == "__main__":
    main()
