#!/usr/bin/env python3
"""Run RedSage-K GRPO inference from the command line."""

if __package__:
    from .inference import build_parser, parse_args, run_inference
else:
    from inference import build_parser, parse_args, run_inference


SYSTEM_PROMPT = """You are a cybersecurity function-calling AI model.
You have access to the following tool:
<tools>
[{'type':'function','function':{'name':'run_terminal','description':'Execute a shell command in a Kali/Linux terminal and return stdout, stderr, and exit code.'}}]
</tools>

TASK:
Given a USER QUERY, generate the single most accurate shell command using any appropriate Kali/Linux tool(s) to solve the query.

REQUIREMENTS:
1. Use the correct command-line tool(s) appropriate for the task.
2. Include all required optional arguments (flags beginning with '-' or '--') necessary to accomplish the task.
3. Correctly pair option keys and values (e.g., `--port 80`, `-A INPUT`, or `--flag=value`).
4. Preserve and include any positional arguments (e.g., IPs, filenames, interfaces).
5. Do NOT invent flags/options that do not exist for real Kali/Linux tools.
6. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

OUTPUT FORMAT:
<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""

USER_TEMPLATE = """USER QUERY: "{query}"

Generate the single most accurate shell command for the query.
Your response must follow the required structure:

<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""


def build_messages(query):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(query=query + "\n<think>\n")},
    ]


def main(argv=None):
    parser = build_parser(
        description="Generate a Kali/Linux command with RedSage-K GRPO.",
        default_model="RISys-Lab/RedSage-K-SFT-GRPO",
        max_new_tokens=8192,
        temperature=0.1,
    )
    args = parse_args(parser, argv)
    run_inference(args, build_messages(args.query))


if __name__ == "__main__":
    main()
