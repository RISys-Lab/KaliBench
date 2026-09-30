#!/usr/bin/env python3
"""Run RedSage-K GRPO inference from the command line."""

if __package__:
    from .inference import (
        GRPO_SYSTEM_PROMPT as SYSTEM_PROMPT,
        GRPO_USER_TEMPLATE as USER_TEMPLATE,
        build_parser, parse_args, run_inference,
    )
else:
    from inference import (
        GRPO_SYSTEM_PROMPT as SYSTEM_PROMPT,
        GRPO_USER_TEMPLATE as USER_TEMPLATE,
        build_parser, parse_args, run_inference,
    )


def build_messages(query):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(query=query)},
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
