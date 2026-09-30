#!/usr/bin/env python3
"""Run RedSage-K SFT inference from the command line."""

if __package__:
    from .inference import build_parser, parse_args, run_inference
else:
    from inference import build_parser, parse_args, run_inference


SYSTEM_PROMPT = 'Translate the request into a single accurate Kali/Linux command. Return only <output>command</output>.'


def build_messages(query):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def main(argv=None):
    parser = build_parser(
        description="Generate a Kali/Linux command with RedSage-K SFT.",
        default_model="RISys-Lab/RedSage-K-SFT",
        max_new_tokens=256,
        temperature=0.0,
    )
    args = parse_args(parser, argv)
    run_inference(args, build_messages(args.query))


if __name__ == "__main__":
    main()
