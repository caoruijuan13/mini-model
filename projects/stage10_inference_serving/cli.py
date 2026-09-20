"""Minimal local command-line interface for the stage 10 runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .runtime import DEFAULT_MODEL_PATH, InferenceRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text with the stage 10 runtime"
    )
    parser.add_argument("prompt", help="text used as generation context")
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--stream", action="store_true", help="write each generated fragment immediately"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = InferenceRuntime.load(args.model)
    options = {
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    if args.stream:
        for fragment in runtime.stream_generate(args.prompt, **options):
            sys.stdout.write(fragment)
            sys.stdout.flush()
        sys.stdout.write("\n")
        return 0

    result = runtime.generate(args.prompt, **options)
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
