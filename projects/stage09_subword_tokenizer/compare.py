"""Compare char and BPE token counts without changing the model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from projects.tokenization import CharTokenizer

from .bpe import BPEConfig, BPETokenizer
from .data import CORPUS_PATH, prepare_text_splits


def compare_tokenizers(
    *, corpus_path: str | Path = CORPUS_PATH, config: BPEConfig = BPEConfig()
) -> dict[str, Any]:
    """Fit both tokenizers on train only; measure the same lines in each split."""
    train_text, valid_text, test_text = prepare_text_splits(corpus_path)
    char_tokenizer = CharTokenizer.from_text(train_text)
    bpe_tokenizer = BPETokenizer.train(train_text, config=config)

    def split_metrics(text: str) -> dict[str, int | float]:
        # Keep line endings in both measurements so no raw character is dropped.
        lines = text.splitlines(keepends=True)
        char_lengths = [len(char_tokenizer.encode(line)) for line in lines]
        bpe_lengths = [len(bpe_tokenizer.encode(line)) for line in lines]
        return {
            "line_count": len(lines),
            "character_count": len(text),
            "char_token_count": sum(char_lengths),
            "bpe_token_count": sum(bpe_lengths),
            "char_mean_tokens_per_line": sum(char_lengths) / len(lines),
            "bpe_mean_tokens_per_line": sum(bpe_lengths) / len(lines),
        }

    return {
        "bpe_config": asdict(config),
        "char_vocab_size": char_tokenizer.vocab_size,
        "bpe_vocab_size": bpe_tokenizer.vocab_size,
        "train": split_metrics(train_text),
        "valid": split_metrics(valid_text),
        "test": split_metrics(test_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--max-merges", type=int, default=BPEConfig.max_merges)
    parser.add_argument("--min-pair-frequency", type=int, default=BPEConfig.min_pair_frequency)
    args = parser.parse_args()
    report = compare_tokenizers(
        corpus_path=args.corpus,
        config=BPEConfig(
            max_merges=args.max_merges,
            min_pair_frequency=args.min_pair_frequency,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
