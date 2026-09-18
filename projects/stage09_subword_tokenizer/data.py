"""Use exactly the stage 07/08 raw-text split before fitting a tokenizer."""

from __future__ import annotations

from pathlib import Path

from projects.stage07_char_transformer import data as stage07_data

CORPUS_PATH = stage07_data.CORPUS_PATH


def prepare_text_splits(
    corpus_path: str | Path = CORPUS_PATH,
) -> tuple[str, str, str]:
    """Return train, validation, test text; fit rules from train only."""
    return stage07_data.make_text_splits(stage07_data.load_corpus(corpus_path))
