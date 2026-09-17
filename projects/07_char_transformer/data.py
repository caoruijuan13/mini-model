"""Completed data boundary for next-token Transformer exercises."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from projects.tokenization import CharTokenizer


PROJECT_ROOT = Path(__file__).parents[2]
CORPUS_PATH = PROJECT_ROOT / "data" / "declaration_excerpt.txt"

def load_corpus(path: str | Path = CORPUS_PATH) -> str:
    return Path(path).read_text(encoding="utf-8").rstrip("\n")


def make_text_splits(text: str) -> tuple[str, str, str]:
    """Preserve the sequential 80/10/10 split used in stages 05 and 06."""
    if len(text) < 30:
        raise ValueError("text must contain at least 30 characters")
    train_end = int(len(text) * 0.8)
    valid_end = int(len(text) * 0.9)
    return text[:train_end], text[train_end:valid_end], text[valid_end:]


def build_sequence_windows(
    token_ids: Sequence[int], *, block_size: int, bos_id: int, eos_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length, one-token-shifted input/target windows.

    Each split receives its own BOS/EOS. For IDs [3, 4, 5] and block_size=2:
    inputs  = [[BOS, 3], [3, 4], [4, 5]]
    targets = [[3, 4], [4, 5], [5, EOS]]

    Windows overlap intentionally so the final target can include EOS without
    padding or mixing text from different data splits.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if token_ids is None:
        raise TypeError("token_ids must be a sequence of integers")
    ids = [*token_ids]
    if any(isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)) or token_id < 0 for token_id in ids):
        raise ValueError("token_ids must contain non-negative integers")
    if len(ids) + 1 < block_size:
        raise ValueError("sequence is too short for block_size")

    sequence = [bos_id, *ids, eos_id]
    inputs = [sequence[start : start + block_size] for start in range(len(sequence) - block_size)]
    targets = [sequence[start + 1 : start + block_size + 1] for start in range(len(sequence) - block_size)]
    return np.asarray(inputs, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def prepare_corpus(
    *, block_size: int = 8, corpus_path: str | Path = CORPUS_PATH
) -> tuple[
    CharTokenizer,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Fit the tokenizer on training text only; build each split independently."""
    train_text, valid_text, test_text = make_text_splits(load_corpus(corpus_path))
    tokenizer = CharTokenizer.from_text(train_text)
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    datasets = [
        build_sequence_windows(
            tokenizer.encode(text), block_size=block_size, bos_id=bos_id, eos_id=eos_id
        )
        for text in (train_text, valid_text, test_text)
    ]
    return tokenizer, datasets[0], datasets[1], datasets[2]
