"""Data preparation skeleton for stage 05."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parents[2]
CORPUS_PATH = PROJECT_ROOT / "data" / "declaration_excerpt.txt"


def load_corpus(path: str | Path = CORPUS_PATH) -> str:
    """Load the shared corpus without treating its final newline as data."""
    return Path(path).read_text(encoding="utf-8").rstrip("\n")


def make_text_splits(text: str) -> tuple[str, str, str]:
    """Keep the existing 80/10/10 sequential split for comparison."""
    if len(text) < 30:
        raise ValueError("text must contain at least 30 characters")
    train_end = int(len(text) * 0.8)
    valid_end = int(len(text) * 0.9)
    return text[:train_end], text[train_end:valid_end], text[valid_end:]


def build_context_targets(
    token_ids: Sequence[int],
    *,
    context_size: int,
    bos_id: int,
    eos_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed-width contexts and next-token targets.

    Expected example for ``token_ids=[3, 4]`` and ``context_size=2``::

        contexts = [[bos_id, bos_id], [bos_id, 3], [3, 4]]
        targets  = [3, 4, eos_id]

    """

    if token_ids is None:
        raise TypeError("token_ids must be a sequence of integers")
    if context_size <= 0:
        raise ValueError("context_size must be positive")

    contexts: list[list[int]] = []
    targets: list[int] = []
    history = [bos_id] * context_size

    for token_id in [*token_ids, eos_id]:
        contexts.append(history[-context_size:])
        targets.append(token_id)
        history.append(token_id)

    return (
        np.asarray(contexts, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )