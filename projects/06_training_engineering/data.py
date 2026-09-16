"""Deterministic data preparation and mini-batch selection for stage 06."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parents[2]
CORPUS_PATH = PROJECT_ROOT / "data" / "declaration_excerpt.txt"


def load_corpus(path: str | Path = CORPUS_PATH) -> str:
    return Path(path).read_text(encoding="utf-8").rstrip("\n")


def make_text_splits(text: str) -> tuple[str, str, str]:
    if len(text) < 30:
        raise ValueError("text must contain at least 30 characters")
    train_end = int(len(text) * 0.8)
    valid_end = int(len(text) * 0.9)
    return text[:train_end], text[train_end:valid_end], text[valid_end:]


def build_context_targets(
    token_ids: Sequence[int], *, context_size: int, bos_id: int, eos_id: int
) -> tuple[np.ndarray, np.ndarray]:
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
    return np.asarray(contexts, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def batch_count(example_count: int, batch_size: int) -> int:
    if example_count <= 0:
        raise ValueError("example_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return (example_count + batch_size - 1) // batch_size


class DeterministicBatcher:
    """Select reproducible mini-batches while caching one permutation per epoch.

    The cache is only a performance optimization. A fresh instance can rebuild
    the same epoch permutation from ``seed + epoch`` after checkpoint recovery.
    """

    def __init__(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if len(contexts) != len(targets):
            raise ValueError("contexts and targets must have equal length")
        self.contexts = contexts
        self.targets = targets
        self.batch_size = batch_size
        self.seed = seed
        self.per_epoch = batch_count(len(targets), batch_size)
        self._cached_epoch: int | None = None
        self._cached_indices: np.ndarray | None = None

    def _build_permutation(self, epoch: int) -> np.ndarray:
        return np.random.default_rng(self.seed + epoch).permutation(len(self.targets))

    def batch_for_step(
        self, step: int
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Return the shuffled batch for a one-based optimizer step."""
        if step < 0:
            raise ValueError("step must be positive or zero")
        epoch, batch_index = divmod(step, self.per_epoch)
        if epoch != self._cached_epoch:
            self._cached_indices = self._build_permutation(epoch)
            self._cached_epoch = epoch
        assert self._cached_indices is not None
        start = batch_index * self.batch_size
        selected = self._cached_indices[start : start + self.batch_size]
        return (
            self.contexts[selected],
            self.targets[selected],
            epoch,
            batch_index
        )
