"""Token-ID bigram model used to demonstrate the stage 04 boundary."""

from collections.abc import Sequence

import numpy as np


class TokenBigram:
    """Learn next-token probabilities without knowing token text."""

    def __init__(self, vocab_size: int, smoothing: float = 1e-4) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")

        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts = np.full(
            (vocab_size, vocab_size), smoothing, dtype=np.float64
        )
        self.probs: np.ndarray | None = None

    def fit(self, token_ids: Sequence[int]) -> None:
        """Estimate P(next_id | current_id) from an ordered ID sequence."""
        self._validate_token_ids(token_ids, minimum_length=2)
        self.counts.fill(self.smoothing)
        for current_id, following_id in zip(token_ids, token_ids[1:]):
            self.counts[current_id, following_id] += 1.0
        self.probs = self.counts / self.counts.sum(axis=1, keepdims=True)

    def loss(self, token_ids: Sequence[int]) -> float:
        """Return mean next-token negative log likelihood."""
        probs = self._required_probs()
        self._validate_token_ids(token_ids, minimum_length=2)
        current_ids = np.asarray(token_ids[:-1], dtype=np.int64)
        following_ids = np.asarray(token_ids[1:], dtype=np.int64)
        return float(-np.mean(np.log(probs[current_ids, following_ids])))

    def sample(
        self,
        start_id: int,
        length: int = 100,
        seed: int = 7,
        eos_id: int | None = None,
    ) -> list[int]:
        """Generate token IDs, optionally stopping after EOS is sampled.

        ``length`` is the maximum total sequence length and includes
        ``start_id``. Text decoding remains the Tokenizer's responsibility.
        """
        probs = self._required_probs()
        self._validate_token_id(start_id, "start_id")
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError("length must be an integer")
        if length < 1:
            raise ValueError("length must be at least one")
        if eos_id is not None:
            self._validate_token_id(eos_id, "eos_id")

        rng = np.random.default_rng(seed)
        result = [start_id]
        for _ in range(length - 1):
            current_id = result[-1]
            next_id = int(rng.choice(self.vocab_size, p=probs[current_id]))
            result.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return result

    def _required_probs(self) -> np.ndarray:
        if self.probs is None:
            raise RuntimeError("fit the model before evaluating or sampling")
        return self.probs

    def _validate_token_ids(
        self, token_ids: Sequence[int], *, minimum_length: int
    ) -> None:
        if len(token_ids) < minimum_length:
            raise ValueError(
                f"token_ids must contain at least {minimum_length} token IDs"
            )
        for token_id in token_ids:
            self._validate_token_id(token_id, "token ID")

    def _validate_token_id(self, token_id: int, name: str) -> None:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"{name} must be an integer")
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"{name} outside vocabulary: {token_id}")
