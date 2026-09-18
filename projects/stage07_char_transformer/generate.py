"""Autoregressive token-ID generation for the character Transformer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .model import CharTransformer
from .utils import softmax


def generate_ids(
    model: CharTransformer,
    initial_ids: Sequence[int],
    *,
    eos_id: int,
    max_new_tokens: int = 100,
    seed: int = 7,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[int]:
    """Sample new IDs from final-position logits, preserving the full prompt."""
    vocab_size = model.config.vocab_size
    block_size = model.config.block_size
    if len(initial_ids) == 0:
        raise ValueError("initial_ids must contain at least one token ID")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float, np.integer, np.floating))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be positive and finite")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer or None")
        if not 1 <= top_k <= vocab_size:
            raise ValueError(f"top_k must be between 1 and {vocab_size}")
    if isinstance(eos_id, bool) or not isinstance(eos_id, (int, np.integer)):
        raise TypeError("eos_id must be an integer")
    if not 0 <= eos_id < vocab_size:
        raise ValueError("eos_id outside vocabulary")

    generated = []
    for token_id in initial_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise TypeError("initial_ids must contain integer token IDs")
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"initial token ID outside vocabulary: {token_id}")
        generated.append(int(token_id))

    if generated[-1] == eos_id:
        return generated

    rng = np.random.default_rng(seed)
    for _ in range(max_new_tokens):
        context_ids = np.asarray(
            [generated[-block_size:]],
            dtype=np.int64,
        )
        batch_logits = np.asarray(model.forward(context_ids))
        expected_shape = (1, context_ids.shape[1], vocab_size)
        if batch_logits.shape != expected_shape:
            raise ValueError(
                "model.forward() must return logits with shape "
                f"{expected_shape}"
            )
        logits = batch_logits[0, -1, :].astype(np.float64, copy=True)
        if not np.all(np.isfinite(logits)):
            raise ValueError("model.forward() returned non-finite logits")
        logits -= logits.max()
        logits /= temperature
        if top_k is not None and top_k < vocab_size:
            top_indices = np.argsort(logits)[-top_k:]
            filtered = np.full_like(logits, -np.inf)
            filtered[top_indices] = logits[top_indices]
            logits = filtered

        next_id = int(rng.choice(vocab_size, p=softmax(logits)))
        generated.append(next_id)
        if next_id == eos_id:
            break
    return generated