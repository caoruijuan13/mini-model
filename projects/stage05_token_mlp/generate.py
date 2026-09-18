"""Autoregressive generation skeleton for stage 05."""

from collections.abc import Sequence
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from projects.tokenization import CharTokenizer

from model import TokenMLP, softmax


def generate_ids(
    model: TokenMLP,
    initial_context: Sequence[int],
    *,
    eos_id: int,
    max_new_tokens: int = 100,
    seed: int = 7,
    temperature: float = 0.8,
    top_k: int | None = 5,
) -> list[int]:
    """Return initial IDs followed by autoregressively sampled IDs."""
    # keep a rolling context_size window, transform logits with
    # temperature/top-k, sample an ID, append it, and stop after EOS.
    context_size = model.config.context_size
    vocab_size = model.config.vocab_size

    if len(initial_context) != context_size:
        raise ValueError(
            f"initial_context must contain exactly {context_size} token IDs"
        )
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer or None")
        if not 1 <= top_k <= vocab_size:
            raise ValueError(f"top_k must be between 1 and {vocab_size}")
    if isinstance(eos_id, bool) or not isinstance(eos_id, int):
        raise TypeError("eos_id must be an integer")
    if not 0 <= eos_id < vocab_size:
        raise ValueError("eos_id outside vocabulary")

    generated: list[int] = []
    for token_id in initial_context:
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise TypeError("initial_context must contain integer token IDs")
        token_id = int(token_id)
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"initial token ID outside vocabulary: {token_id}")
        generated.append(token_id)

    rng = np.random.default_rng(seed)

    for _ in range(max_new_tokens):
        context_ids = np.asarray(
            [generated[-context_size:]],
            dtype=np.int64,
        )
        batch_logits = np.asarray(model.forward(context_ids))
        if batch_logits.shape != (1, vocab_size):
            raise ValueError(
                "model.forward() must return logits with shape "
                f"(1, {vocab_size})"
            )

        logits = batch_logits[0].astype(np.float64, copy=True)
        logits /= temperature

        if top_k is not None and top_k < vocab_size:
            top_indices = np.argsort(logits)[-top_k:]
            filtered_logits = np.full_like(logits, -np.inf)
            filtered_logits[top_indices] = logits[top_indices]
            logits = filtered_logits

        probabilities = softmax(logits)

        next_id = int(
            rng.choice(vocab_size, p=probabilities)
        )
        generated.append(next_id)

        if next_id == eos_id:
            break

    return generated

def generate_text(
    model: TokenMLP,
    tokenizer: CharTokenizer,
    *,
    max_new_tokens: int = 100,
    seed: int = 7,
) -> str:
    """Generate token IDs from BOS context and decode them as text."""
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    initial_context = [bos_id] * model.config.context_size
    generated_ids = generate_ids(
        model,
        initial_context,
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
