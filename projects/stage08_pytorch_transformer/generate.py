"""Autoregressive sampling from a PyTorch character Transformer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import math

import torch

from .model import TorchCharTransformer


def iter_generated_ids(
    model: TorchCharTransformer,
    initial_ids: Sequence[int],
    *,
    eos_id: int,
    max_new_tokens: int = 100,
    seed: int = 7,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> Iterator[int]:
    """Yield sampled IDs one at a time, including EOS when selected."""
    vocab_size = model.config.vocab_size
    if not initial_ids:
        raise ValueError("initial_ids must contain at least one token ID")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= vocab_size):
        raise ValueError("top_k must be between 1 and vocab_size")
    if isinstance(eos_id, bool) or not isinstance(eos_id, int) or not 0 <= eos_id < vocab_size:
        raise ValueError("eos_id outside vocabulary")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < vocab_size for token_id in initial_ids):
        raise ValueError("initial_ids must contain valid integer token IDs")

    generated = list(initial_ids)
    if generated[-1] == eos_id:
        return

    rng = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        context = torch.tensor(
            [generated[-model.config.block_size:]], dtype=torch.long, device=device
        )
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                logits = model(context)
                expected_shape = (1, context.shape[1], vocab_size)
                if logits.shape != expected_shape:
                    raise ValueError(f"model logits must have shape {expected_shape}")
                next_logits = logits[0, -1].float().cpu()
                if not torch.isfinite(next_logits).all():
                    raise ValueError("model returned non-finite logits")
                next_logits = next_logits / temperature
                if top_k is not None and top_k < vocab_size:
                    top_indices = torch.topk(next_logits, top_k).indices
                    filtered = torch.full_like(next_logits, float("-inf"))
                    filtered[top_indices] = next_logits[top_indices]
                    next_logits = filtered
                probabilities = torch.softmax(next_logits, dim=-1)
                next_id = int(
                    torch.multinomial(probabilities, 1, generator=rng).item()
                )
        finally:
            model.train(was_training)

        generated.append(next_id)
        yield next_id
        if next_id == eos_id:
            return


def generate_ids(
    model: TorchCharTransformer,
    initial_ids: Sequence[int],
    *,
    eos_id: int,
    max_new_tokens: int = 100,
    seed: int = 7,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[int]:
    """Return the prompt plus sampled IDs; stop at EOS or the token limit."""
    generated = list(initial_ids)
    generated.extend(
        iter_generated_ids(
            model,
            initial_ids,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
        )
    )
    return generated
