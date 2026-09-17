"""Learner TODOs: causal, scaled dot-product, and multi-head attention."""

from __future__ import annotations

import numpy as np
from .utils import softmax


def causal_mask(sequence_length: int) -> np.ndarray:
    """Return additive (T, T) mask: 0 for j <= i, -inf for j > i.
    construct the upper-triangular future-position mask.
    """
    future = np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1)
    mask = np.where(future, -np.inf, 0.0)
    return mask

def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (output, weights) for Q/K/V shaped (B, H, T, d_head).

    Output: (B, H, T, d_head); weights: (B, H, T, T).
    QK^T / sqrt(d_head), mask before softmax, weights @ V.
    """
    scores = query @ key.swapaxes(-1, -2) / np.sqrt(key.shape[-1])
    weights = softmax(scores + mask)
    return weights @ value, weights

class MultiHeadAttention:
    """Project (B, T, D) to H heads, attend, concatenate, project to D."""

    def __init__(self, model_dim: int, num_heads: int, *, seed: int = 7) -> None:
        if model_dim <= 0 or num_heads <= 0 or model_dim % num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.seed = seed
        # initialize W_Q, W_K, W_V and W_O.
        rng = np.random.default_rng(seed)
        self.params = {
            "W_Q": rng.normal(0.0, np.sqrt(1.0 / model_dim), size=(model_dim, model_dim)),
            "W_K": rng.normal(0.0, np.sqrt(1.0 / model_dim), size=(model_dim, model_dim)),
            "W_V": rng.normal(0.0, np.sqrt(1.0 / model_dim), size=(model_dim, model_dim))
        }
        self.params["W_O"] = rng.normal(0.0, np.sqrt(1.0 / model_dim), size=(model_dim, model_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return context-mixed vectors with the same (B, T, D) shape."""
        output, _ = self.forward_with_cache(x)
        return output

    def forward_with_cache(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        """Run attention and retain intermediates for its backward pass."""
        batch_size, sequence_length, _ = x.shape
        query = (x @ self.params["W_Q"]).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
        key = (x @ self.params["W_K"]).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
        value = (x @ self.params["W_V"]).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
        context, weights = scaled_dot_product_attention(
            query, key, value, causal_mask(sequence_length)
        )
        merged = context.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.model_dim
        )
        return merged @ self.params["W_O"], (x, query, key, value, weights, merged)

    def backward(self, grad_output: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return input and projection gradients for multi-head attention."""
        x, query, key, value, weights, merged = cache
        batch_size, sequence_length, _ = x.shape
        flat_output = grad_output.reshape(-1, self.model_dim)
        grad_w_o = merged.reshape(-1, self.model_dim).T @ flat_output
        grad_merged = grad_output @ self.params["W_O"].T
        grad_context = grad_merged.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        grad_weights = grad_context @ value.swapaxes(-1, -2)
        grad_value = weights.swapaxes(-1, -2) @ grad_context
        grad_scores = weights * (
            grad_weights - np.sum(grad_weights * weights, axis=-1, keepdims=True)
        )
        scale = 1.0 / np.sqrt(self.head_dim)
        grad_query = (grad_scores @ key) * scale
        grad_key = (grad_scores.swapaxes(-1, -2) @ query) * scale

        grad_x = np.zeros_like(x)
        grads = {"W_O": grad_w_o}
        for name, grad_head in (
            ("W_Q", grad_query), ("W_K", grad_key), ("W_V", grad_value)
        ):
            grad_projected = grad_head.transpose(0, 2, 1, 3).reshape(
                batch_size, sequence_length, self.model_dim
            )
            grads[name] = x.reshape(-1, self.model_dim).T @ grad_projected.reshape(
                -1, self.model_dim
            )
            grad_x += grad_projected @ self.params[name].T
        return grad_x, grads
