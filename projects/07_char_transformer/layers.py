"""Learner TODOs: normalization, position-wise FFN, and residual block."""

from __future__ import annotations

import numpy as np
from .attention import MultiHeadAttention


class LayerNorm:
    def __init__(self, model_dim: int, *, epsilon: float = 1e-5) -> None:
        self.model_dim = model_dim
        self.epsilon = epsilon
        # initialize trainable gamma and beta, each (D,).
        self.params = {
            "gamma": np.ones(model_dim),
            "beta": np.zeros(model_dim),
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize over the last feature axis only; preserve (B, T, D)."""
        output, _ = self.forward_with_cache(x)
        return output

    def forward_with_cache(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.epsilon)
        normalized = (x - mean) * inv_std
        output = self.params["gamma"] * normalized + self.params["beta"]
        return output, (normalized, inv_std)

    def backward(self, grad_output: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        normalized, inv_std = cache
        grad_gamma = np.sum(grad_output * normalized, axis=(0, 1))
        grad_beta = np.sum(grad_output, axis=(0, 1))
        grad_normalized = grad_output * self.params["gamma"]
        grad_input = inv_std * (
            grad_normalized
            - np.mean(grad_normalized, axis=-1, keepdims=True)
            - normalized * np.mean(
                grad_normalized * normalized, axis=-1, keepdims=True
            )
        )
        return grad_input, {"gamma": grad_gamma, "beta": grad_beta}

class FeedForward:
    def __init__(self, model_dim: int, hidden_dim: int, *, seed: int = 7) -> None:
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0.0, np.sqrt(1.0 / model_dim), size=(model_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "W2": rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, model_dim)),
            "b2": np.zeros(model_dim),
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the same two-layer nonlinearity to each sequence position."""
        output, _ = self.forward_with_cache(x)
        return output

    def forward_with_cache(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        pre_activation = x @ self.params["W1"] + self.params["b1"]
        hidden = np.maximum(pre_activation, 0.0)
        output = hidden @ self.params["W2"] + self.params["b2"]
        return output, (x, pre_activation, hidden)

    def backward(self, grad_output: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x, pre_activation, hidden = cache
        grad_w2 = hidden.reshape(-1, self.hidden_dim).T @ grad_output.reshape(
            -1, self.model_dim
        )
        grad_b2 = grad_output.sum(axis=(0, 1))
        grad_pre_activation = (grad_output @ self.params["W2"].T) * (
            pre_activation > 0.0
        )
        grad_w1 = x.reshape(-1, self.model_dim).T @ grad_pre_activation.reshape(
            -1, self.hidden_dim
        )
        grad_b1 = grad_pre_activation.sum(axis=(0, 1))
        grad_input = grad_pre_activation @ self.params["W1"].T
        return grad_input, {"W1": grad_w1, "b1": grad_b1, "W2": grad_w2, "b2": grad_b2}


class TransformerBlock:
    """Pre-LN: x + attention(LN(x)), then x + FFN(LN(x))."""

    def __init__(
        self, model_dim: int, num_heads: int, hidden_dim: int, *, seed: int = 7
    ) -> None:
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.seed = seed
        # compose two norms, one attention, and one FFN.
        self.norm1 = LayerNorm(model_dim)
        self.norm2 = LayerNorm(model_dim)
        self.attention = MultiHeadAttention(model_dim, num_heads, seed=seed)
        self.ffn = FeedForward(model_dim, hidden_dim, seed=seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Preserve (B, T, D) and keep all attention paths causal."""
        output, _ = self.forward_with_cache(x)
        return output

    def forward_with_cache(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        normed1, norm1_cache = self.norm1.forward_with_cache(x)
        attention_output, attention_cache = self.attention.forward_with_cache(normed1)
        after_attention = x + attention_output
        normed2, norm2_cache = self.norm2.forward_with_cache(after_attention)
        ffn_output, ffn_cache = self.ffn.forward_with_cache(normed2)
        return after_attention + ffn_output, (
            norm1_cache, attention_cache, norm2_cache, ffn_cache
        )

    def backward(self, grad_output: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        norm1_cache, attention_cache, norm2_cache, ffn_cache = cache
        grad_ffn_input, ffn_grads = self.ffn.backward(grad_output, ffn_cache)
        grad_after_norm2, norm2_grads = self.norm2.backward(
            grad_ffn_input, norm2_cache
        )
        grad_after_attention = grad_output + grad_after_norm2
        grad_attention_input, attention_grads = self.attention.backward(
            grad_after_attention, attention_cache
        )
        grad_before_norm1, norm1_grads = self.norm1.backward(
            grad_attention_input, norm1_cache
        )
        grads = {
            **{f"norm1.{name}": grad for name, grad in norm1_grads.items()},
            **{f"attention.{name}": grad for name, grad in attention_grads.items()},
            **{f"norm2.{name}": grad for name, grad in norm2_grads.items()},
            **{f"ffn.{name}": grad for name, grad in ffn_grads.items()},
        }
        return grad_after_attention + grad_before_norm1, grads
