"""Configuration and learner-owned Transformer model boundary."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from .layers import LayerNorm, TransformerBlock
from .utils import softmax

@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    block_size: int = 8
    model_dim: int = 16
    num_heads: int = 4
    num_layers: int = 2
    feed_forward_dim: int = 32

    def __post_init__(self) -> None:
        for name in (
            "vocab_size", "block_size", "model_dim", "num_heads", "num_layers", "feed_forward_dim"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads


class CharTransformer:
    """Causal next-token model: token IDs (B, T) -> logits (B, T, V)."""

    def __init__(self, config: TransformerConfig, *, seed: int = 7) -> None:
        self.config = config
        self.seed = seed
        # initialize token/position embeddings, blocks,
        # final norm, and output projection. Keep a named parameter collection
        # if you later reuse the stage 06 optimizer/checkpoint approach.
        rng = np.random.default_rng(seed)
        D = self.config.model_dim
        V = self.config.vocab_size
        T = self.config.block_size
        self.params = {
            "token_embedding": rng.normal(0.0, np.sqrt(1.0 / D), size=(V, D)),
            "position_embedding": rng.normal(0.0, np.sqrt(1.0 / D), size=(T, D)),
            "W_out": rng.normal(0.0, 1 / np.sqrt(D), size=(D, V)),
            "b_out": np.zeros(V),
        }

        self.blocks = [
            TransformerBlock(
                config.model_dim,
                config.num_heads,
                config.feed_forward_dim,
                seed=seed + i,
            )
            for i in range(config.num_layers)
        ]
        self.final_norm = LayerNorm(D)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Return logits for every position; require 1 <= T <= block_size."""
        logits, _ = self._forward_with_cache(token_ids)
        return logits

    def _forward_with_cache(self, token_ids: np.ndarray) -> tuple[np.ndarray, tuple]:
        if token_ids.ndim != 2 or token_ids.shape[0] < 1:
            raise ValueError("token_ids must have shape (B, T) with B >= 1")
        B, T = token_ids.shape
        if T < 1:
            raise ValueError("T must be >= 1")
        if T > self.config.block_size:
            raise ValueError(f"T ({T}) must be <= block_size ({self.config.block_size})")
        if not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token_ids must contain integer token IDs")
        if np.any((token_ids < 0) | (token_ids >= self.config.vocab_size)):
            raise ValueError("token_ids must be within the vocabulary")
        x = self.params["token_embedding"][token_ids] + self.params["position_embedding"][
            np.arange(T), :
        ]
        block_caches = []
        for block in self.blocks:
            x, block_cache = block.forward_with_cache(x)
            block_caches.append(block_cache)
        normalized, norm_cache = self.final_norm.forward_with_cache(x)
        logits = normalized @ self.params["W_out"] + self.params["b_out"]
        return logits, (normalized, norm_cache, block_caches)

    def _validate_targets(self, token_ids: np.ndarray, targets: np.ndarray) -> None:
        if token_ids.shape != targets.shape:
            raise ValueError("token_ids and targets must have the same shape")
        if targets.size == 0:
            raise ValueError("targets must not be empty")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("targets must contain integer token IDs")
        if np.any((targets < 0) | (targets >= self.config.vocab_size)):
            raise ValueError("target token IDs must be within the vocabulary")

    def _cross_entropy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        flat_logits = logits.reshape(-1, self.config.vocab_size)
        target_ids = targets.reshape(-1)
        log_normalizer = np.logaddexp.reduce(flat_logits, axis=1)
        chosen_logits = flat_logits[np.arange(target_ids.size), target_ids]
        return float(np.mean(log_normalizer - chosen_logits))

    def loss(self, token_ids: np.ndarray, targets: np.ndarray) -> float:
        """Mean next-token cross-entropy over batch and sequence positions."""
        self._validate_targets(token_ids, targets)
        return self._cross_entropy(self.forward(token_ids), targets)

    def named_parameters(self) -> dict[str, np.ndarray]:
        """Expose all trainable arrays under the same names as their gradients."""
        params = dict(self.params)
        params.update({f"final_norm.{name}": value for name, value in self.final_norm.params.items()})
        for index, block in enumerate(self.blocks):
            for module_name in ("norm1", "attention", "norm2", "ffn"):
                module = getattr(block, module_name)
                params.update({
                    f"blocks.{index}.{module_name}.{name}": value
                    for name, value in module.params.items()
                })
        return params

    def loss_and_gradients(
        self, token_ids: np.ndarray, targets: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Return loss and gradients for every trainable parameter."""
        self._validate_targets(token_ids, targets)
        logits, (normalized, norm_cache, block_caches) = self._forward_with_cache(token_ids)
        loss = self._cross_entropy(logits, targets)
        grad_logits = softmax(logits)
        flat_grad_logits = grad_logits.reshape(-1, self.config.vocab_size)
        target_ids = targets.reshape(-1)
        flat_grad_logits[np.arange(target_ids.size), target_ids] -= 1.0
        grad_logits /= target_ids.size

        grads = {
            "W_out": normalized.reshape(-1, self.config.model_dim).T
            @ grad_logits.reshape(-1, self.config.vocab_size),
            "b_out": grad_logits.sum(axis=(0, 1)),
        }
        grad_normalized = grad_logits @ self.params["W_out"].T
        grad_x, norm_grads = self.final_norm.backward(grad_normalized, norm_cache)
        grads.update({f"final_norm.{name}": grad for name, grad in norm_grads.items()})

        for index in reversed(range(len(self.blocks))):
            grad_x, block_grads = self.blocks[index].backward(
                grad_x, block_caches[index]
            )
            grads.update({
                f"blocks.{index}.{name}": grad for name, grad in block_grads.items()
            })

        grad_token_embedding = np.zeros_like(self.params["token_embedding"])
        np.add.at(grad_token_embedding, token_ids, grad_x)
        grad_position_embedding = np.zeros_like(self.params["position_embedding"])
        grad_position_embedding[: token_ids.shape[1]] = grad_x.sum(axis=0)
        grads["token_embedding"] = grad_token_embedding
        grads["position_embedding"] = grad_position_embedding
        return loss, grads

    def save(self, path: str | Path) -> None:
        """Save an inference-only model with config and parameters."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": np.asarray(1, dtype=np.int64),
            **{
                field.name: np.asarray(getattr(self.config, field.name), dtype=np.int64)
                for field in fields(TransformerConfig)
            },
            **self.named_parameters(),
        }
        with output_path.open("wb") as output_file:
            np.savez(output_file, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "CharTransformer":
        """Restore an inference-only model without changing its logits."""
        with np.load(path, allow_pickle=False) as data:
            config_fields = fields(TransformerConfig)
            required_names = {"format_version", *(field.name for field in config_fields)}
            if not required_names.issubset(data.files):
                raise ValueError("saved model is missing config or format fields")
            version = data["format_version"]
            if version.shape != () or version.dtype != np.dtype("int64") or version.item() != 1:
                raise ValueError("unsupported model format version")
            config_values = {}
            for field in config_fields:
                value = data[field.name]
                if value.shape != () or not np.issubdtype(value.dtype, np.integer):
                    raise ValueError(f"invalid config value for {field.name}")
                config_values[field.name] = int(value.item())
            config = TransformerConfig(**config_values)
            model = cls(config)
            parameters = model.named_parameters()
            expected_names = required_names | set(parameters)
            if set(data.files) != expected_names:
                raise ValueError("saved model has missing or unexpected fields")
            for name, expected in parameters.items():
                saved = data[name]
                if saved.shape != expected.shape or saved.dtype != expected.dtype:
                    raise ValueError(f"invalid shape or dtype for {name}")
                np.copyto(expected, saved)
        return model
