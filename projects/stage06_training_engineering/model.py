"""The stage 05 Token MLP, kept unchanged as the stage 06 model boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class MLPConfig:
    vocab_size: int
    context_size: int = 2
    embedding_dim: int = 8
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        for name, value in (
            ("vocab_size", self.vocab_size),
            ("context_size", self.context_size),
            ("embedding_dim", self.embedding_dim),
            ("hidden_dim", self.hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def input_dim(self) -> int:
        return self.context_size * self.embedding_dim


class TokenMLP:
    """Predict next-token logits from fixed-width token-ID contexts."""

    PARAMETER_NAMES = ("embedding", "W1", "b1", "W2", "b2")

    def __init__(self, config: MLPConfig, seed: int = 7) -> None:
        self.config = config
        rng = np.random.default_rng(seed)
        self.params = {
            "embedding": rng.normal(
                0.0, 0.1, size=(config.vocab_size, config.embedding_dim)
            ),
            "W1": rng.normal(
                0.0,
                np.sqrt(1.0 / config.input_dim),
                size=(config.input_dim, config.hidden_dim),
            ),
            "b1": np.zeros(config.hidden_dim, dtype=np.float64),
            "W2": rng.normal(
                0.0,
                np.sqrt(1.0 / config.hidden_dim),
                size=(config.hidden_dim, config.vocab_size),
            ),
            "b2": np.zeros(config.vocab_size, dtype=np.float64),
        }

    def forward(self, context_ids: np.ndarray) -> np.ndarray:
        batch_size = context_ids.shape[0]
        embedded = self.params["embedding"][context_ids]
        features = embedded.reshape(batch_size, self.config.input_dim)
        hidden = np.tanh(features @ self.params["W1"] + self.params["b1"])
        return hidden @ self.params["W2"] + self.params["b2"]

    def predict_proba(self, context_ids: np.ndarray) -> np.ndarray:
        return softmax(self.forward(context_ids))

    def loss(self, context_ids: np.ndarray, targets: np.ndarray) -> float:
        probs = self.predict_proba(context_ids)
        target_probs = probs[np.arange(len(targets)), targets]
        return float(-np.mean(np.log(target_probs)))

    def loss_and_gradients(
        self, context_ids: np.ndarray, targets: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        batch_size = context_ids.shape[0]
        embedded = self.params["embedding"][context_ids]
        features = embedded.reshape(batch_size, self.config.input_dim)
        hidden = np.tanh(features @ self.params["W1"] + self.params["b1"])
        probs = softmax(hidden @ self.params["W2"] + self.params["b2"])
        loss = float(-np.mean(np.log(probs[np.arange(batch_size), targets])))

        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), targets] -= 1.0
        grad_logits /= batch_size
        grad_W2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ self.params["W2"].T
        grad_z1 = grad_hidden * (1.0 - hidden**2)
        grad_W1 = features.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)
        grad_embedded = (grad_z1 @ self.params["W1"].T).reshape(
            batch_size, self.config.context_size, self.config.embedding_dim
        )
        grad_embedding = np.zeros_like(self.params["embedding"])
        np.add.at(grad_embedding, context_ids, grad_embedded)
        return loss, {
            "embedding": grad_embedding,
            "W1": grad_W1,
            "b1": grad_b1,
            "W2": grad_W2,
            "b2": grad_b2,
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            vocab_size=np.asarray(self.config.vocab_size, dtype=np.int64),
            context_size=np.asarray(self.config.context_size, dtype=np.int64),
            embedding_dim=np.asarray(self.config.embedding_dim, dtype=np.int64),
            hidden_dim=np.asarray(self.config.hidden_dim, dtype=np.int64),
            **self.params,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TokenMLP":
        with np.load(path, allow_pickle=False) as data:
            config = MLPConfig(
                vocab_size=int(data["vocab_size"].item()),
                context_size=int(data["context_size"].item()),
                embedding_dim=int(data["embedding_dim"].item()),
                hidden_dim=int(data["hidden_dim"].item()),
            )
            model = cls(config)
            for name in cls.PARAMETER_NAMES:
                value = data[name]
                if value.shape != model.params[name].shape:
                    raise ValueError(f"invalid shape for {name}: {value.shape}")
                model.params[name] = value.copy()
        return model
