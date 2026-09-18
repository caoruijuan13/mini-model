"""NumPy Token MLP skeleton for stage 05."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable softmax over the last axis."""
    # subtract each row maximum before exponentiation.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    result = exp_values / exp_values.sum(axis=-1, keepdims=True)
    return result


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
        self.params: dict[str, np.ndarray] = {}
        # initialize these arrays with deterministic float64 data:
        # embedding: (vocab_size, embedding_dim)
        # W1:        (input_dim, hidden_dim)
        # b1:        (hidden_dim,)
        # W2:        (hidden_dim, vocab_size)
        # b2:        (vocab_size,)
        rng = np.random.default_rng(seed)

        self.params["embedding"] = rng.normal(
            0.0,
            0.1,
            size=(config.vocab_size, config.embedding_dim),
        )

        self.params["W1"] = rng.normal(
            0.0,
            np.sqrt(1.0 / config.input_dim),
            size=(config.input_dim, config.hidden_dim),
        )

        self.params["b1"] = np.zeros(
            config.hidden_dim,
            dtype=np.float64,
        )

        self.params["W2"] = rng.normal(
            0.0,
            np.sqrt(1.0 / config.hidden_dim),
            size=(config.hidden_dim, config.vocab_size),
        )

        self.params["b2"] = np.zeros(
            config.vocab_size,
            dtype=np.float64,
        )

    def forward(self, context_ids: np.ndarray) -> np.ndarray:
        """Return logits with shape ``(batch_size, vocab_size)``."""
        # embedding lookup -> flatten -> linear -> tanh -> linear.
        batch_size, context_size = context_ids.shape 
        embedded = self.params["embedding"][context_ids]
        flattened = embedded.reshape(batch_size, context_size * self.config.embedding_dim)
        linear1 = np.matmul(flattened, self.params["W1"]) + self.params["b1"]
        tanh = np.tanh(linear1)
        linear2 = np.matmul(tanh, self.params["W2"]) + self.params["b2"]
        return linear2

    def predict_proba(self, context_ids: np.ndarray) -> np.ndarray:
        """Return a numerically stable softmax over the vocabulary axis."""
        # subtract each row maximum before exponentiation.
        logits = self.forward(context_ids)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        probs = exp_values / exp_values.sum(axis=-1, keepdims=True)
        return probs

    def loss(self, context_ids: np.ndarray, targets: np.ndarray) -> float:
        """Return mean next-token cross-entropy."""
        # select each target probability and average -log(p).
        probs = self.predict_proba(context_ids)[np.arange(len(context_ids)), targets]
        return -np.mean(np.log(probs))

    def loss_and_gradients(
        self, context_ids: np.ndarray, targets: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Return loss and gradients for every entry in ``PARAMETER_NAMES``."""
        # hand-write backward propagation. Repeated token IDs
        # must accumulate embedding gradients instead of overwriting them.
        batch_size, context_size = context_ids.shape

        embedding = self.params["embedding"]
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]

        embedded = embedding[context_ids]
        flattened = embedded.reshape(batch_size, context_size * self.config.embedding_dim)
        linear1 = np.matmul(flattened, W1) + b1
        hidden = np.tanh(linear1)
        linear2 = np.matmul(hidden, W2) + b2
        probs = softmax(linear2)
        loss = -np.mean(np.log(probs[np.arange(batch_size), targets]))

        grad_logits = probs.copy()
        grad_logits[np.arange(len(context_ids)), targets] -= 1.0
        grad_logits /= batch_size

        grad_W2 = hidden.T @ grad_logits
        grad_b2 = np.sum(grad_logits, axis=0)
        grad_hidden = np.matmul(grad_logits, W2.T)

        grad_linear1 = grad_hidden * (1 - hidden ** 2)

        grad_W1 = flattened.T @ grad_linear1
        grad_b1 = np.sum(grad_linear1, axis=0)
        grad_flattened = np.matmul(grad_linear1, W1.T)

        grad_embedded = grad_flattened.reshape(batch_size, context_size, self.config.embedding_dim)
        grad_embedding = np.zeros_like(embedding)

        np.add.at(grad_embedding, context_ids, grad_embedded)

        gradients = dict(
            embedding=grad_embedding,
            W1=grad_W1,
            b1=grad_b1,
            W2=grad_W2,
            b2=grad_b2,
        )

        return loss, gradients
    def apply_gradients(
        self, gradients: dict[str, np.ndarray], learning_rate: float
    ) -> None:
        """Apply one full-batch gradient-descent update."""
        # validate gradient keys/shapes, then update parameters.
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")

        expected_names = set(self.PARAMETER_NAMES)
        actual_names = set(gradients)

        if actual_names != expected_names:
            missing = expected_names - actual_names
            extra = actual_names - expected_names
            raise ValueError(
                f"invalid gradient keys: missing={missing}, extra={extra}"
            )

        for name in self.PARAMETER_NAMES:
            parameter = self.params[name]
            gradient = gradients[name]

            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"gradient shape mismatch for {name}: "
                    f"expected {parameter.shape}, got {gradient.shape}"
                )

            if not np.all(np.isfinite(gradient)):
                raise ValueError(
                    f"gradient for {name} contains non-finite values"
                )

        for name in self.PARAMETER_NAMES:
            self.params[name] -= learning_rate * gradients[name]
        
    def save(self, path: str | Path) -> None:
        """Save config and inference parameters to an NPZ artifact."""
        # persist config fields and every model parameter.
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_path,
            vocab_size=np.asarray(
                self.config.vocab_size,
                dtype=np.int64,
            ),
            context_size=np.asarray(
                self.config.context_size,
                dtype=np.int64,
            ),
            embedding_dim=np.asarray(
                self.config.embedding_dim,
                dtype=np.int64,
            ),
            hidden_dim=np.asarray(
                self.config.hidden_dim,
                dtype=np.int64,
            ),
            **self.params,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TokenMLP":
        """Load a model without random reinitialization changing its output."""
        # rebuild config and restore every saved parameter.
        with np.load(path, allow_pickle=False) as data:
            config = MLPConfig(
                vocab_size=int(data["vocab_size"].item()),
                context_size=int(data["context_size"].item()),
                embedding_dim=int(data["embedding_dim"].item()),
                hidden_dim=int(data["hidden_dim"].item()),
            )

            model = cls(config)

            for name in cls.PARAMETER_NAMES:
                loaded_parameter = data[name]

                expected_shape = model.params[name].shape
                if loaded_parameter.shape != expected_shape:
                    raise ValueError(
                        f"invalid shape for {name}: "
                        f"expected {expected_shape}, "
                        f"got {loaded_parameter.shape}"
                    )

                model.params[name] = loaded_parameter.copy()

        return model