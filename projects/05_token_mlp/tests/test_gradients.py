from pathlib import Path
import sys

import numpy as np
import pytest

STAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(STAGE_ROOT))
sys.modules.pop("model", None)

from model import MLPConfig, TokenMLP

def test_analytic_gradients_match_finite_differences():
    model = TokenMLP(
        MLPConfig(vocab_size=4, context_size=2, embedding_dim=2, hidden_dim=3),
        seed=5,
    )
    contexts = np.array([[0, 1], [1, 2]], dtype=np.int64)
    targets = np.array([2, 3], dtype=np.int64)
    _, gradients = model.loss_and_gradients(contexts, targets)
    epsilon = 1e-6

    checks = {
        "embedding": (1, 0),
        "W1": (0, 0),
        "b1": (0,),
        "W2": (0, 0),
        "b2": (0,),
    }
    assert set(gradients) == set(TokenMLP.PARAMETER_NAMES)
    for name, index in checks.items():
        original = model.params[name][index]
        model.params[name][index] = original + epsilon
        plus = model.loss(contexts, targets)
        model.params[name][index] = original - epsilon
        minus = model.loss(contexts, targets)
        model.params[name][index] = original
        numerical = (plus - minus) / (2 * epsilon)
        assert gradients[name][index] == pytest.approx(
            numerical, rel=1e-4, abs=1e-5
        )


def test_gradient_descent_reduces_tiny_dataset_loss():
    model = TokenMLP(
        MLPConfig(vocab_size=4, context_size=2, embedding_dim=3, hidden_dim=6),
        seed=7,
    )
    contexts = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int64)
    targets = np.array([1, 2, 3], dtype=np.int64)
    initial_loss = model.loss(contexts, targets)

    for _ in range(200):
        _, gradients = model.loss_and_gradients(contexts, targets)
        model.apply_gradients(gradients, learning_rate=0.1)

    assert model.loss(contexts, targets) < initial_loss * 0.5
