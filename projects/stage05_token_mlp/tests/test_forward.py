from pathlib import Path
import sys

import numpy as np
import pytest

STAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(STAGE_ROOT))
sys.modules.pop("model", None)

from model import MLPConfig, TokenMLP

def test_parameter_shapes_and_seed_are_deterministic():
    config = MLPConfig(vocab_size=5, context_size=2, embedding_dim=3, hidden_dim=4)
    first = TokenMLP(config, seed=11)
    second = TokenMLP(config, seed=11)

    assert first.params["embedding"].shape == (5, 3)
    assert first.params["W1"].shape == (6, 4)
    assert first.params["b1"].shape == (4,)
    assert first.params["W2"].shape == (4, 5)
    assert first.params["b2"].shape == (5,)
    for name in TokenMLP.PARAMETER_NAMES:
        assert np.array_equal(first.params[name], second.params[name])


def test_forward_and_softmax_shapes_are_valid():
    model = TokenMLP(
        MLPConfig(vocab_size=5, context_size=2, embedding_dim=3, hidden_dim=4)
    )
    contexts = np.array([[0, 1], [2, 3]], dtype=np.int64)

    logits = model.forward(contexts)
    probs = model.predict_proba(contexts)

    assert logits.shape == (2, 5)
    assert probs.shape == (2, 5)
    assert np.all(np.isfinite(logits))
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_softmax_is_stable_for_large_logits():
    model = TokenMLP(
        MLPConfig(vocab_size=3, context_size=1, embedding_dim=2, hidden_dim=2)
    )
    model.params["W2"].fill(1e4)

    probs = model.predict_proba(np.array([[0]], dtype=np.int64))

    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)
