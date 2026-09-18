from pathlib import Path
import sys

import numpy as np
import pytest

STAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(STAGE_ROOT))
sys.modules.pop("generate", None)
sys.modules.pop("model", None)

from generate import generate_ids
from model import MLPConfig, TokenMLP

class _ScriptedModel:
    def __init__(self) -> None:
        self.config = MLPConfig(
            vocab_size=5, context_size=2, embedding_dim=1, hidden_dim=1
        )
        self.seen_contexts: list[tuple[int, ...]] = []

    def forward(self, context_ids: np.ndarray) -> np.ndarray:
        context = tuple(int(value) for value in context_ids[0])
        self.seen_contexts.append(context)
        next_by_last_id = {1: 3, 3: 4, 4: 2}
        logits = np.full((1, self.config.vocab_size), -100.0)
        logits[0, next_by_last_id[context[-1]]] = 100.0
        return logits


def test_generation_rolls_context_and_stops_after_eos():
    model = _ScriptedModel()

    generated = generate_ids(
        model,
        [1, 1],
        eos_id=2,
        max_new_tokens=8,
        seed=9,
        temperature=1.0,
        top_k=1,
    )

    assert generated == [1, 1, 3, 4, 2]
    assert model.seen_contexts == [(1, 1), (1, 3), (3, 4)]


def test_generation_is_reproducible_and_keeps_initial_context():
    model = TokenMLP(
        MLPConfig(vocab_size=5, context_size=2, embedding_dim=3, hidden_dim=4),
        seed=7,
    )

    first = generate_ids(model, [1, 1], eos_id=2, max_new_tokens=8, seed=9)
    second = generate_ids(model, [1, 1], eos_id=2, max_new_tokens=8, seed=9)

    assert first == second
    assert first[:2] == [1, 1]
    assert len(first) <= 10


def test_save_and_load_preserve_logits(tmp_path):
    model = TokenMLP(
        MLPConfig(vocab_size=5, context_size=2, embedding_dim=3, hidden_dim=4),
        seed=13,
    )
    contexts = np.array([[0, 1], [2, 3]], dtype=np.int64)
    path = tmp_path / "token_mlp.npz"

    expected = model.forward(contexts)
    model.save(path)
    loaded = TokenMLP.load(path)

    assert loaded.config == model.config
    assert np.array_equal(loaded.forward(contexts), expected)
