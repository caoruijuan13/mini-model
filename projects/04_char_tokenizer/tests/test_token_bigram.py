from pathlib import Path
import sys

import numpy as np
import pytest

STAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(STAGE_ROOT))
from model import TokenBigram


def test_fit_uses_vocabulary_size_and_preserves_fractional_smoothing():
    model = TokenBigram(vocab_size=4, smoothing=1e-4)
    model.fit([0, 1, 0, 2])

    assert model.counts.shape == (4, 4)
    assert model.counts.dtype == np.float64
    assert model.counts[3, 3] == pytest.approx(1e-4)
    assert np.allclose(model.probs.sum(axis=1), 1.0)
    assert np.isfinite(model.loss([0, 1, 0, 2]))


def test_sample_follows_previous_generated_token_and_is_reproducible():
    model = TokenBigram(vocab_size=3, smoothing=1e-12)
    model.fit([0, 1, 2, 0, 1, 2, 0, 1, 2])

    expected = [0, 1, 2, 0, 1, 2, 0]
    assert model.sample(start_id=0, length=7, seed=3) == expected
    assert model.sample(start_id=0, length=7, seed=3) == model.sample(
        start_id=0, length=7, seed=3
    )


def test_sample_stops_after_eos_and_returns_token_ids():
    model = TokenBigram(vocab_size=3, smoothing=1e-12)
    model.fit([0, 1, 2, 0, 1, 2])

    assert model.sample(start_id=0, length=20, seed=5, eos_id=2) == [0, 1, 2]


def test_model_rejects_invalid_state_and_token_ids():
    model = TokenBigram(vocab_size=3)

    with pytest.raises(RuntimeError, match="fit the model"):
        model.sample(start_id=0)
    with pytest.raises(ValueError, match="outside vocabulary"):
        model.fit([0, 3])

    model.fit([0, 1])
    with pytest.raises(ValueError, match="at least one"):
        model.sample(start_id=0, length=0)
