import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.modules.pop("data", None)
sys.modules.pop("model", None)
from data import REAL_CORPUS, make_char_splits
from model import CharTrigram

sys.modules.pop("data", None)
sys.modules.pop("model", None)


def test_trigram_probability_and_real_corpus_evaluation():
    train, valid, test = make_char_splits(REAL_CORPUS)
    model = CharTrigram("".join(sorted(set(REAL_CORPUS))))
    model.fit(train)

    assert np.allclose(model.probs.sum(axis=2), 1.0)
    assert np.isfinite(model.loss(valid))
    assert np.isfinite(model.perplexity(test))


def test_trigram_new_sampling_is_reproducible():
    train, _, _ = make_char_splits(REAL_CORPUS)
    model = CharTrigram("".join(sorted(set(REAL_CORPUS))))
    model.fit(train)

    first = model.sample_new(start="We", length=50, seed=7)
    second = model.sample_new(start="We", length=50, seed=7)
    assert first == second
    assert len(first) == 50


def test_trigram_save_and_load(tmp_path):
    train, _, _ = make_char_splits(REAL_CORPUS)
    model = CharTrigram("".join(sorted(set(REAL_CORPUS))))
    model.fit(train)

    path = tmp_path / "char_trigram.npz"
    model.save(path)
    loaded = CharTrigram.load(path)

    assert loaded.vocab == model.vocab
    assert np.allclose(loaded.probs, model.probs)
    assert np.allclose(loaded.backoff_probs, model.backoff_probs)
    assert np.array_equal(loaded.context_seen, model.context_seen)
    assert loaded.sample_new(length=30) == model.sample_new(length=30)
