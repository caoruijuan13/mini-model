import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 项目脚本使用直接导入；清除其他阶段可能缓存的同名模块。
sys.modules.pop("data", None)
sys.modules.pop("model", None)
from data import DEFAULT_CORPUS, REAL_CORPUS, make_char_splits
from model import CharBigram

# 保留类和函数引用，但释放顶层模块名，避免影响其他项目测试的直接导入。
sys.modules.pop("data", None)
sys.modules.pop("model", None)

def test_stage_two_probability_and_sampling(tmp_path):
    train, valid, _ = make_char_splits(DEFAULT_CORPUS)
    model = CharBigram("".join(sorted(set(DEFAULT_CORPUS))))
    model.fit(train)
    assert np.allclose(model.probs.sum(axis=1), 1.0)
    assert model.perplexity(valid) > 1.0
    assert len(model.sample(length=25)) == 25

    path = tmp_path / "char_bigram.npz"
    model.save(path)
    loaded = CharBigram.load(path)
    assert loaded.vocab == model.vocab
    assert np.allclose(loaded.probs, model.probs)


def test_new_sampling_strategy_is_separate_and_reproducible():
    train, _, _ = make_char_splits(DEFAULT_CORPUS)
    model = CharBigram("".join(sorted(set(DEFAULT_CORPUS))))
    model.fit(train)

    assert len(model.sample_new(length=25, temperature=0.8, top_k=5)) == 25
    assert model.sample_new(seed=3) == model.sample_new(seed=3)
    assert model.sample_new(top_k=None, temperature=1.0) != model.sample_new()


def test_real_corpus_supports_original_and_new_sampling():
    train, valid, test = make_char_splits(REAL_CORPUS)
    model = CharBigram("".join(sorted(set(REAL_CORPUS))))
    model.fit(train)

    assert REAL_CORPUS != DEFAULT_CORPUS
    assert np.isfinite(model.loss(valid))
    assert np.isfinite(model.perplexity(test))
    assert len(model.sample(start="W", length=50, seed=7)) == 50
    assert len(model.sample_new(start="W", length=50, seed=7)) == 50
