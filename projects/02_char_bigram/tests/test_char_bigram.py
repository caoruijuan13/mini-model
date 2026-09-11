import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 项目脚本使用直接导入；清除其他阶段可能缓存的同名模块。
sys.modules.pop("data", None)
sys.modules.pop("model", None)
from data import DEFAULT_CORPUS, make_char_splits
from model import CharBigram

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
