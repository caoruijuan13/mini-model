"""阶段二的训练、评估和模型保存入口。"""

from pathlib import Path

try:
    from .data import DEFAULT_CORPUS, make_char_splits
    from .model import CharBigram
except ImportError:
    from data import DEFAULT_CORPUS, make_char_splits
    from model import CharBigram

DEFAULT_MODEL_PATH = Path(__file__).with_name("artifacts") / "char_bigram.npz"

def train_and_evaluate(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, float]:
    train, valid, test = make_char_splits(DEFAULT_CORPUS)
    vocabulary = "".join(sorted(set(DEFAULT_CORPUS)))

    # 创建模型：dataclass 自动生成 __init__，随后调用 model.__post_init__()。
    # 这里只初始化词表映射和计数矩阵，还没有从语料学习概率。
    model = CharBigram(vocabulary)

    # 训练模型：统计 train 中的相邻字符转移，并归一化为 probs。
    model.fit(train)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return {
        "train_loss": model.loss(train),
        "valid_perplexity": model.perplexity(valid),
        "test_perplexity": model.perplexity(test),
    }


if __name__ == "__main__":
    print(train_and_evaluate())
