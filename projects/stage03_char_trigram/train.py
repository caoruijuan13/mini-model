"""使用真实语料训练、评估并保存 trigram 模型。"""

from pathlib import Path

try:
    from .data import REAL_CORPUS, make_char_splits
    from .model import CharTrigram
except ImportError:
    from data import REAL_CORPUS, make_char_splits
    from model import CharTrigram


DEFAULT_TRIGRAM_MODEL_PATH = (
    Path(__file__).with_name("artifacts") / "char_trigram.npz"
)


def train_and_evaluate(
    model_path: str | Path = DEFAULT_TRIGRAM_MODEL_PATH,
) -> dict[str, float]:
    train, valid, test = make_char_splits(REAL_CORPUS)
    vocabulary = "".join(sorted(set(REAL_CORPUS)))
    model = CharTrigram(vocabulary)
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
