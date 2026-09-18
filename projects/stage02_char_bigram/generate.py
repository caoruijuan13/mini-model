"""阶段二的文本生成入口：只加载已训练模型。"""

from pathlib import Path

try:
    from .model import CharBigram
    from .train import DEFAULT_MODEL_PATH
except ImportError:
    from model import CharBigram
    from train import DEFAULT_MODEL_PATH


def generate(
    length: int = 100,
    seed: int = 7,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    start: str = "s",
) -> str:
    model = CharBigram.load(model_path)
    return model.sample(start=start, length=length, seed=seed)


if __name__ == "__main__":
    print(generate())
