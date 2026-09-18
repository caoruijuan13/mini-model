"""加载 trigram 模型并使用改进采样策略生成文本。"""

from pathlib import Path

try:
    from .train import DEFAULT_TRIGRAM_MODEL_PATH
    from .model import CharTrigram
except ImportError:
    from train import DEFAULT_TRIGRAM_MODEL_PATH
    from model import CharTrigram


def generate(
    length: int = 100,
    seed: int = 7,
    model_path: str | Path = DEFAULT_TRIGRAM_MODEL_PATH,
    start: str = "We",
    temperature: float = 0.8,
    top_k: int | None = 5,
) -> str:
    model = CharTrigram.load(model_path)
    return model.sample_new(
        start=start,
        length=length,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
    )


if __name__ == "__main__":
    print(generate())
