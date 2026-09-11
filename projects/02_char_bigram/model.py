"""阶段二：字符级 bigram 模型。"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class CharBigram:
    # vocab 是模型可以处理和生成的全部字符集合。
    vocab: str
    # 平滑系数用于避免未出现的字符转移概率为 0。
    smoothing: float = 1e-3

    def __post_init__(self) -> None:
        # dataclass 的 __init__ 完成字段赋值后，自动执行这里的初始化逻辑。
        self.to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.counts: np.ndarray | None = np.full(
            (len(self.vocab), len(self.vocab)), self.smoothing
        )
        self.probs: np.ndarray | None = None

    def fit(self, text: str) -> None:
        # fit 才是真正读取语料、统计 bigram 并生成条件概率表的地方。
        unknown = sorted(set(text) - set(self.to_id))
        if unknown:
            raise ValueError(f"text contains characters outside the vocabulary: {unknown}")
        if self.counts is None:
            self.counts = np.full((len(self.vocab), len(self.vocab)), self.smoothing)
        self.counts.fill(self.smoothing)
        ids = [self.to_id[c] for c in text]
        for current, following in zip(ids, ids[1:]):
            self.counts[current, following] += 1
        self.probs = self.counts / self.counts.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> None:
        if self.probs is None:
            raise RuntimeError("fit the model before saving it")
        # 推理阶段只需要词表和概率；counts 只属于训练过程，不写入推理模型。
        np.savez(path, vocab=self.vocab, probs=self.probs)

    @classmethod
    def load(cls, path: str | Path) -> "CharBigram":
        with np.load(path, allow_pickle=False) as data:
            vocab = str(data["vocab"].item())
            model = cls(vocab)
            model.counts = None
            model.probs = data["probs"].copy()
        return model

    def loss(self, text: str) -> float:
        if self.probs is None:
            raise RuntimeError("fit or load the model before evaluating it")
        unknown = sorted(set(text) - set(self.to_id))
        if unknown:
            raise ValueError(f"text contains characters outside the vocabulary: {unknown}")
        ids = [self.to_id[c] for c in text]
        if len(ids) < 2:
            raise ValueError("text must contain two known characters")
        return float(-np.mean(np.log(self.probs[ids[:-1], ids[1:]])))

    def perplexity(self, text: str) -> float:
        return float(np.exp(self.loss(text)))

    def sample(self, start: str = "s", length: int = 100, seed: int = 7) -> str:
        if self.probs is None:
            raise RuntimeError("fit or load the model before sampling")
        if start not in self.to_id:
            raise ValueError("start character is outside the vocabulary")
        rng = np.random.default_rng(seed)
        result = [start]
        for _ in range(max(0, length - 1)):
            row = self.probs[self.to_id[result[-1]]]
            result.append(self.vocab[rng.choice(len(self.vocab), p=row)])
        return "".join(result)

    def sample_new(
        self,
        start: str = "s",
        length: int = 100,
        seed: int = 7,
        temperature: float = 0.8,
        top_k: int | None = 5,
    ) -> str:
        """使用 temperature 和 top-k 的改进采样，不改变原 sample。"""
        if self.probs is None:
            raise RuntimeError("fit or load the model before sampling")
        if start not in self.to_id:
            raise ValueError("start character is outside the vocabulary")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and not 1 <= top_k <= len(self.vocab):
            raise ValueError("top_k must be between 1 and vocabulary size")

        rng = np.random.default_rng(seed)
        result = [start]
        for _ in range(max(0, length - 1)):
            row = self.probs[self.to_id[result[-1]]]
            adjusted = np.log(row) / temperature
            if top_k is None:
                candidate_ids = np.arange(len(self.vocab))
            else:
                candidate_ids = np.argpartition(adjusted, -top_k)[-top_k:]
            candidate_probs = np.exp(adjusted[candidate_ids] - adjusted[candidate_ids].max())
            candidate_probs /= candidate_probs.sum()
            next_id = rng.choice(candidate_ids, p=candidate_probs)
            result.append(self.vocab[next_id])
        return "".join(result)
