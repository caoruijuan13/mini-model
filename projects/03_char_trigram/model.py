"""阶段三：字符级 trigram 模型。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CharTrigram:
    """根据前两个字符预测下一个字符的计数语言模型。"""

    vocab: str
    # 两字符上下文更稀疏；该值由真实语料验证集候选比较选定。
    smoothing: float = 0.1

    def __post_init__(self) -> None:
        self.to_id = {ch: i for i, ch in enumerate(self.vocab)}
        size = len(self.vocab)
        self.counts: np.ndarray | None = np.full(
            (size, size, size), self.smoothing
        )
        self.probs: np.ndarray | None = None
        self.backoff_counts: np.ndarray | None = np.full(
            (size, size), self.smoothing
        )
        self.backoff_probs: np.ndarray | None = None
        self.context_seen = np.zeros((size, size), dtype=bool)

    def fit(self, text: str) -> None:
        unknown = sorted(set(text) - set(self.to_id))
        if unknown:
            raise ValueError(f"text contains characters outside the vocabulary: {unknown}")
        if len(text) < 3:
            raise ValueError("text must contain at least three characters")
        if self.counts is None:
            size = len(self.vocab)
            self.counts = np.full((size, size, size), self.smoothing)
        if self.backoff_counts is None:
            size = len(self.vocab)
            self.backoff_counts = np.full((size, size), self.smoothing)

        self.counts.fill(self.smoothing)
        self.backoff_counts.fill(self.smoothing)
        self.context_seen.fill(False)
        ids = [self.to_id[ch] for ch in text]
        for first, second, following in zip(ids, ids[1:], ids[2:]):
            self.counts[first, second, following] += 1
            self.context_seen[first, second] = True
        for current, following in zip(ids, ids[1:]):
            self.backoff_counts[current, following] += 1

        # 每个 [first, second] 上下文对应一个下一字符概率分布。
        self.probs = self.counts / self.counts.sum(axis=2, keepdims=True)
        self.backoff_probs = self.backoff_counts / self.backoff_counts.sum(
            axis=1, keepdims=True
        )

    def loss(self, text: str) -> float:
        if self.probs is None:
            raise RuntimeError("fit or load the model before evaluating it")
        unknown = sorted(set(text) - set(self.to_id))
        if unknown:
            raise ValueError(f"text contains characters outside the vocabulary: {unknown}")
        if len(text) < 3:
            raise ValueError("text must contain at least three characters")

        ids = [self.to_id[ch] for ch in text]
        probabilities = self.probs[ids[:-2], ids[1:-1], ids[2:]]
        return float(-np.mean(np.log(probabilities)))

    def perplexity(self, text: str) -> float:
        return float(np.exp(self.loss(text)))

    def sample_new(
        self,
        start: str = "We",
        length: int = 100,
        seed: int = 7,
        temperature: float = 0.8,
        top_k: int | None = 5,
    ) -> str:
        """根据两个字符的上下文，使用 temperature 和 top-k 采样。"""
        if self.probs is None:
            raise RuntimeError("fit or load the model before sampling")
        if len(start) != 2:
            raise ValueError("start must contain exactly two characters")
        unknown = sorted(set(start) - set(self.to_id))
        if unknown:
            raise ValueError(f"start contains characters outside the vocabulary: {unknown}")
        if length < 2:
            raise ValueError("length must be at least two")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and not 1 <= top_k <= len(self.vocab):
            raise ValueError("top_k must be between 1 and vocabulary size")

        rng = np.random.default_rng(seed)
        result = list(start)
        for _ in range(length - 2):
            first = self.to_id[result[-2]]
            second = self.to_id[result[-1]]
            if self.context_seen[first, second]:
                row = self.probs[first, second]
            else:
                # 训练中没见过该两字符上下文时，退回最后一个字符的 bigram 分布。
                row = self.backoff_probs[second]
            adjusted = np.log(row) / temperature
            if top_k is None:
                candidate_ids = np.arange(len(self.vocab))
            else:
                candidate_ids = np.argpartition(adjusted, -top_k)[-top_k:]
            candidate_probs = np.exp(
                adjusted[candidate_ids] - adjusted[candidate_ids].max()
            )
            candidate_probs /= candidate_probs.sum()
            next_id = rng.choice(candidate_ids, p=candidate_probs)
            result.append(self.vocab[next_id])
        return "".join(result)

    def save(self, path: str | Path) -> None:
        if self.probs is None:
            raise RuntimeError("fit the model before saving it")
        np.savez(
            path,
            vocab=self.vocab,
            probs=self.probs,
            backoff_probs=self.backoff_probs,
            context_seen=self.context_seen,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTrigram":
        with np.load(path, allow_pickle=False) as data:
            model = cls(str(data["vocab"].item()))
            model.counts = None
            model.backoff_counts = None
            model.probs = data["probs"].copy()
            model.backoff_probs = data["backoff_probs"].copy()
            model.context_seen = data["context_seen"].copy()
        return model
