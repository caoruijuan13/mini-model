"""PyTorch implementation of the stage 07 causal character Transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from projects.tokenization import CharTokenizer

FORMAT_NAME = "mini-model-char-transformer"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    block_size: int = 8
    model_dim: int = 16
    num_heads: int = 4
    num_layers: int = 2
    feed_forward_dim: int = 32

    def __post_init__(self) -> None:
        for name in (
            "vocab_size", "block_size", "model_dim", "num_heads",
            "num_layers", "feed_forward_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads


class CausalSelfAttention(nn.Module):
    """(B, T, D) -> (B, T, D); no position may read a later position."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        # Stage 07's attention projections have no bias parameters.
        self.qkv = nn.Linear(self.config.model_dim, 3 * self.config.model_dim, bias=False)
        self.output = nn.Linear(self.config.model_dim, self.config.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split heads, scale scores, mask future positions, and combine heads.
        B, T, _ = x.shape
        head_nums = self.config.num_heads
        head_dim = self.config.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, T, head_nums, head_dim).transpose(1, 2)
        k = k.reshape(B, T, head_nums, head_dim).transpose(1, 2)
        v = v.reshape(B, T, head_nums, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        merged = torch.matmul(attention, v)
        combined = merged.transpose(1, 2).reshape(B, T, head_nums * head_dim)
        return self.output(combined)

class TransformerBlock(nn.Module):
    """Pre-LN residual attention followed by a Pre-LN residual feed-forward."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(self.config.model_dim)
        self.norm2 = nn.LayerNorm(self.config.model_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.config.model_dim, self.config.feed_forward_dim),
            nn.ReLU(),
            nn.Linear(self.config.feed_forward_dim, self.config.model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed1 = x + self.attention(self.norm1(x))
        return normed1 + self.mlp(self.norm2(normed1))

class TorchCharTransformer(nn.Module):
    """Token IDs (B, T) -> next-token logits (B, T, vocab_size)."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.model_dim)
        self.position_embedding = nn.Embedding(self.config.block_size, self.config.model_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(self.config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.config.model_dim)
        self.vocab_projection = nn.Linear(self.config.model_dim, self.config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[0] < 1:
            raise ValueError("token_ids must have shape (B, T) with B >= 1")
        sequence_length = token_ids.shape[1]
        if not 1 <= sequence_length <= self.config.block_size:
            raise ValueError("T must be between 1 and block_size")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must have dtype torch.long")
        if token_ids.min() < 0 or token_ids.max() >= self.config.vocab_size:
            raise ValueError("token_ids must be in range [0, vocab_size)")
        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.vocab_projection(self.final_norm(x))
        return logits

    def loss(self, token_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(token_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return loss

    def save(self, path: str | Path, *, tokenizer: CharTokenizer | None = None) -> None:
        """Save inference weights, config, and optionally their exact vocabulary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }
        if tokenizer is not None:
            if tokenizer.vocab_size != self.config.vocab_size:
                raise ValueError("tokenizer vocabulary size does not match model")
            payload["tokenizer"] = self._tokenizer_spec(tokenizer)
        torch.save(payload, path)

    @staticmethod
    def _tokenizer_spec(tokenizer: CharTokenizer) -> dict[str, object]:
        return {
            "tokens": list(tokenizer.tokens),
            "unk_token": tokenizer.unk_token,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
        }

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> TorchCharTransformer:
        config = TransformerConfig(**payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    @classmethod
    def _load_payload(cls, path: str | Path) -> dict[str, Any]:
        """Read and validate the versioned inference-bundle envelope."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("model file must contain an inference-bundle mapping")
        actual_format = payload.get("format")
        if actual_format != FORMAT_NAME:
            raise ValueError(
                f"unsupported model format: expected {FORMAT_NAME!r}, "
                f"got {actual_format!r}"
            )
        actual_version = payload.get("version")
        if actual_version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported model version: expected {FORMAT_VERSION}, "
                f"got {actual_version!r}"
            )
        if "config" not in payload or "state_dict" not in payload:
            raise ValueError("model file is missing config or state_dict")
        return payload

    @classmethod
    def load(
        cls, path: str | Path, *, tokenizer: CharTokenizer | None = None
    ) -> TorchCharTransformer:
        """Load inference weights; reject a supplied mismatched tokenizer."""
        payload = cls._load_payload(path)
        if tokenizer is not None and payload.get("tokenizer") != cls._tokenizer_spec(tokenizer):
            raise ValueError("tokenizer does not match the saved model")
        return cls._from_payload(payload)

    @classmethod
    def load_with_tokenizer(
        cls, path: str | Path
    ) -> tuple[TorchCharTransformer, CharTokenizer]:
        """Load a self-contained inference bundle, including token-ID mapping."""
        payload = cls._load_payload(path)
        spec = payload.get("tokenizer")
        if spec is None:
            raise ValueError("model file does not contain a tokenizer")
        tokenizer = CharTokenizer(
            tokens=tuple(spec["tokens"]),
            unk_token=spec["unk_token"],
            bos_token=spec["bos_token"],
            eos_token=spec["eos_token"],
        )
        model = cls._from_payload(payload)
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError("saved tokenizer vocabulary size does not match model")
        return model, tokenizer
