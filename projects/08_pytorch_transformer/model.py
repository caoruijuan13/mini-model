"""PyTorch model exercise: fill the modules and forward paths yourself."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


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
        # register Q/K/V and output projections as nn.Module children.
        self.qkv = nn.Linear(self.config.model_dim, 3 * self.config.model_dim)
        self.output = nn.Linear(self.config.model_dim, self.config.model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # split heads, scale scores, apply causal mask, combine heads.
        # You may use torch.nn.functional.scaled_dot_product_attention with
        # is_causal=True, or implement the forward calculation explicitly.
        B, T, D = x.shape
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
        # register two LayerNorms, attention, and a feed-forward MLP.
        self.norm1 = nn.LayerNorm(self.config.model_dim)
        self.norm2 = nn.LayerNorm(self.config.model_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.config.model_dim, self.config.feed_forward_dim),
            nn.ReLU(),
            nn.Linear(self.config.feed_forward_dim, self.config.model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x + attention(norm1(x)); then x + mlp(norm2(x)).
        normed1 = x + self.attention(self.norm1(x))
        return normed1 + self.mlp(self.norm2(normed1))

class TorchCharTransformer(nn.Module):
    """Token IDs (B, T) -> next-token logits (B, T, vocab_size)."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        # register token/position Embedding, ModuleList of blocks,
        # final LayerNorm, and vocabulary projection.
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.model_dim)
        self.position_embedding = nn.Embedding(self.config.block_size, self.config.model_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(self.config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.config.model_dim)
        self.vocab_projection = nn.Linear(self.config.model_dim, self.config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # validate shape/range, add token and position embeddings,
        # apply all blocks, and return logits (not softmax probabilities).
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
        # validate shape/range, add token and position embeddings,
        # apply all blocks, project to logits, and return loss.
        logits = self.forward(token_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return loss

    def save(self, path: str | Path) -> None:
        # save config and state_dict for inference; document tokenizer
        # matching. Do not save a pickled whole model object.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> TorchCharTransformer:
        # reconstruct config/model and load state_dict.
        checkpoint = torch.load(path, map_location="cpu")
        config = TransformerConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
