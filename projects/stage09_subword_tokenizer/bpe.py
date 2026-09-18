"""Learner-owned minimal character-initialized BPE implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch

Pair = tuple[int, int]
UnknownPolicy = Literal["use_unk", "error"]
SPECIAL_TOKENS = ("<UNK>", "<BOS>", "<EOS>")


@dataclass(frozen=True)
class BPEConfig:
    max_merges: int = 40
    min_pair_frequency: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.max_merges, bool) or not isinstance(self.max_merges, int) or self.max_merges < 0:
            raise ValueError("max_merges must be a non-negative integer")
        if isinstance(self.min_pair_frequency, bool) or not isinstance(self.min_pair_frequency, int) or self.min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be a positive integer")


def count_pairs(token_ids: Sequence[int]) -> dict[Pair, int]:
    """count each adjacent pair, including overlapping occurrences."""
    pairs: dict[Pair, int] = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

def select_best_pair(
    counts: dict[Pair, int], *, min_frequency: int
) -> Pair | None:
    """highest count first; ties use smallest (left_id, right_id)."""
    # sorted_counts = sorted(counts.items(), key=lambda pair_count: (-pair_count[1], pair_count[0]))
    best_pair = None
    best_pair_count = min_frequency
    for pair, count in counts.items():
        if count < min_frequency:
            continue
        if best_pair is None or count > best_pair_count:
            best_pair = pair
            best_pair_count = count
        elif count == best_pair_count and pair < best_pair:
            best_pair = pair
    return best_pair

def merge_pair(
    token_ids: Sequence[int], pair: Pair, new_token_id: int
) -> list[int]:
    """replace non-overlapping matches from left to right."""
    new_token_ids = []
    i = 0
    while i < len(token_ids):
        if i+1 < len(token_ids) and (token_ids[i], token_ids[i+1]) == pair:
            new_token_ids.append(new_token_id)
            i += 2
        else:
            new_token_ids.append(token_ids[i])
            i += 1
    return new_token_ids


class BPETokenizer:
    """Base tokens plus ordered merges; merge i creates ID len(base_tokens)+i."""

    FORMAT_NAME = "mini-model-char-bpe"
    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        base_tokens: Sequence[str],
        tokens: Sequence[str],
        merges: Sequence[Pair],
        unk_token: str = SPECIAL_TOKENS[0],
        bos_token: str = SPECIAL_TOKENS[1],
        eos_token: str = SPECIAL_TOKENS[2],
    ) -> None:
        self.base_tokens = tuple(base_tokens)
        self.tokens = tuple(tokens)
        self.merges = tuple(merges)
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        if any(not isinstance(token, str) or not token for token in self.tokens):
            raise ValueError("tokens must be non-empty strings")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("tokens must be unique")
        if self.tokens[:len(self.base_tokens)] != self.base_tokens:
            raise ValueError("tokens must start with base_tokens in the same order")

        special_tokens = (self.unk_token, self.bos_token, self.eos_token)
        if any(not isinstance(token, str) or not token for token in special_tokens):
            raise ValueError("special tokens must be non-empty strings")
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special tokens must be distinct")
        if any(token not in self.base_tokens for token in special_tokens):
            raise ValueError("special tokens must be present in base_tokens")

        if len(self.tokens) != len(self.base_tokens) + len(self.merges):
            raise ValueError("each merge must create exactly one token")
        for new_id, pair in enumerate(self.merges, start=len(self.base_tokens)):
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in pair)
            ):
                raise ValueError("each merge must be a pair of integer IDs")
            left_id, right_id = pair
            if not (0 <= left_id < new_id and 0 <= right_id < new_id):
                raise ValueError("merge IDs must refer to previously defined tokens")
            if self.tokens[new_id] != self.tokens[left_id] + self.tokens[right_id]:
                raise ValueError("merged token must equal the concatenation of its pair")

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def train(
        cls, train_text: str, *, config: BPEConfig = BPEConfig()
    ) -> BPETokenizer:
        """initialize from train-only characters, learn ordered merges."""
        char_tokens = tuple(sorted(set(train_text)))
        base_tokens = SPECIAL_TOKENS + char_tokens
        token_ids = [base_tokens.index(char) for char in train_text]
        tokens = base_tokens
        merges: list[Pair] = []
        while len(merges) < config.max_merges:
            counts = count_pairs(token_ids)
            pair = select_best_pair(counts, min_frequency=config.min_pair_frequency)
            if not pair:
                break
            merges.append(pair)
            merge = merge_pair(token_ids, pair, len(tokens))
            if merge:
                new_token = tokens[pair[0]] + tokens[pair[1]]
                tokens += (new_token,)
            token_ids = merge
        
        return cls(
            base_tokens=base_tokens,
            tokens=tokens,
            merges=merges,
            unk_token=base_tokens[0],
            bos_token=base_tokens[1],
            eos_token=base_tokens[2],
        )

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        on_unknown: UnknownPolicy = "use_unk",
    ) -> list[int]:
        """character IDs, then apply learned merges in rank order."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if on_unknown not in ("use_unk", "error"):
            raise ValueError("on_unknown must be 'use_unk' or 'error'")
        token_ids = []
        for char in text:
            if char not in self.base_tokens:
                if on_unknown == "use_unk":
                    token_ids.append(self.base_tokens.index(self.unk_token))
                else:
                    raise ValueError(f"unknown token {char}")
            else:
                token_ids.append(self.base_tokens.index(char))
        for i in range(len(self.merges)):
            pair = self.merges[i]
            token_ids = merge_pair(token_ids, pair, len(self.base_tokens) + i)

        if add_bos:
            token_ids = [self.base_tokens.index(self.bos_token)] + token_ids
        if add_eos:
            token_ids = token_ids + [self.base_tokens.index(self.eos_token)]
        return token_ids

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        """join token strings; state when UNK loses information."""

        result = ""
        unk_token_id = self.base_tokens.index(self.unk_token)
        bos_token_id = self.base_tokens.index(self.bos_token)
        eos_token_id = self.base_tokens.index(self.eos_token)
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id >= len(self.tokens) or token_id < 0:
                raise ValueError("invalid token ID")
            if token_id in (unk_token_id, bos_token_id, eos_token_id):
                if skip_special_tokens:
                    continue
                if token_id == unk_token_id:
                    result += self.unk_token
                elif token_id == bos_token_id:
                    result += self.bos_token
                elif token_id == eos_token_id:
                    result += self.eos_token
            else:
                result += self.tokens[token_id]
        return result

    def save(self, path: str | Path) -> None:
        """save version, base/final vocab, merges, special tokens."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT_NAME,
            "version": self.FORMAT_VERSION,
            "base_tokens": self.base_tokens,
            "tokens": self.tokens,
            "merges": self.merges,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """load and validate a versioned tokenizer."""
        torch_payload = torch.load(path)
        if torch_payload["format"] != cls.FORMAT_NAME:
            raise ValueError("unsupported tokenizer format")
        if torch_payload["version"] != cls.FORMAT_VERSION:
            raise ValueError("unsupported tokenizer version")
        return cls(
            base_tokens=tuple(torch_payload["base_tokens"]),
            tokens=tuple(torch_payload["tokens"]),
            merges=tuple(torch_payload["merges"]),
            unk_token=torch_payload["unk_token"],
            bos_token=torch_payload["bos_token"],
            eos_token=torch_payload["eos_token"],
        )