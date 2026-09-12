"""Deterministic character tokenizer used by stage 04 and later projects."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping


UnknownPolicy = Literal["use_unk", "error"]


@dataclass(frozen=True)
class CharTokenizer:
    """Map Unicode characters to stable integer IDs and back.

    Character tokens and special tokens share one vocabulary. The tokenizer is
    immutable after construction so that a model cannot silently observe a
    different token-to-ID mapping during training and inference.
    """

    tokens: tuple[str, ...]
    unk_token: str | None = "<UNK>"
    bos_token: str | None = "<BOS>"
    eos_token: str | None = "<EOS>"
    token_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)
    id_to_token: tuple[str, ...] = field(init=False)

    FORMAT_NAME = "mini-model-char-tokenizer"
    FORMAT_VERSION = 1

    def __post_init__(self) -> None:
        if any(not isinstance(token, str) or not token for token in self.tokens):
            raise ValueError("every vocabulary token must be a non-empty string")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")

        mapping = {token: token_id for token_id, token in enumerate(self.tokens)}
        for name, token in (
            ("unk_token", self.unk_token),
            ("bos_token", self.bos_token),
            ("eos_token", self.eos_token),
        ):
            if token is not None and token not in mapping:
                raise ValueError(f"{name} must be present in the vocabulary")

        special_tokens = [
            token
            for token in (self.unk_token, self.bos_token, self.eos_token)
            if token is not None
        ]
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special tokens must be distinct")

        object.__setattr__(self, "token_to_id", MappingProxyType(mapping))
        object.__setattr__(self, "id_to_token", self.tokens)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        unk_token: str | None = "<UNK>",
        bos_token: str | None = "<BOS>",
        eos_token: str | None = "<EOS>",
    ) -> "CharTokenizer":
        """Build a deterministic vocabulary from training text only."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        special_tokens = tuple(
            token
            for token in (unk_token, bos_token, eos_token)
            if token is not None
        )
        if any(not isinstance(token, str) or not token for token in special_tokens):
            raise ValueError("special tokens must be non-empty strings")
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special tokens must be distinct")

        character_tokens = tuple(sorted(set(text) - set(special_tokens)))

        return cls(
            tokens=special_tokens + character_tokens,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return tuple(
            token
            for token in (self.unk_token, self.bos_token, self.eos_token)
            if token is not None
        )

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        on_unknown: UnknownPolicy = "use_unk",
    ) -> list[int]:
        """Convert text to IDs with an explicit unknown-character policy."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if on_unknown not in ("use_unk", "error"):
            raise ValueError("on_unknown must be 'use_unk' or 'error'")

        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self._required_special_id(self.bos_token, "BOS"))

        for character in text:
            token_id = self.token_to_id.get(character)
            if token_id is not None:
                token_ids.append(token_id)
                continue
            if on_unknown == "error":
                raise ValueError(f"unknown character: {character!r}")
            token_ids.append(self._required_special_id(self.unk_token, "UNK"))

        if add_eos:
            token_ids.append(self._required_special_id(self.eos_token, "EOS"))
        return token_ids

    def decode(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Convert token IDs back to their text representation."""
        special_tokens = set(self.special_tokens)
        decoded: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID outside vocabulary: {token_id}")
            token = self.id_to_token[token_id]
            if skip_special_tokens and token in special_tokens:
                continue
            decoded.append(token)
        return "".join(decoded)

    def save(self, path: str | Path) -> None:
        """Save a versioned, human-readable tokenizer artifact."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT_NAME,
            "version": self.FORMAT_VERSION,
            "tokens": list(self.tokens),
            "special_tokens": {
                "unk": self.unk_token,
                "bos": self.bos_token,
                "eos": self.eos_token,
            },
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load and validate a tokenizer artifact."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload["format"] != cls.FORMAT_NAME:
                raise ValueError("unsupported tokenizer format")
            if payload["version"] != cls.FORMAT_VERSION:
                raise ValueError("unsupported tokenizer version")
            tokens = payload["tokens"]
            special_tokens = payload["special_tokens"]
            if not isinstance(tokens, list):
                raise ValueError("tokenizer tokens must be a list")
            return cls(
                tokens=tuple(tokens),
                unk_token=special_tokens["unk"],
                bos_token=special_tokens["bos"],
                eos_token=special_tokens["eos"],
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid tokenizer artifact") from exc

    def _required_special_id(self, token: str | None, name: str) -> int:
        if token is None:
            raise ValueError(f"{name} token is not configured")
        return self.token_to_id[token]
