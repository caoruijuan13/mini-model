"""Versioned training checkpoints, separate from inference model files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from model import MLPConfig, TokenMLP
from optimizers import Optimizer


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: TokenMLP,
    optimizer: Optimizer,
    training_config: dict[str, Any],
    trainer_state: dict[str, Any],
    best_params: dict[str, np.ndarray],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": CHECKPOINT_VERSION,
        "model_config": {
            "vocab_size": model.config.vocab_size,
            "context_size": model.config.context_size,
            "embedding_dim": model.config.embedding_dim,
            "hidden_dim": model.config.hidden_dim,
        },
        "optimizer": optimizer.name,
        "training_config": training_config,
        "trainer_state": trainer_state,
    }
    arrays: dict[str, np.ndarray] = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    arrays.update({f"model.{name}": value for name, value in model.params.items()})
    arrays.update({f"best.{name}": value for name, value in best_params.items()})
    arrays.update(
        {f"optimizer.{name}": value for name, value in optimizer.state_dict().items()}
    )
    np.savez(output_path, **arrays)


def load_checkpoint(
    path: str | Path,
) -> tuple[
    TokenMLP,
    str,
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version: {metadata.get('version')}")
        model = TokenMLP(MLPConfig(**metadata["model_config"]))
        for name in model.PARAMETER_NAMES:
            model.params[name] = data[f"model.{name}"].copy()
        best_params = {
            name: data[f"best.{name}"].copy() for name in model.PARAMETER_NAMES
        }
        optimizer_state = {
            key.removeprefix("optimizer."): data[key].copy()
            for key in data.files
            if key.startswith("optimizer.")
        }
    return (
        model,
        metadata["optimizer"],
        optimizer_state,
        metadata["training_config"],
        metadata["trainer_state"],
        best_params,
    )
