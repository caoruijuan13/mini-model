"""Train and evaluate the character Transformer on fixed corpus splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib import import_module
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .data import prepare_corpus
from .generate import generate_ids
from .model import CharTransformer, TransformerConfig


# Reuse stage 06 batch/optimizer mechanics without duplicating their semantics.
DeterministicBatcher = import_module(
    "projects.06_training_engineering.data"
).DeterministicBatcher
make_optimizer = import_module(
    "projects.06_training_engineering.optimizers"
).make_optimizer

ARTIFACTS_DIR = Path(__file__).with_name("artifacts")
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "char_transformer.npz"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "experiment_report.json"


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 0.002
    batch_size: int = 16
    max_steps: int = 1000
    eval_interval: int = 25
    patience_evaluations: int = 12
    seed: int = 7

    def __post_init__(self) -> None:
        if self.optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        for name in ("batch_size", "max_steps", "eval_interval", "patience_evaluations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class TrainingOutcome:
    model: CharTransformer  # Best validation parameters are restored in place.
    initial_train_loss: float
    initial_valid_loss: float
    best_valid_loss: float
    best_step: int
    completed_step: int
    stopped_early: bool
    elapsed_seconds: float
    history: list[dict[str, float | int]]


def train_model(
    *,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    train_data: tuple[np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray],
) -> TrainingOutcome:
    """Select model parameters using train/valid only; never read test data."""
    model = CharTransformer(model_config, seed=training_config.seed)
    initial_train_loss = model.loss(*train_data)
    initial_valid_loss = model.loss(*valid_data)
    best_valid_loss = initial_valid_loss
    best_step = 0
    best_params = {
        name: value.copy() for name, value in model.named_parameters().items()
    }
    optimizer = make_optimizer(training_config.optimizer, training_config.learning_rate)
    batcher = DeterministicBatcher(
        *train_data, batch_size=training_config.batch_size, seed=training_config.seed
    )
    history: list[dict[str, float | int]] = []
    stale_evaluations = 0
    stopped_early = False
    completed_step = 0
    started = perf_counter()

    for step in range(1, training_config.max_steps + 1):
        batch_inputs, batch_targets, epoch, _ = batcher.batch_for_step(step - 1)
        batch_loss, gradients = model.loss_and_gradients(batch_inputs, batch_targets)
        gradient_norm = float(np.sqrt(sum(
            np.sum(gradient**2) for gradient in gradients.values()
        )))
        optimizer.step(model.named_parameters(), gradients)
        completed_step = step

        if step % training_config.eval_interval != 0 and step != training_config.max_steps:
            continue
        train_loss = model.loss(*train_data)
        valid_loss = model.loss(*valid_data)
        if not np.isfinite(train_loss) or not np.isfinite(valid_loss):
            raise FloatingPointError("non-finite training or validation loss")
        history.append({
            "step": step,
            "epoch": epoch,
            "batch_loss": batch_loss,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "gradient_norm": gradient_norm,
        })
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_step = step
            stale_evaluations = 0
            best_params = {
                name: value.copy() for name, value in model.named_parameters().items()
            }
        else:
            stale_evaluations += 1
        if stale_evaluations >= training_config.patience_evaluations:
            stopped_early = True
            break

    elapsed_seconds = perf_counter() - started
    for name, value in model.named_parameters().items():
        np.copyto(value, best_params[name])
    return TrainingOutcome(
        model=model,
        initial_train_loss=initial_train_loss,
        initial_valid_loss=initial_valid_loss,
        best_valid_loss=best_valid_loss,
        best_step=best_step,
        completed_step=completed_step,
        stopped_early=stopped_early,
        elapsed_seconds=elapsed_seconds,
        history=history,
    )


def train_and_evaluate(
    *,
    model_config: TransformerConfig | None = None,
    training_config: TrainingConfig = TrainingConfig(),
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Run the fixed split, then evaluate test once on validation-selected weights."""
    block_size = model_config.block_size if model_config is not None else 8
    tokenizer, train_data, valid_data, test_data = prepare_corpus(block_size=block_size)
    if model_config is None:
        model_config = TransformerConfig(vocab_size=tokenizer.vocab_size)
    elif model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary size does not match the training tokenizer")
    outcome = train_model(
        model_config=model_config,
        training_config=training_config,
        train_data=train_data,
        valid_data=valid_data,
    )
    model = outcome.model
    train_loss = model.loss(*train_data)
    valid_loss = model.loss(*valid_data)
    test_loss = model.loss(*test_data)
    model.save(model_path)
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    generated_ids = generate_ids(
        model, [bos_id], eos_id=eos_id, max_new_tokens=80,
        seed=training_config.seed, top_k=5,
    )
    report: dict[str, Any] = {
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "initial_train_loss": outcome.initial_train_loss,
        "initial_valid_loss": outcome.initial_valid_loss,
        "best_step": outcome.best_step,
        "completed_step": outcome.completed_step,
        "stopped_early": outcome.stopped_early,
        "train_examples": len(train_data[0]),
        "valid_examples": len(valid_data[0]),
        "test_examples": len(test_data[0]),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "test_loss": test_loss,
        "train_perplexity": float(np.exp(train_loss)),
        "valid_perplexity": float(np.exp(valid_loss)),
        "test_perplexity": float(np.exp(test_loss)),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "elapsed_seconds": outcome.elapsed_seconds,
        "history": outcome.history,
    }
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default=defaults.optimizer)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--patience-evaluations", type=int, default=defaults.patience_evaluations)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()
    tokenizer, _, _, _ = prepare_corpus(block_size=args.block_size)
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, block_size=args.block_size
    )
    training_config = TrainingConfig(
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        patience_evaluations=args.patience_evaluations,
        seed=args.seed,
    )
    report = train_and_evaluate(
        model_config=model_config, training_config=training_config
    )
    print(json.dumps({
        "best_step": report["best_step"],
        "completed_step": report["completed_step"],
        "train_loss": report["train_loss"],
        "valid_loss": report["valid_loss"],
        "test_loss": report["test_loss"],
        "generated_text": report["generated_text"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
