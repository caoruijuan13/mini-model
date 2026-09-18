"""Configurable, resumable mini-batch training for stage 06."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from projects.tokenization import CharTokenizer

from checkpoint import load_checkpoint, save_checkpoint
from data import DeterministicBatcher, build_context_targets, load_corpus, make_text_splits
from model import MLPConfig, TokenMLP
from optimizers import Optimizer, make_optimizer


ARTIFACTS_DIR = Path(__file__).with_name("artifacts")
DEFAULT_CHECKPOINT_PATH = ARTIFACTS_DIR / "training_checkpoint.npz"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "token_mlp.npz"
DEFAULT_TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "experiment_report.json"


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 0.002
    batch_size: int = 32
    max_steps: int = 1_500
    eval_interval: int = 25
    patience_evaluations: int = 16
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


@dataclass
class TrainerState:
    step: int = 0
    best_valid_loss: float = float("inf")
    best_step: int = 0
    stale_evaluations: int = 0
    stopped_early: bool = False
    elapsed_seconds: float = 0.0
    history: list[dict[str, float | int]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainerState":
        return cls(**values)


@dataclass(frozen=True)
class TrainingOutcome:
    model: TokenMLP
    optimizer: Optimizer
    state: TrainerState
    best_params: dict[str, np.ndarray]
    completed: bool


def gradient_norm(gradients: dict[str, np.ndarray]) -> float:
    squared = sum(float(np.sum(gradient**2)) for gradient in gradients.values())
    return float(np.sqrt(squared))


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in params.items()}


def _model_from_params(model: TokenMLP, params: dict[str, np.ndarray]) -> TokenMLP:
    result = TokenMLP(model.config)
    result.params = _copy_params(params)
    return result


def train_model(
    *,
    model_config: MLPConfig,
    training_config: TrainingConfig,
    train_data: tuple[np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray],
    checkpoint_path: str | Path,
    resume: bool = False,
    stop_after_steps: int | None = None,
) -> TrainingOutcome:
    """Train or resume without reading test data.

    ``stop_after_steps`` simulates an interruption while preserving the original
    configured ``max_steps``. It is used by the resume-equivalence test.
    """
    if resume:
        (
            model,
            optimizer_name,
            optimizer_state,
            saved_config,
            saved_state,
            best_params,
        ) = load_checkpoint(checkpoint_path)
        if model.config != model_config:
            raise ValueError("checkpoint model config does not match")
        if saved_config != asdict(training_config):
            raise ValueError("checkpoint training config does not match")
        if optimizer_name != training_config.optimizer:
            raise ValueError("checkpoint optimizer does not match")
        optimizer = make_optimizer(optimizer_name, training_config.learning_rate)
        optimizer.load_state_dict(optimizer_state, model.params)
        state = TrainerState.from_dict(saved_state)
    else:
        model = TokenMLP(model_config, seed=training_config.seed)
        optimizer = make_optimizer(
            training_config.optimizer, training_config.learning_rate
        )
        state = TrainerState()
        state.best_valid_loss = model.loss(*valid_data)
        best_params = _copy_params(model.params)

    if optimizer.step_count != state.step:
        raise ValueError("optimizer and trainer step counts differ")
    target_step = state.step if state.stopped_early else training_config.max_steps
    if stop_after_steps is not None:
        if stop_after_steps < state.step:
            raise ValueError("stop_after_steps cannot precede the restored step")
        target_step = min(target_step, stop_after_steps)

    train_contexts, train_targets = train_data
    valid_contexts, valid_targets = valid_data
    batcher = DeterministicBatcher(
        train_contexts,
        train_targets,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    started = perf_counter()
    early_stopped = False
    while state.step < target_step:
        batch_contexts, batch_targets, epoch, _ = batcher.batch_for_step(state.step)
        batch_loss, gradients = model.loss_and_gradients(batch_contexts, batch_targets)
        norm = gradient_norm(gradients)
        optimizer.step(model.params, gradients)
        state.step += 1

        if state.step % training_config.eval_interval == 0 or state.step == training_config.max_steps:
            valid_loss = model.loss(valid_contexts, valid_targets)
            if valid_loss < state.best_valid_loss:
                state.best_valid_loss = valid_loss
                state.best_step = state.step
                state.stale_evaluations = 0
                best_params = _copy_params(model.params)
            else:
                state.stale_evaluations += 1
            state.history.append(
                {
                    "step": state.step,
                    "epoch": epoch,
                    "batch_loss": batch_loss,
                    "valid_loss": valid_loss,
                    "gradient_norm": norm,
                }
            )
            if state.stale_evaluations >= training_config.patience_evaluations:
                state.stopped_early = True
                early_stopped = True
                break

    state.elapsed_seconds += perf_counter() - started
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        training_config=asdict(training_config),
        trainer_state=asdict(state),
        best_params=best_params,
    )
    completed = state.step >= training_config.max_steps or state.stopped_early
    return TrainingOutcome(model, optimizer, state, best_params, completed)


def prepare_corpus() -> tuple[
    CharTokenizer,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    train_text, valid_text, test_text = make_text_splits(load_corpus())
    tokenizer = CharTokenizer.from_text(train_text)
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    datasets = []
    for text in (train_text, valid_text, test_text):
        datasets.append(
            build_context_targets(
                tokenizer.encode(text), context_size=2, bos_id=bos_id, eos_id=eos_id
            )
        )
    return tokenizer, datasets[0], datasets[1], datasets[2]


def train_and_evaluate(
    *,
    training_config: TrainingConfig = TrainingConfig(),
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    resume: bool = False,
) -> dict[str, Any]:
    tokenizer, train_data, valid_data, test_data = prepare_corpus()
    model_config = MLPConfig(vocab_size=tokenizer.vocab_size)
    outcome = train_model(
        model_config=model_config,
        training_config=training_config,
        train_data=train_data,
        valid_data=valid_data,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
    inference_model = _model_from_params(outcome.model, outcome.best_params)
    inference_model.save(model_path)
    tokenizer.save(tokenizer_path)
    report = {
        "training_config": asdict(training_config),
        "best_step": outcome.state.best_step,
        "completed_step": outcome.state.step,
        "early_stopped": outcome.state.stopped_early,
        "train_examples": len(train_data[1]),
        "valid_examples": len(valid_data[1]),
        "test_examples": len(test_data[1]),
        "train_loss": inference_model.loss(*train_data),
        "valid_loss": inference_model.loss(*valid_data),
        "test_loss": inference_model.loss(*test_data),
        "train_perplexity": float(np.exp(inference_model.loss(*train_data))),
        "valid_perplexity": float(np.exp(inference_model.loss(*valid_data))),
        "test_perplexity": float(np.exp(inference_model.loss(*test_data))),
        "elapsed_seconds": outcome.state.elapsed_seconds,
        "history": outcome.state.history,
    }
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default=defaults.optimizer)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument(
        "--patience-evaluations", type=int, default=defaults.patience_evaluations
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        patience_evaluations=args.patience_evaluations,
        seed=args.seed,
    )
    print(json.dumps(train_and_evaluate(training_config=config, resume=args.resume), indent=2))
