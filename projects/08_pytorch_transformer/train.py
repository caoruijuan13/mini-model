"""Stage 08 training exercise: use autograd instead of manual backward."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from importlib import import_module
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import TorchCharTransformer, TransformerConfig
from .data import prepare_corpus
from .generate import generate_ids

DeterministicBatcher = import_module(
    "projects.06_training_engineering.data"
).DeterministicBatcher

ARTIFACTS_DIR = Path(__file__).with_name("artifacts")
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "char_transformer.pt"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "experiment_report.json"

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.002
    batch_size: int = 16
    max_steps: int = 1000
    eval_interval: int = 25
    patience_evaluations: int = 12
    seed: int = 7

    def __post_init__(self) -> None:
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
    model: TorchCharTransformer  # Restore best validation weights before returning.
    initial_train_loss: float
    initial_valid_loss: float
    best_valid_loss: float
    best_step: int
    completed_step: int
    stopped_early: bool


def train_model(
    *,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    train_data: tuple[np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray],
) -> TrainingOutcome:
    """Train on train_data, select on valid_data; never inspect test_data here."""
    # convert NumPy IDs to torch.long, create model and optimizer.
    # For each batch: optimizer.zero_grad(), forward, cross_entropy on
    # flattened logits/targets, loss.backward(), optimizer.step().
    # Evaluate under torch.no_grad() in model.eval() mode. Copy the best
    # state_dict and restore it before returning.
    torch.manual_seed(training_config.seed)
    model = TorchCharTransformer(model_config)
    best_params = model.state_dict()
    
    initial_train_loss = float(model.loss(*to_tensors(train_data)).item())
    initial_valid_loss = float(model.loss(*to_tensors(valid_data)).item())
    best_valid_loss = initial_valid_loss
    best_step = 0
    stale_count = 0
    stopped_early = False

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
    )

    batcher = DeterministicBatcher(
        *train_data, batch_size=training_config.batch_size, seed=training_config.seed
    )
    
    for step in range(1, training_config.max_steps+1):
        model.train()
        # zero out gradients of above step
        optimizer.zero_grad(set_to_none=True)
        contexts, targets, _, _ = batcher.batch_for_step(step - 1)
        loss = model.loss(torch.from_numpy(contexts), torch.from_numpy(targets))
        # get gradients, and save to model params
        loss.backward()
        # update model params by optimizer with gradients
        optimizer.step()

        if step % training_config.eval_interval != 0 and step != training_config.max_steps:
            continue

        model.eval()
        with torch.no_grad():
            valid_loss = float(model.loss(*to_tensors(valid_data)).item())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_step = step
            stale_count = 0
            best_params = copy.deepcopy(model.state_dict())
        else:
            stale_count += 1

        if stale_count >= training_config.patience_evaluations:
            stopped_early = True
            break

    model.load_state_dict(best_params)

    return TrainingOutcome(
        model=model,
        initial_train_loss=initial_train_loss,
        initial_valid_loss=initial_valid_loss,
        best_valid_loss=best_valid_loss,
        best_step=best_step,
        completed_step=step,
        stopped_early=stopped_early,
    )

def to_tensors(
    data: tuple[np.ndarray, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = data
    return torch.from_numpy(inputs).long(), torch.from_numpy(targets).long()

def train_and_evaluate(
    *,
    model_config: TransformerConfig | None = None,
    training_config: TrainingConfig = TrainingConfig(),
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    block_size = model_config.block_size if model_config is not None else 8
    tokenizer, train_data, valid_data, test_data = prepare_corpus(block_size=block_size)

    if model_config is None:
        model_config = TransformerConfig(vocab_size=tokenizer.vocab_size, block_size=block_size)
    elif model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary size does not match the training tokenizer")
    
    outcome = train_model(
        model_config=model_config,
        training_config=training_config,
        train_data=train_data,
        valid_data=valid_data,
    )
    model = outcome.model
    model.eval()
    with torch.no_grad():
        train_loss = float(model.loss(*to_tensors(train_data)).item())
        valid_loss = float(model.loss(*to_tensors(valid_data)).item())
        test_loss = float(model.loss(*to_tensors(test_data)).item())
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
        "initial_train_loss": float(outcome.initial_train_loss),
        "initial_valid_loss": float(outcome.initial_valid_loss),
        "best_valid_loss": float(outcome.best_valid_loss),
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
    }
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

if __name__ == "__main__":
    report = train_and_evaluate()
    print(json.dumps(report, ensure_ascii=False, indent=2))
