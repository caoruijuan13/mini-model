"""Full-batch training skeleton for stage 05."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from projects.tokenization import CharTokenizer

from data import build_context_targets, load_corpus, make_text_splits
from model import MLPConfig, TokenMLP


DEFAULT_MODEL_PATH = Path(__file__).with_name("artifacts") / "token_mlp.npz"
DEFAULT_TOKENIZER_PATH = Path(__file__).with_name("artifacts") / "tokenizer.json"
TRAINING_STEPS = 2_000
LEARNING_RATE = 0.1
RANDOM_SEED = 7


def train_and_evaluate(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
) -> dict[str, float | int]:
    """Train, select with validation data, then evaluate the sealed test data."""
    # 1. split text before building the tokenizer;
    # 2. build the tokenizer from train text only;
    # 3. encode each split without adding BOS/EOS here;
    # 4. use build_context_targets() to add sequence boundaries;
    # 5. train with full-batch gradient descent and fixed seed;
    # 6. restore the parameters with the best validation loss;
    # 7. save model and tokenizer, then evaluate test data once.
    train_text, valid_text, test_text = make_text_splits(load_corpus())
    tokenizer = CharTokenizer.from_text(train_text)
    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)
    test_ids = tokenizer.encode(test_text)

    context_size=MLPConfig.context_size
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    train_contexts, train_targets = build_context_targets(
        train_ids,
        context_size = context_size,
        bos_id = bos_id,
        eos_id = eos_id,
    )
    valid_contexts, valid_targets = build_context_targets(
        valid_ids,
        context_size = context_size,
        bos_id = bos_id,
        eos_id = eos_id,
    )
    test_contexts, test_targets = build_context_targets(
        test_ids,
        context_size = context_size,
        bos_id = bos_id,
        eos_id = eos_id,
    )

    config = MLPConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=context_size,
        embedding_dim=8,
        hidden_dim=32,
    )

    model = TokenMLP(config, RANDOM_SEED)

    best_valid_loss = model.loss(valid_contexts, valid_targets)
    best_step = 0
    best_params = {
        name: parameter.copy()
        for name, parameter in model.params.items()
    }

    for step in range(TRAINING_STEPS):
        _, grads = model.loss_and_gradients(train_contexts, train_targets)
        model.apply_gradients(grads, LEARNING_RATE)

        loss = model.loss(valid_contexts, valid_targets)
        if loss < best_valid_loss:
            best_valid_loss = loss
            best_step = step
            best_params = {
                name: parameter.copy()
                for name, parameter in model.params.items()
            }

        if step % 100 == 0:
            print(f"step: {step}, loss: {loss:.2f}")

    for name in model.PARAMETER_NAMES:
        model.params[name] = best_params[name].copy()
    
    train_loss = model.loss(train_contexts, train_targets)
    valid_loss = model.loss(valid_contexts, valid_targets)
    test_loss = model.loss(test_contexts, test_targets)
    
    model.save(model_path)
    tokenizer.save(tokenizer_path)
            
    return {
        "best_step": best_step,
        "train_examples": len(train_targets),
        "valid_examples": len(valid_targets),
        "test_examples": len(test_targets),
        "train_loss": float(train_loss),
        "valid_loss": float(valid_loss),
        "test_loss": float(test_loss),
    }


if __name__ == "__main__":
    print(train_and_evaluate())
