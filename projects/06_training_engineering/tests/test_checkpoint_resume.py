import json

import numpy as np

from checkpoint import load_checkpoint
from model import MLPConfig
from train import TrainingConfig, train_and_evaluate, train_model


def _tiny_data():
    contexts = np.array(
        [[0, 0], [0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.int64
    )
    targets = np.array([1, 2, 3, 0, 2, 3], dtype=np.int64)
    return contexts, targets


def test_interrupted_resume_matches_uninterrupted_training(tmp_path):
    model_config = MLPConfig(vocab_size=4, context_size=2, embedding_dim=3, hidden_dim=5)
    training_config = TrainingConfig(
        optimizer="adam",
        learning_rate=0.01,
        batch_size=2,
        max_steps=12,
        eval_interval=3,
        patience_evaluations=20,
        seed=17,
    )
    data = _tiny_data()
    uninterrupted = train_model(
        model_config=model_config,
        training_config=training_config,
        train_data=data,
        valid_data=data,
        checkpoint_path=tmp_path / "uninterrupted.npz",
    )
    interrupted = train_model(
        model_config=model_config,
        training_config=training_config,
        train_data=data,
        valid_data=data,
        checkpoint_path=tmp_path / "resumed.npz",
        stop_after_steps=5,
    )
    assert not interrupted.completed
    resumed = train_model(
        model_config=model_config,
        training_config=training_config,
        train_data=data,
        valid_data=data,
        checkpoint_path=tmp_path / "resumed.npz",
        resume=True,
    )
    assert resumed.completed
    assert resumed.state.step == uninterrupted.state.step == 12
    assert resumed.state.best_step == uninterrupted.state.best_step
    for name in resumed.model.PARAMETER_NAMES:
        assert np.array_equal(resumed.model.params[name], uninterrupted.model.params[name])
        assert np.array_equal(resumed.best_params[name], uninterrupted.best_params[name])
    resumed_state = resumed.optimizer.state_dict()
    uninterrupted_state = uninterrupted.optimizer.state_dict()
    assert set(resumed_state) == set(uninterrupted_state)
    for name in resumed_state:
        assert np.array_equal(resumed_state[name], uninterrupted_state[name])


def test_checkpoint_restores_model_optimizer_and_trainer_state(tmp_path):
    config = TrainingConfig(max_steps=4, eval_interval=2, patience_evaluations=10)
    model_config = MLPConfig(vocab_size=4, embedding_dim=2, hidden_dim=4)
    path = tmp_path / "checkpoint.npz"
    outcome = train_model(
        model_config=model_config,
        training_config=config,
        train_data=_tiny_data(),
        valid_data=_tiny_data(),
        checkpoint_path=path,
    )
    model, optimizer_name, optimizer_state, saved_config, state, best_params = load_checkpoint(path)
    assert optimizer_name == "adam"
    assert saved_config["batch_size"] == config.batch_size
    assert state["step"] == outcome.state.step == 4
    assert int(optimizer_state["step_count"].item()) == 4
    for name in model.PARAMETER_NAMES:
        assert np.array_equal(model.params[name], outcome.model.params[name])
        assert np.array_equal(best_params[name], outcome.best_params[name])


def test_same_config_repeats_identically(tmp_path):
    config = TrainingConfig(max_steps=8, eval_interval=2, patience_evaluations=10)
    model_config = MLPConfig(vocab_size=4, embedding_dim=2, hidden_dim=4)
    first = train_model(
        model_config=model_config,
        training_config=config,
        train_data=_tiny_data(),
        valid_data=_tiny_data(),
        checkpoint_path=tmp_path / "first.npz",
    )
    second = train_model(
        model_config=model_config,
        training_config=config,
        train_data=_tiny_data(),
        valid_data=_tiny_data(),
        checkpoint_path=tmp_path / "second.npz",
    )
    assert first.state.history == second.state.history
    for name in first.model.PARAMETER_NAMES:
        assert np.array_equal(first.model.params[name], second.model.params[name])


def test_training_outputs_are_separate_artifacts(tmp_path):
    report = train_and_evaluate(
        training_config=TrainingConfig(
            max_steps=4, eval_interval=2, patience_evaluations=10
        ),
        checkpoint_path=tmp_path / "checkpoint.npz",
        model_path=tmp_path / "model.npz",
        tokenizer_path=tmp_path / "tokenizer.json",
        report_path=tmp_path / "report.json",
    )
    assert report["completed_step"] == 4
    assert set(json.loads((tmp_path / "report.json").read_text())) >= {
        "training_config",
        "history",
        "test_loss",
    }
    with np.load(tmp_path / "checkpoint.npz", allow_pickle=False) as checkpoint:
        assert any(key.startswith("optimizer.") for key in checkpoint.files)
    with np.load(tmp_path / "model.npz", allow_pickle=False) as model:
        assert not any(key.startswith("optimizer.") for key in model.files)
