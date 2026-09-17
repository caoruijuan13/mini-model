"""Training checks kept separate from the attention/model scaffold tests."""

from importlib import import_module
import json

import numpy as np
import pytest


data = import_module("projects.07_char_transformer.data")
model_module = import_module("projects.07_char_transformer.model")
train_module = import_module("projects.07_char_transformer.train")


def tiny_problem():
    inputs = np.array(
        [[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int64
    )
    targets = np.array(
        [[2, 3, 1], [3, 1, 2], [1, 2, 3]], dtype=np.int64
    )
    config = model_module.TransformerConfig(
        vocab_size=4, block_size=3, model_dim=8, num_heads=2,
        num_layers=1, feed_forward_dim=16,
    )
    return config, (inputs, targets)


def test_tiny_sequence_can_be_overfit():
    model_config, dataset = tiny_problem()
    outcome = train_module.train_model(
        model_config=model_config,
        training_config=train_module.TrainingConfig(
            learning_rate=0.01, batch_size=3, max_steps=250,
            eval_interval=10, patience_evaluations=30,
        ),
        train_data=dataset,
        valid_data=dataset,
    )
    assert outcome.initial_train_loss > 1.0
    assert outcome.model.loss(*dataset) < 0.01
    assert outcome.best_step > 0
    assert outcome.completed_step == 250


def test_validation_selects_and_restores_best_weights():
    model_config, (inputs, targets) = tiny_problem()
    conflicting_targets = np.ones_like(targets)
    outcome = train_module.train_model(
        model_config=model_config,
        training_config=train_module.TrainingConfig(
            learning_rate=0.01, batch_size=3, max_steps=20,
            eval_interval=1, patience_evaluations=3,
        ),
        train_data=(inputs, targets),
        valid_data=(inputs, conflicting_targets),
    )
    assert outcome.stopped_early
    assert outcome.best_step < outcome.completed_step
    assert outcome.model.loss(inputs, conflicting_targets) == pytest.approx(
        outcome.best_valid_loss
    )
    assert outcome.best_valid_loss == pytest.approx(
        min(outcome.initial_valid_loss, *(
            item["valid_loss"] for item in outcome.history
        ))
    )


def test_training_is_deterministic_for_fixed_seed():
    model_config, dataset = tiny_problem()
    config = train_module.TrainingConfig(
        learning_rate=0.01, batch_size=2, max_steps=20,
        eval_interval=5, patience_evaluations=5, seed=13,
    )
    first = train_module.train_model(
        model_config=model_config, training_config=config,
        train_data=dataset, valid_data=dataset,
    )
    second = train_module.train_model(
        model_config=model_config, training_config=config,
        train_data=dataset, valid_data=dataset,
    )
    assert first.history == second.history
    for name, parameter in first.model.named_parameters().items():
        assert np.array_equal(parameter, second.model.named_parameters()[name])


def test_real_corpus_pipeline_reports_test_after_selection(tmp_path):
    tokenizer, _, _, test_data = data.prepare_corpus(block_size=8)
    model_config = model_module.TransformerConfig(
        vocab_size=tokenizer.vocab_size, block_size=8,
        model_dim=8, num_heads=2, num_layers=1, feed_forward_dim=16,
    )
    model_path = tmp_path / "model.npz"
    report_path = tmp_path / "report.json"
    report = train_module.train_and_evaluate(
        model_config=model_config,
        training_config=train_module.TrainingConfig(
            max_steps=5, eval_interval=5, patience_evaluations=2,
        ),
        model_path=model_path,
        report_path=report_path,
    )
    loaded = model_module.CharTransformer.load(model_path)
    assert report == json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded.loss(*test_data) == pytest.approx(report["test_loss"])
    assert report["best_step"] <= report["completed_step"]
    for split in ("train", "valid", "test"):
        assert np.isfinite(report[f"{split}_loss"])
        assert report[f"{split}_perplexity"] == pytest.approx(
            np.exp(report[f"{split}_loss"])
        )
