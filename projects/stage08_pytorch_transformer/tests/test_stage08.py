"""Active boundary checks plus learner-unlocked acceptance tests."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import projects.stage07_char_transformer.data as stage07_data
import projects.stage07_char_transformer.model as stage07_model
import projects.stage08_pytorch_transformer.data as data
import projects.stage08_pytorch_transformer.generate as generation
import projects.stage08_pytorch_transformer.model as model_module
import projects.stage08_pytorch_transformer.train as train_module
from projects.tokenization.char_tokenizer import CharTokenizer


def test_config_matches_stage07_dimensions_and_validates_heads():
    config = model_module.TransformerConfig(vocab_size=11)
    assert (config.block_size, config.model_dim, config.num_heads,
            config.num_layers, config.feed_forward_dim, config.head_dim) == (8, 16, 4, 2, 32, 4)
    with pytest.raises(ValueError, match="divisible"):
        model_module.TransformerConfig(vocab_size=11, model_dim=10, num_heads=4)


def test_corpus_interface_is_the_stage07_boundary():
    assert data.prepare_corpus is stage07_data.prepare_corpus
    tokenizer, train, valid, test = data.prepare_corpus(block_size=8)
    assert tokenizer.vocab_size > 0
    for inputs, targets in (train, valid, test):
        assert inputs.shape == targets.shape
        assert inputs.shape[1] == 8
        assert inputs.dtype == targets.dtype == np.int64


def test_model_classes_are_torch_modules():
    config = model_module.TransformerConfig(vocab_size=11)
    for cls in (model_module.CausalSelfAttention,
                model_module.TransformerBlock,
                model_module.TorchCharTransformer):
        assert isinstance(cls(config), nn.Module)


def test_default_corpus_architecture_has_stage07_parameter_count():
    model = model_module.TorchCharTransformer(
        model_module.TransformerConfig(vocab_size=40)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 5800


def test_training_config_rejects_invalid_steps():
    with pytest.raises(ValueError, match="max_steps"):
        train_module.TrainingConfig(max_steps=0)


def test_attention_shape_and_future_isolation():
    torch.manual_seed(3)
    layer = model_module.CausalSelfAttention(
        model_module.TransformerConfig(vocab_size=11, model_dim=8, num_heads=2)
    )
    x = torch.randn(2, 4, 8)
    changed = x.clone()
    changed[:, 3] += 10
    with torch.no_grad():
        output = layer(x)
        changed_output = layer(changed)
    assert output.shape == x.shape
    torch.testing.assert_close(output[:, :3], changed_output[:, :3])


def test_block_shape():
    block = model_module.TransformerBlock(model_module.TransformerConfig(vocab_size=11))
    assert block(torch.randn(2, 4, 16)).shape == (2, 4, 16)


def test_model_logits_and_gradients():
    config = model_module.TransformerConfig(vocab_size=11, block_size=4)
    model = model_module.TorchCharTransformer(config)
    ids = torch.tensor([[1, 2, 3, 4], [1, 4, 3, 2]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5], [4, 3, 2, 5]], dtype=torch.long)
    logits = model(ids)
    assert logits.shape == (2, 4, 11)
    model.loss(ids, targets).backward()
    params = list(model.parameters())
    assert params
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in params)


def test_fixed_parameters_match_stage07_logits():
    config = model_module.TransformerConfig(
        vocab_size=7, block_size=4, model_dim=8, num_heads=2,
        num_layers=1, feed_forward_dim=16,
    )
    old = stage07_model.CharTransformer(
        stage07_model.TransformerConfig(**vars(config)), seed=3
    )
    new = model_module.TorchCharTransformer(config).double()

    def copy_into(parameter, values):
        parameter.copy_(torch.as_tensor(values, dtype=torch.float64))

    with torch.no_grad():
        copy_into(new.token_embedding.weight, old.params["token_embedding"])
        copy_into(new.position_embedding.weight, old.params["position_embedding"])
        copy_into(new.vocab_projection.weight, old.params["W_out"].T)
        copy_into(new.vocab_projection.bias, old.params["b_out"])
        copy_into(new.final_norm.weight, old.final_norm.params["gamma"])
        copy_into(new.final_norm.bias, old.final_norm.params["beta"])
        old_block, new_block = old.blocks[0], new.blocks[0]
        for old_norm, new_norm in (
            (old_block.norm1, new_block.norm1),
            (old_block.norm2, new_block.norm2),
        ):
            copy_into(new_norm.weight, old_norm.params["gamma"])
            copy_into(new_norm.bias, old_norm.params["beta"])
        weights = old_block.attention.params
        copy_into(new_block.attention.qkv.weight, np.concatenate(
            [weights[f"W_{name}"].T for name in ("Q", "K", "V")], axis=0
        ))
        copy_into(new_block.attention.output.weight, weights["W_O"].T)
        for index, old_weight, old_bias in (
            (0, "W1", "b1"), (2, "W2", "b2"),
        ):
            copy_into(new_block.mlp[index].weight, old_block.ffn.params[old_weight].T)
            copy_into(new_block.mlp[index].bias, old_block.ffn.params[old_bias])

    for ids in (
        np.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64),
        np.asarray([[1, 2]], dtype=np.int64),
    ):
        with torch.no_grad():
            actual = new(torch.from_numpy(ids)).numpy()
        np.testing.assert_allclose(actual, old.forward(ids), rtol=1e-10, atol=1e-10)


def test_save_load_preserves_logits(tmp_path):
    model = model_module.TorchCharTransformer(
        model_module.TransformerConfig(vocab_size=11, block_size=4)
    ).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    path = tmp_path / "nested" / "model.pt"
    model.save(path)
    restored = model_module.TorchCharTransformer.load(path).eval()
    assert restored.config == model.config
    with torch.no_grad():
        torch.testing.assert_close(model(ids), restored(ids))


def test_inference_bundle_preserves_and_checks_tokenizer(tmp_path):
    tokenizer = CharTokenizer.from_text("ab")
    model = model_module.TorchCharTransformer(
        model_module.TransformerConfig(vocab_size=tokenizer.vocab_size)
    ).eval()
    path = tmp_path / "bundle.pt"
    model.save(path, tokenizer=tokenizer)
    restored, loaded_tokenizer = model_module.TorchCharTransformer.load_with_tokenizer(path)
    assert loaded_tokenizer == tokenizer
    ids = torch.tensor([[1, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(model(ids), restored(ids))
    assert model_module.TorchCharTransformer.load(path, tokenizer=tokenizer).config == model.config
    with pytest.raises(ValueError, match="tokenizer does not match"):
        model_module.TorchCharTransformer.load(
            path, tokenizer=CharTokenizer.from_text("ac")
        )


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("format", "another-format", "unsupported model format"),
        ("version", 99, "unsupported model version"),
    ],
)
def test_all_load_paths_reject_incompatible_bundle_envelope(
    tmp_path, field, invalid_value, message
):
    tokenizer = CharTokenizer.from_text("ab")
    model = model_module.TorchCharTransformer(
        model_module.TransformerConfig(vocab_size=tokenizer.vocab_size)
    )
    valid_path = tmp_path / "valid.pt"
    invalid_path = tmp_path / "invalid.pt"
    model.save(valid_path, tokenizer=tokenizer)
    payload = torch.load(valid_path, weights_only=True)
    payload[field] = invalid_value
    torch.save(payload, invalid_path)

    with pytest.raises(ValueError, match=message):
        model_module.TorchCharTransformer.load(invalid_path)
    with pytest.raises(ValueError, match=message):
        model_module.TorchCharTransformer.load_with_tokenizer(invalid_path)


def test_generation_from_bos_is_reproducible_and_restores_mode():
    torch.manual_seed(11)
    model = model_module.TorchCharTransformer(
        model_module.TransformerConfig(vocab_size=7, block_size=4)
    )
    first = generation.generate_ids(
        model, [1], eos_id=2, max_new_tokens=9, seed=5, top_k=3
    )
    second = generation.generate_ids(
        model, [1], eos_id=2, max_new_tokens=9, seed=5, top_k=3
    )
    assert first == second
    assert 1 <= len(first) <= 10
    assert model.training


def test_generation_stops_at_eos_and_rolls_context():
    class FixedNextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=5, block_size=3)
            self.weight = nn.Parameter(torch.zeros(()))
            self.context_lengths = []
            self.next_id = 3

        def forward(self, token_ids):
            self.context_lengths.append(token_ids.shape[1])
            logits = torch.zeros(1, token_ids.shape[1], 5)
            logits[..., self.next_id] = 10
            return logits

    model = FixedNextModel()
    generated = generation.generate_ids(
        model, [1], eos_id=2, max_new_tokens=5, top_k=1
    )
    assert generated == [1, 3, 3, 3, 3, 3]
    assert model.context_lengths == [1, 2, 3, 3, 3]
    assert generation.generate_ids(model, [1, 2], eos_id=2) == [1, 2]
    model.next_id = 2
    assert generation.generate_ids(model, [1], eos_id=2, top_k=1) == [1, 2]


def test_tiny_training_reduces_loss_without_test_split():
    ids = np.tile(np.asarray([[1, 2, 3, 4]], dtype=np.int64), (16, 1))
    targets = np.tile(np.asarray([[2, 3, 4, 5]], dtype=np.int64), (16, 1))
    outcome = train_module.train_model(
        model_config=model_module.TransformerConfig(vocab_size=6, block_size=4),
        training_config=train_module.TrainingConfig(max_steps=100, eval_interval=10),
        train_data=(ids, targets),
        valid_data=(ids.copy(), targets.copy()),
    )
    assert outcome.best_valid_loss < outcome.initial_valid_loss
    assert outcome.best_valid_loss < 0.1
    assert outcome.best_step > 0
    with torch.no_grad():
        restored_loss = outcome.model.loss(
            *train_module.to_tensors((ids, targets))
        ).item()
    assert restored_loss == pytest.approx(outcome.best_valid_loss, abs=1e-6)


def test_initial_model_is_restored_when_validation_never_improves(monkeypatch):
    class OpposingTargetModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def loss(self, contexts, targets):
            return (self.weight - targets.float().mean()).square()

    monkeypatch.setattr(train_module, "TorchCharTransformer", OpposingTargetModel)
    contexts = np.zeros((4, 1), dtype=np.int64)
    outcome = train_module.train_model(
        model_config=model_module.TransformerConfig(vocab_size=3, block_size=1),
        training_config=train_module.TrainingConfig(
            learning_rate=0.1, batch_size=2, max_steps=3,
            eval_interval=1, patience_evaluations=2,
        ),
        train_data=(contexts, np.ones((4, 1), dtype=np.int64)),
        valid_data=(contexts, -np.ones((4, 1), dtype=np.int64)),
    )
    assert outcome.best_step == 0
    assert outcome.completed_step == 2
    assert outcome.stopped_early
    assert outcome.model.weight.item() == pytest.approx(0.0)
    assert outcome.best_valid_loss == pytest.approx(outcome.initial_valid_loss)


def test_training_is_reproducible_and_evaluates_final_step():
    ids = np.tile(np.asarray([[1, 2, 3, 4]], dtype=np.int64), (4, 1))
    targets = np.tile(np.asarray([[2, 3, 4, 5]], dtype=np.int64), (4, 1))
    kwargs = dict(
        model_config=model_module.TransformerConfig(vocab_size=6, block_size=4),
        training_config=train_module.TrainingConfig(
            batch_size=2, max_steps=3, eval_interval=10, seed=17,
        ),
        train_data=(ids, targets),
        valid_data=(ids.copy(), targets.copy()),
    )
    first = train_module.train_model(**kwargs)
    second = train_module.train_model(**kwargs)
    assert first.completed_step == second.completed_step == 3
    assert first.history[-1]["step"] == second.history[-1]["step"] == 3
    assert first.best_valid_loss == second.best_valid_loss
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)


def test_train_and_evaluate_keeps_numpy_training_boundary_and_writes_report(
    tmp_path, monkeypatch
):
    ids = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    targets = np.asarray([[2, 3, 4, 5]], dtype=np.int64)
    train_data = (ids, targets)
    valid_data = (ids.copy(), targets.copy())
    test_data = (ids.copy(), targets.copy())
    tokenizer = CharTokenizer.from_text("abc")
    config = model_module.TransformerConfig(vocab_size=tokenizer.vocab_size, block_size=4)

    def fake_prepare_corpus(*, block_size):
        assert block_size == 4
        return tokenizer, train_data, valid_data, test_data

    def fake_train_model(*, model_config, training_config, train_data, valid_data):
        assert model_config == config
        assert isinstance(train_data[0], np.ndarray)
        assert isinstance(valid_data[0], np.ndarray)
        return train_module.TrainingOutcome(
            model=model_module.TorchCharTransformer(config),
            initial_train_loss=2.0,
            initial_valid_loss=2.0,
            best_valid_loss=1.5,
            best_step=10,
            completed_step=10,
            stopped_early=False,
            elapsed_seconds=0.1,
            history=[],
        )

    monkeypatch.setattr(train_module, "prepare_corpus", fake_prepare_corpus)
    monkeypatch.setattr(train_module, "train_model", fake_train_model)
    report_path = tmp_path / "reports" / "result.json"
    model_path = tmp_path / "models" / "model.pt"
    report = train_module.train_and_evaluate(
        model_config=config, model_path=model_path, report_path=report_path
    )

    assert report["model_config"]["block_size"] == 4
    assert report["best_valid_loss"] == 1.5
    assert isinstance(report["generated_text"], str)
    assert all(isinstance(report[key], float)
               for key in ("train_loss", "valid_loss", "test_loss"))
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    loaded_model, loaded_tokenizer = model_module.TorchCharTransformer.load_with_tokenizer(model_path)
    assert loaded_model.config == config
    assert loaded_tokenizer == tokenizer
