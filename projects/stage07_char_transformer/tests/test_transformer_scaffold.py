"""Runnable scaffold checks; skipped tests become acceptance tests later."""

import numpy as np
import pytest

import projects.stage07_char_transformer.attention as attention
import projects.stage07_char_transformer.data as data
import projects.stage07_char_transformer.generate as generation
import projects.stage07_char_transformer.layers as layers
import projects.stage07_char_transformer.model as model_module


def test_config_defines_head_dimension_and_rejects_invalid_split():
    config = model_module.TransformerConfig(vocab_size=7, model_dim=16, num_heads=4)
    assert config.head_dim == 4
    with pytest.raises(ValueError, match="divisible"):
        model_module.TransformerConfig(vocab_size=7, model_dim=10, num_heads=4)


def test_shifted_windows_include_bos_and_eos_without_split_overlap():
    inputs, targets = data.build_sequence_windows(
        [3, 4, 5], block_size=2, bos_id=1, eos_id=2
    )
    assert inputs.tolist() == [[1, 3], [3, 4], [4, 5]]
    assert targets.tolist() == [[3, 4], [4, 5], [5, 2]]


def test_short_sequence_fails_instead_of_silently_dropping_eos():
    with pytest.raises(ValueError, match="too short"):
        data.build_sequence_windows([3], block_size=4, bos_id=1, eos_id=2)


def test_real_corpus_data_shapes_and_train_only_vocabulary():
    tokenizer, train, valid, test = data.prepare_corpus(block_size=8)
    train_text, _, _ = data.make_text_splits(data.load_corpus())
    assert tokenizer.tokens == data.CharTokenizer.from_text(train_text).tokens
    for inputs, targets in (train, valid, test):
        assert inputs.ndim == targets.ndim == 2
        assert inputs.shape == targets.shape
        assert inputs.shape[1] == 8


def test_multihead_parameters_have_expected_shapes_and_seed():
    first = attention.MultiHeadAttention(model_dim=8, num_heads=2, seed=11)
    second = attention.MultiHeadAttention(model_dim=8, num_heads=2, seed=11)
    assert first.head_dim == 4
    assert set(first.params) == {"W_Q", "W_K", "W_V", "W_O"}
    for name in first.params:
        assert first.params[name].shape == (8, 8)
        assert np.array_equal(first.params[name], second.params[name])


def test_future_positions_are_masked():
    mask = attention.causal_mask(4)
    assert mask.shape == (4, 4)
    assert np.all(mask[np.tril_indices(4)] == 0)
    assert np.all(np.isneginf(mask[np.triu_indices(4, k=1)]))


def test_attention_weights_are_causal_and_normalized():
    q = np.ones((2, 3, 4, 5))
    k = np.ones_like(q)
    v = np.ones_like(q)
    output, weights = attention.scaled_dot_product_attention(
        q, k, v, attention.causal_mask(4)
    )
    assert output.shape == (2, 3, 4, 5)
    assert weights.shape == (2, 3, 4, 4)
    assert np.allclose(weights.sum(axis=-1), 1)
    assert np.all(weights[..., np.triu_indices(4, k=1)[0], np.triu_indices(4, k=1)[1]] == 0)


def test_scaled_attention_uses_the_supplied_mask():
    q = np.ones((1, 1, 2, 2))
    k = np.ones_like(q)
    v = np.ones_like(q)
    _, weights = attention.scaled_dot_product_attention(
        q, k, v, np.zeros((2, 2))
    )
    assert np.allclose(weights, 0.5)


def test_multihead_output_shape_and_no_future_leak():
    layer = attention.MultiHeadAttention(model_dim=8, num_heads=2)
    first = np.random.default_rng(3).normal(size=(2, 4, 8))
    changed_future = first.copy()
    changed_future[:, 3, :] += 100
    original_output = layer.forward(first)
    changed_output = layer.forward(changed_future)
    assert original_output.shape == (2, 4, 8)
    assert np.allclose(original_output[:, :3, :], changed_output[:, :3, :])


def test_multihead_handles_one_batch_and_one_head():
    layer = attention.MultiHeadAttention(model_dim=2, num_heads=1, seed=5)
    for name in layer.params:
        layer.params[name] = np.eye(2)
    inputs = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    output = layer.forward(inputs)
    assert output.shape == (1, 2, 2)
    assert np.allclose(output[0, 0], [1.0, 0.0])
    assert np.allclose(output[0, 1].sum(), 1.0)


def test_multihead_batch_matches_individual_examples():
    layer = attention.MultiHeadAttention(model_dim=8, num_heads=2, seed=13)
    inputs = np.random.default_rng(17).normal(size=(2, 4, 8))
    batch_output = layer.forward(inputs)
    individual_output = np.concatenate(
        [layer.forward(inputs[index : index + 1]) for index in range(2)], axis=0
    )
    assert np.allclose(batch_output, individual_output)


def test_block_preserves_batch_sequence_and_feature_axes():
    block = layers.TransformerBlock(model_dim=8, num_heads=2, hidden_dim=16)
    assert block.forward(np.ones((2, 4, 8))).shape == (2, 4, 8)


def test_transformer_logits_and_loss_contract():
    config = model_module.TransformerConfig(vocab_size=7, block_size=4)
    model = model_module.CharTransformer(config)
    inputs = np.array([[1, 3, 4, 5], [1, 5, 4, 3]], dtype=np.int64)
    targets = np.array([[3, 4, 5, 2], [5, 4, 3, 2]], dtype=np.int64)
    assert model.forward(inputs).shape == (2, 4, 7)
    assert np.isfinite(model.loss(inputs, targets))


def test_loss_includes_last_position_and_handles_one_token():
    model = model_module.CharTransformer(
        model_module.TransformerConfig(vocab_size=2, block_size=2)
    )
    model.forward = lambda _: np.array([[[2.0, 0.0], [0.0, 2.0]]])
    inputs = np.array([[0, 0]])
    assert model.loss(inputs, np.array([[0, 1]])) == pytest.approx(
        np.logaddexp(2.0, 0.0) - 2.0
    )
    assert model.loss(inputs, np.array([[0, 0]])) > model.loss(
        inputs, np.array([[0, 1]])
    )

    model.forward = lambda _: np.array([[[2.0, 0.0]]])
    assert model.loss(np.array([[0]]), np.array([[1]])) == pytest.approx(
        np.logaddexp(2.0, 0.0)
    )


def test_loss_stays_finite_for_very_unlikely_target():
    model = model_module.CharTransformer(
        model_module.TransformerConfig(vocab_size=2, block_size=1)
    )
    model.forward = lambda _: np.array([[[1000.0, 0.0]]])
    assert model.loss(np.array([[0]]), np.array([[1]])) == pytest.approx(1000.0)


def test_loss_rejects_invalid_targets():
    model = model_module.CharTransformer(
        model_module.TransformerConfig(vocab_size=2, block_size=1)
    )
    inputs = np.array([[0]])
    for targets in (np.array([[-1]]), np.array([[2]]), np.array([[0.5]])):
        with pytest.raises(ValueError):
            model.loss(inputs, targets)
    with pytest.raises(ValueError, match="same shape"):
        model.loss(inputs, np.array([[0, 1]]))


def test_transformer_gradients_match_finite_differences():
    config = model_module.TransformerConfig(
        vocab_size=5, block_size=3, model_dim=4, num_heads=2,
        num_layers=2, feed_forward_dim=6,
    )
    model = model_module.CharTransformer(config, seed=11)
    inputs = np.array([[0, 1, 0], [2, 3, 1]])
    targets = np.array([[1, 0, 2], [3, 1, 4]])
    loss, grads = model.loss_and_gradients(inputs, targets)
    params = model.named_parameters()
    assert loss == pytest.approx(model.loss(inputs, targets))
    assert set(grads) == set(params)

    step = 1e-6
    for name, param in params.items():
        assert grads[name].shape == param.shape
        assert np.all(np.isfinite(grads[name]))
        for index in {tuple(0 for _ in param.shape), tuple(size - 1 for size in param.shape)}:
            original = param[index]
            param[index] = original + step
            positive = model.loss(inputs, targets)
            param[index] = original - step
            negative = model.loss(inputs, targets)
            param[index] = original
            numerical = (positive - negative) / (2 * step)
            assert grads[name][index] == pytest.approx(
                numerical, abs=2e-5, rel=2e-3
            ), name


def test_inference_model_round_trip_preserves_logits(tmp_path):
    config = model_module.TransformerConfig(vocab_size=7, block_size=4)
    model = model_module.CharTransformer(config, seed=19)
    for index, parameter in enumerate(model.named_parameters().values()):
        parameter.flat[0] += (index + 1) / 100.0
    inputs = np.array([[1, 3, 4, 5]], dtype=np.int64)
    expected = model.forward(inputs)
    path = tmp_path / "char_transformer.npz"
    model.save(path)
    loaded = model_module.CharTransformer.load(path)
    assert loaded.config == model.config
    assert np.array_equal(loaded.forward(inputs), expected)
    for name, parameter in model.named_parameters().items():
        assert np.array_equal(loaded.named_parameters()[name], parameter)


def test_model_load_rejects_invalid_parameter_shape(tmp_path):
    model = model_module.CharTransformer(
        model_module.TransformerConfig(vocab_size=7, block_size=4)
    )
    path = tmp_path / "model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name].copy() for name in data.files}
    payload["blocks.0.attention.W_Q"] = np.zeros((1, 1))
    invalid_path = tmp_path / "invalid_model.npz"
    np.savez(invalid_path, **payload)
    with pytest.raises(ValueError, match="shape or dtype"):
        model_module.CharTransformer.load(invalid_path)

    missing_path = tmp_path / "missing_model.npz"
    np.savez(missing_path, **{name: value for name, value in payload.items() if name != "vocab_size"})
    with pytest.raises(ValueError, match="missing config"):
        model_module.CharTransformer.load(missing_path)


def test_generation_keeps_prompt_and_respects_maximum_length():
    model = model_module.CharTransformer(
        model_module.TransformerConfig(vocab_size=7, block_size=4)
    )
    generated = generation.generate_ids(
        model, [1, 3], eos_id=2, max_new_tokens=3, seed=11
    )
    assert generated[:2] == [1, 3]
    assert 2 <= len(generated) <= 5
    assert generated == generation.generate_ids(
        model, [1, 3], eos_id=2, max_new_tokens=3, seed=11
    )


def test_generation_uses_final_logits_rolls_context_and_stops_at_eos():
    class ScriptedModel:
        config = model_module.TransformerConfig(vocab_size=5, block_size=3)

        def __init__(self):
            self.seen_contexts = []

        def forward(self, context_ids):
            context = tuple(context_ids[0])
            self.seen_contexts.append(context)
            next_by_last = {1: 3, 3: 4, 4: 2}
            logits = np.full((1, len(context), self.config.vocab_size), -100.0)
            logits[0, :-1, 2] = 100.0  # Wrong if generation reads an earlier position.
            logits[0, -1, next_by_last[context[-1]]] = 100.0
            return logits

    model = ScriptedModel()
    result = generation.generate_ids(
        model, [1, 1, 1, 1], eos_id=2, max_new_tokens=8, top_k=1
    )
    assert result == [1, 1, 1, 1, 3, 4, 2]
    assert model.seen_contexts == [(1, 1, 1), (1, 1, 3), (1, 3, 4)]


def test_generation_rejects_invalid_settings_and_keeps_finished_prompt():
    model = model_module.CharTransformer(
        model_module.TransformerConfig(vocab_size=5, block_size=3)
    )
    assert generation.generate_ids(model, [1, 2], eos_id=2) == [1, 2]
    assert generation.generate_ids(model, [1], eos_id=2, max_new_tokens=0) == [1]
    with pytest.raises(ValueError, match="at least one"):
        generation.generate_ids(model, [], eos_id=2)
    with pytest.raises(ValueError, match="positive and finite"):
        generation.generate_ids(model, [1], eos_id=2, temperature=0)
    with pytest.raises(ValueError, match="top_k"):
        generation.generate_ids(model, [1], eos_id=2, top_k=6)
    with pytest.raises(ValueError, match="outside vocabulary"):
        generation.generate_ids(model, [5], eos_id=2)
    with pytest.raises(ValueError, match="non-negative"):
        generation.generate_ids(model, [1], eos_id=2, max_new_tokens=-1)


def test_generation_rejects_incorrect_model_output_shape():
    model = model_module.CharTransformer(
        model_module.TransformerConfig(vocab_size=5, block_size=3)
    )
    model.forward = lambda _: np.zeros((1, 5))
    with pytest.raises(ValueError, match="must return logits with shape"):
        generation.generate_ids(model, [1], eos_id=2, max_new_tokens=1)
