"""Stage 10 loading and generation contracts without training fixtures."""

from types import SimpleNamespace

import pytest
import torch

import projects.stage10_inference_serving.runtime as runtime_module
import projects.stage10_inference_serving.cli as cli_module
from projects.stage08_pytorch_transformer.model import (
    TorchCharTransformer,
    TransformerConfig,
)
from projects.tokenization.char_tokenizer import CharTokenizer


def test_runtime_keeps_injected_model_and_tokenizer():
    model = object()
    tokenizer = object()

    runtime = runtime_module.InferenceRuntime(model, tokenizer)

    assert runtime.model is model
    assert runtime.tokenizer is tokenizer


def test_load_calls_model_loader_once_and_keeps_its_objects(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    model = object()
    tokenizer = object()
    calls = []

    class FakeModelLoader:
        @staticmethod
        def load_with_tokenizer(received_path):
            calls.append(received_path)
            return model, tokenizer

    monkeypatch.setattr(runtime_module, "TorchCharTransformer", FakeModelLoader)

    runtime = runtime_module.InferenceRuntime.load(path)

    assert calls == [path]
    assert runtime.model is model
    assert runtime.tokenizer is tokenizer


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("model file missing"), ValueError("tokenizer mismatch")],
)
def test_load_propagates_artifact_errors(tmp_path, monkeypatch, error):
    class FailingModelLoader:
        @staticmethod
        def load_with_tokenizer(_path):
            raise error

    monkeypatch.setattr(runtime_module, "TorchCharTransformer", FailingModelLoader)

    with pytest.raises(type(error), match=str(error)):
        runtime_module.InferenceRuntime.load(tmp_path / "invalid.pt")


def test_load_real_inference_bundle_without_training(tmp_path):
    tokenizer = CharTokenizer.from_text("ab")
    model = TorchCharTransformer(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=4,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            feed_forward_dim=16,
        )
    )
    path = tmp_path / "model.pt"
    model.save(path, tokenizer=tokenizer)

    runtime = runtime_module.InferenceRuntime.load(path)

    assert runtime.model.config == model.config
    assert runtime.model.training is False
    assert runtime.tokenizer.tokens == tokenizer.tokens


def test_cli_returns_structured_non_streaming_result(monkeypatch, capsys):
    class FakeRuntime:
        def generate(self, prompt, **options):
            assert prompt == "ab"
            assert options["top_k"] == 3
            return runtime_module.GenerationResult(prompt, "c", "eos", 1)

    monkeypatch.setattr(
        cli_module.InferenceRuntime,
        "load",
        classmethod(lambda cls, path: FakeRuntime()),
    )

    assert cli_module.main(["ab", "--top-k", "3"]) == 0
    assert capsys.readouterr().out == (
        '{"input_text": "ab", "generated_text": "c", "finish_reason": "eos", '
        '"generated_token_count": 1}\n'
    )


def test_cli_streams_fragments_without_json_wrapper(monkeypatch, capsys):
    class FakeRuntime:
        def stream_generate(self, prompt, **options):
            assert prompt == "ab"
            yield "c"
            yield "d"

    monkeypatch.setattr(
        cli_module.InferenceRuntime,
        "load",
        classmethod(lambda cls, path: FakeRuntime()),
    )

    assert cli_module.main(["ab", "--stream"]) == 0
    assert capsys.readouterr().out == "cd\n"


def test_generate_returns_only_new_text_and_eos_reason():
    tokenizer = CharTokenizer.from_text("abc")
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    b_id = tokenizer.token_to_id["b"]
    c_id = tokenizer.token_to_id["c"]

    class FixedNextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(vocab_size=tokenizer.vocab_size, block_size=4)
            self.seen_contexts = []

        def forward(self, token_ids):
            self.seen_contexts.append(token_ids[0].tolist())
            next_id = c_id if int(token_ids[0, -1]) == b_id else eos_id
            logits = torch.zeros(1, token_ids.shape[1], tokenizer.vocab_size)
            logits[..., next_id] = 10
            return logits

    model = FixedNextModel()
    runtime = runtime_module.InferenceRuntime(model, tokenizer)

    result = runtime.generate("ab", max_new_tokens=3, top_k=1)

    assert model.seen_contexts[0] == [bos_id, tokenizer.token_to_id["a"], b_id]
    assert result == runtime_module.GenerationResult(
        input_text="ab",
        generated_text="c",
        finish_reason="eos",
        generated_token_count=1,
    )


def test_generate_length_limit_does_not_include_prompt_tokens():
    tokenizer = CharTokenizer.from_text("ab")
    b_id = tokenizer.token_to_id["b"]

    class FixedNextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(vocab_size=tokenizer.vocab_size, block_size=4)

        def forward(self, token_ids):
            logits = torch.zeros(1, token_ids.shape[1], tokenizer.vocab_size)
            logits[..., b_id] = 10
            return logits

    runtime = runtime_module.InferenceRuntime(FixedNextModel(), tokenizer)

    assert runtime.generate(
        "a", max_new_tokens=1, top_k=1
    ) == runtime_module.GenerationResult(
        input_text="a",
        generated_text="b",
        finish_reason="length",
        generated_token_count=1,
    )
    assert runtime.generate(
        "", max_new_tokens=0, top_k=1
    ) == runtime_module.GenerationResult(
        input_text="",
        generated_text="",
        finish_reason="length",
        generated_token_count=0,
    )


def test_batch_returns_every_input_even_if_an_earlier_request_reaches_eos(monkeypatch):
    runtime = runtime_module.InferenceRuntime(object(), object())
    calls = []

    def fake_generate(prompt, **kwargs):
        calls.append(prompt)
        return runtime_module.GenerationResult(
            input_text=prompt,
            generated_text="",
            finish_reason="eos" if prompt == "first" else "length",
            generated_token_count=0,
        )

    monkeypatch.setattr(runtime, "generate", fake_generate)

    results = runtime.generate_batch(["first", "second"])

    assert calls == ["first", "second"]
    assert [result.input_text for result in results] == ["first", "second"]
    assert runtime.generate_batch([]) == []


def test_batch_matches_individual_generation_with_per_request_seeds(monkeypatch):
    tokenizer = CharTokenizer.from_text("ab")
    runtime = runtime_module.InferenceRuntime(object(), tokenizer)
    calls = []

    def fake_generate_ids(
        model, initial_ids, *, eos_id, max_new_tokens, seed, temperature, top_k
    ):
        calls.append((list(initial_ids), seed, max_new_tokens, temperature, top_k))
        next_token = "a" if seed % 2 == 0 else "b"
        return [*initial_ids, tokenizer.token_to_id[next_token]]

    monkeypatch.setattr(runtime_module, "generate_ids", fake_generate_ids)
    prompts = ["a", "", "b"]
    options = {"max_new_tokens": 1, "temperature": 0.5, "top_k": 2}
    expected = [
        runtime.generate(prompt, seed=31 + index, **options)
        for index, prompt in enumerate(prompts)
    ]
    calls.clear()

    actual = runtime.generate_batch(prompts, seed=31, **options)

    assert actual == expected
    assert [result.generated_text for result in actual] == ["b", "a", "b"]
    assert [call[1:] for call in calls] == [
        (31, 1, 0.5, 2),
        (32, 1, 0.5, 2),
        (33, 1, 0.5, 2),
    ]
    assert all(ids[0] == tokenizer.token_to_id[tokenizer.bos_token] for ids, *_ in calls)


def _scripted_stream_runtime():
    tokenizer = CharTokenizer.from_text("abcd")
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    next_ids = {
        tokenizer.token_to_id["b"]: tokenizer.token_to_id["c"],
        tokenizer.token_to_id["c"]: tokenizer.token_to_id["d"],
    }

    class ScriptedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(vocab_size=tokenizer.vocab_size, block_size=4)
            self.seen_contexts = []

        def forward(self, token_ids):
            self.seen_contexts.append(token_ids[0].tolist())
            next_id = next_ids.get(int(token_ids[0, -1]), eos_id)
            logits = torch.zeros(1, token_ids.shape[1], tokenizer.vocab_size)
            logits[..., next_id] = 10
            return logits

    model = ScriptedModel()
    return runtime_module.InferenceRuntime(model, tokenizer), model


def test_stream_yields_each_new_token_before_sampling_the_next_one():
    runtime, model = _scripted_stream_runtime()
    stream = iter(runtime.stream_generate("ab", max_new_tokens=3, top_k=1))

    assert model.seen_contexts == []
    assert next(stream) == "c"
    assert len(model.seen_contexts) == 1
    assert model.training
    assert next(stream) == "d"
    assert len(model.seen_contexts) == 2
    assert model.training
    assert list(stream) == []  # EOS ends the stream without becoming text.
    assert len(model.seen_contexts) == 3
    assert model.training

    non_streamed = runtime.generate("ab", max_new_tokens=3, top_k=1)
    assert non_streamed.finish_reason == "eos"
    assert non_streamed.generated_text == "cd"


def test_stream_emits_nothing_at_zero_limit_or_immediate_eos():
    runtime, model = _scripted_stream_runtime()

    assert list(runtime.stream_generate("", max_new_tokens=0, top_k=1)) == []
    assert model.seen_contexts == []
    assert list(runtime.stream_generate("d", max_new_tokens=3, top_k=1)) == []
    assert len(model.seen_contexts) == 1


def test_stream_matches_non_streamed_stochastic_sampling():
    torch.manual_seed(19)
    tokenizer = CharTokenizer.from_text("abcd")
    model = TorchCharTransformer(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=4,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            feed_forward_dim=16,
        )
    )
    runtime = runtime_module.InferenceRuntime(model, tokenizer)
    options = {
        "max_new_tokens": 8,
        "seed": 23,
        "temperature": 0.7,
        "top_k": 3,
    }

    non_streamed = runtime.generate("ab", **options)
    streamed = "".join(runtime.stream_generate("ab", **options))

    assert streamed == non_streamed.generated_text
    assert model.training


def test_perplexity_scores_each_target_once_with_rolling_context():
    tokenizer = CharTokenizer.from_text("abcd")
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    a_id = tokenizer.token_to_id["a"]
    b_id = tokenizer.token_to_id["b"]
    c_id = tokenizer.token_to_id["c"]
    d_id = tokenizer.token_to_id["d"]
    next_ids = {
        (bos_id,): a_id,
        (bos_id, a_id): b_id,
        (a_id, b_id): c_id,
        (b_id, c_id): d_id,
        (c_id, d_id): eos_id,
    }

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(vocab_size=tokenizer.vocab_size, block_size=2)
            self.contexts = []
            self.training_states = []

        def forward(self, token_ids):
            context = tuple(token_ids[0].tolist())
            self.contexts.append(context)
            self.training_states.append(self.training)
            logits = torch.zeros(1, token_ids.shape[1], tokenizer.vocab_size)
            logits[..., next_ids[context]] = 2.5
            return logits

    model = RecordingModel()
    runtime = runtime_module.InferenceRuntime(model, tokenizer)
    reference_logits = torch.zeros(1, tokenizer.vocab_size)
    reference_logits[0, a_id] = 2.5
    expected = float(
        torch.exp(
            torch.nn.functional.cross_entropy(
                reference_logits, torch.tensor([a_id])
            )
        ).item()
    )

    perplexity = runtime.evaluate_perplexity(["abcd"])

    assert perplexity == pytest.approx(expected)
    assert model.contexts == list(next_ids)
    assert model.training_states == [False] * 5
    assert model.training


def test_perplexity_treats_each_text_as_an_independent_sequence():
    tokenizer = CharTokenizer.from_text("a")
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    a_id = tokenizer.token_to_id["a"]

    class UniformModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(vocab_size=tokenizer.vocab_size, block_size=3)
            self.contexts = []

        def forward(self, token_ids):
            self.contexts.append(token_ids[0].tolist())
            return torch.zeros(1, token_ids.shape[1], tokenizer.vocab_size)

    model = UniformModel()
    runtime = runtime_module.InferenceRuntime(model, tokenizer)

    perplexity = runtime.evaluate_perplexity(["a", ""])

    assert perplexity == pytest.approx(tokenizer.vocab_size)
    assert model.contexts == [[bos_id], [bos_id, a_id], [bos_id]]


def test_perplexity_rejects_an_empty_text_collection():
    runtime = runtime_module.InferenceRuntime(object(), object())

    with pytest.raises(ValueError, match="at least one text"):
        runtime.evaluate_perplexity([])

    with pytest.raises(TypeError, match="not one string"):
        runtime.evaluate_perplexity("abc")

    with pytest.raises(TypeError, match="only strings"):
        runtime.evaluate_perplexity(["abc", 3])


def test_perplexity_restores_model_mode_after_model_error():
    tokenizer = CharTokenizer.from_text("a")

    class FailingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(vocab_size=tokenizer.vocab_size, block_size=2)

        def forward(self, token_ids):
            assert not self.training
            raise RuntimeError("evaluation failed")

    model = FailingModel()
    runtime = runtime_module.InferenceRuntime(model, tokenizer)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        runtime.evaluate_perplexity(["a"])
    assert model.training


@pytest.mark.parametrize("invalid_output", ["shape", "non_finite"])
def test_perplexity_rejects_invalid_model_logits(invalid_output):
    tokenizer = CharTokenizer.from_text("a")

    class InvalidModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(vocab_size=tokenizer.vocab_size, block_size=2)

        def forward(self, token_ids):
            if invalid_output == "shape":
                return torch.zeros(1, tokenizer.vocab_size)
            logits = torch.zeros(1, token_ids.shape[1], tokenizer.vocab_size)
            logits[..., 0] = float("nan")
            return logits

    runtime = runtime_module.InferenceRuntime(InvalidModel(), tokenizer)
    expected_error = "shape" if invalid_output == "shape" else "non-finite"

    with pytest.raises(ValueError, match=expected_error):
        runtime.evaluate_perplexity(["a"])


def test_benchmark_separates_warmup_and_reports_measured_workload(monkeypatch):
    tokenizer = CharTokenizer.from_text("abc")

    class DeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    runtime = runtime_module.InferenceRuntime(DeviceModel(), tokenizer)
    calls = []

    def fake_generate_batch(prompts, *, max_new_tokens, **kwargs):
        calls.append((list(prompts), max_new_tokens))
        return [
            # The displayed text can expand a special token such as <UNK>;
            # benchmark accounting must use sampled IDs, not text length.
            runtime_module.GenerationResult(prompts[0], "<UNK>", "length", 1),
            runtime_module.GenerationResult(prompts[1], "c", "eos", 1),
        ]

    clock_values = iter([10.0, 10.1, 20.0, 20.2, 30.0, 30.4])
    monkeypatch.setattr(runtime, "generate_batch", fake_generate_batch)
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: next(clock_values))

    report = runtime.benchmark(
        ["a", "b"], max_new_tokens=4, warmup_runs=2, measured_runs=3
    )

    assert calls == [(["a", "b"], 4)] * 5
    assert report["device"] == "cpu"
    assert report["python_version"]
    assert report["torch_version"] == torch.__version__
    assert report["model_parameter_count"] == 1
    assert report["model_parameter_bytes"] == 4
    assert report["process_peak_rss_bytes"] is None or report["process_peak_rss_bytes"] > 0
    assert report["prompt_token_counts"] == [2, 2]
    assert report["prompt_count"] == 2
    assert report["max_new_tokens"] == 4
    assert report["warmup_runs"] == 2
    assert report["measured_runs"] == 3
    assert report["latencies_seconds"] == pytest.approx([0.1, 0.2, 0.4])
    assert report["latency_mean_seconds"] == pytest.approx(0.7 / 3)
    assert report["latency_p50_seconds"] == pytest.approx(0.2)
    assert report["latency_p95_seconds"] == pytest.approx(0.4)
    assert report["generated_tokens"] == 6
    assert report["tokens_per_second"] == pytest.approx(6 / 0.7)


def test_benchmark_from_path_separates_load_first_request_and_warmed_metrics(
    monkeypatch, tmp_path
):
    tokenizer = CharTokenizer.from_text("abc")
    runtime = runtime_module.InferenceRuntime(object(), tokenizer)
    model_path = tmp_path / "model.pt"
    calls = []

    def fake_load(cls, path):
        calls.append(("load", path))
        return runtime

    def fake_generate(prompt, *, max_new_tokens, **kwargs):
        calls.append(("first_request", prompt, max_new_tokens))
        return runtime_module.GenerationResult(prompt, "<UNK>", "length", 1)

    def fake_benchmark(prompts, *, max_new_tokens, warmup_runs, measured_runs):
        calls.append(
            (
                "warmed",
                list(prompts),
                max_new_tokens,
                warmup_runs,
                measured_runs,
            )
        )
        return {"latency_mean_seconds": 0.05, "tokens_per_second": 40.0}

    clock_values = iter([1.0, 1.25, 2.0, 2.4])
    monkeypatch.setattr(
        runtime_module.InferenceRuntime, "load", classmethod(fake_load)
    )
    monkeypatch.setattr(runtime, "generate", fake_generate)
    monkeypatch.setattr(runtime, "benchmark", fake_benchmark)
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: next(clock_values))

    report = runtime_module.InferenceRuntime.benchmark_from_path(
        model_path,
        ["first", "second"],
        max_new_tokens=4,
        warmup_runs=2,
        measured_runs=3,
    )

    assert calls == [
        ("load", model_path),
        ("first_request", "first", 4),
        ("warmed", ["first", "second"], 4, 2, 3),
    ]
    assert report == {
        "model_path": str(model_path),
        "cold_start_seconds": pytest.approx(0.25),
        "first_request_seconds": pytest.approx(0.4),
        "first_request_generated_tokens": 1,
        "warmed": {"latency_mean_seconds": 0.05, "tokens_per_second": 40.0},
    }


@pytest.mark.parametrize(
    ("prompts", "options", "error_type", "message"),
    [
        ([], {}, ValueError, "at least one prompt"),
        ("ab", {}, TypeError, "not one string"),
        (["a", 3], {}, TypeError, "only strings"),
        (["a"], {"max_new_tokens": 0}, ValueError, "positive integer"),
        (["a"], {"max_new_tokens": True}, ValueError, "positive integer"),
        (["a"], {"warmup_runs": -1}, ValueError, "non-negative integer"),
        (["a"], {"warmup_runs": True}, ValueError, "non-negative integer"),
        (["a"], {"measured_runs": 0}, ValueError, "positive integer"),
        (["a"], {"measured_runs": True}, ValueError, "positive integer"),
    ],
)
def test_benchmark_validates_workload(prompts, options, error_type, message):
    runtime = runtime_module.InferenceRuntime(object(), object())

    with pytest.raises(error_type, match=message):
        runtime.benchmark(prompts, **options)
