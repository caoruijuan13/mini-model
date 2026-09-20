"""Stage 10 evaluation, generation, and benchmark runtime."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
import platform
from pathlib import Path
import statistics
import sys
import time
from typing import Literal

import torch

from projects.stage08_pytorch_transformer.generate import generate_ids, iter_generated_ids
from projects.stage08_pytorch_transformer.model import TorchCharTransformer
from projects.tokenization.char_tokenizer import CharTokenizer

DEFAULT_MODEL_PATH = "projects/stage08_pytorch_transformer/artifacts/char_transformer.pt"


@dataclass(frozen=True)
class GenerationResult:
    """Keep the input and newly generated text as separate fields."""

    input_text: str
    generated_text: str
    finish_reason: Literal["eos", "length"]
    generated_token_count: int


class InferenceRuntime:
    def __init__(self, model: TorchCharTransformer, tokenizer: CharTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, path: str | Path) -> InferenceRuntime:
        """Load the model, configuration, and bound character tokenizer."""
        # Check a shared artifact format/version and report
        # unsupported versions and tokenizer mismatches explicitly.
        model, tokenizer = TorchCharTransformer.load_with_tokenizer(path)
        return cls(model, tokenizer)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 100,
        seed: int = 7,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Generate one response from a text prompt."""
        # Encode with the saved tokenizer, reuse stage 08 sampling,
        # decode only new tokens, and identify EOS versus the length limit.
        initial_ids = self.tokenizer.encode(prompt, add_bos=True)
        eos_id = self.tokenizer.tokens.index(self.tokenizer.eos_token)
        generated_ids = generate_ids(
            self.model,
            initial_ids,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
        )
        new_ids = generated_ids[len(initial_ids) :]
        finish_reason = "length"
        if eos_id in new_ids:
            new_ids = new_ids[: new_ids.index(eos_id)]
            finish_reason = "eos"
        generated_text = self.tokenizer.decode(new_ids)
        return GenerationResult(
            input_text=prompt,
            generated_text=generated_text,
            finish_reason=finish_reason,
            generated_token_count=len(new_ids),
        )

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 100,
        seed: int = 7,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[GenerationResult]:
        """Generate one ordered result per prompt."""
        # Define per-request random seeds and EOS handling so
        # each result matches the corresponding single-request call.
        results = []
        for index, prompt in enumerate(prompts):
            result = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                seed=seed + index,
                temperature=temperature,
                top_k=top_k,
            )
            results.append(result)
        return results

    def stream_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 100,
        seed: int = 7,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Iterator[str]:
        """Yield newly generated text fragments in order."""
        initial_ids = self.tokenizer.encode(prompt, add_bos=True)
        eos_id = self.tokenizer.token_to_id[self.tokenizer.eos_token]
        for next_id in iter_generated_ids(
            self.model,
            initial_ids,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
        ):
            if next_id == eos_id:
                return
            yield self.tokenizer.decode([next_id])

    def evaluate_perplexity(self, texts: Sequence[str]) -> float:
        """Evaluate the frozen model on explicitly supplied held-out texts."""
        # Fix BOS/EOS and overlapping-window accounting, then
        # compute token-weighted NLL without training or optimizer state.
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not one string")
        if not texts:
            raise ValueError("texts must contain at least one text")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must contain only strings")
        bos_id = self.tokenizer.token_to_id[self.tokenizer.bos_token]
        eos_id = self.tokenizer.token_to_id[self.tokenizer.eos_token]
        device = next(self.model.parameters()).device
        vocab_size = self.model.config.vocab_size
        total_loss = 0.0
        target_count = 0

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for text in texts:
                    text_ids = [bos_id, *self.tokenizer.encode(text), eos_id]
                    for target_index in range(1, len(text_ids)):
                        context_start = max(0, target_index - self.model.config.block_size)
                        context = torch.tensor(
                            [text_ids[context_start:target_index]],
                            dtype=torch.long,
                            device=device,
                        )
                        logits = self.model(context)
                        expected_shape = (1, context.shape[1], vocab_size)
                        if logits.shape != expected_shape:
                            raise ValueError(
                                f"model logits must have shape {expected_shape}"
                            )
                        next_logits = logits[0, -1].float()
                        if not torch.isfinite(next_logits).all():
                            raise ValueError("model returned non-finite logits")
                        target = torch.tensor(
                            [text_ids[target_index]], dtype=torch.long, device=device
                        )
                        loss = torch.nn.functional.cross_entropy(
                            next_logits.unsqueeze(0), target, reduction="sum"
                        )
                        total_loss += float(loss.item())
                        target_count += 1
        finally:
            self.model.train(was_training)

        return math.exp(total_loss / target_count)

    @staticmethod
    def _validate_benchmark_inputs(
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        warmup_runs: int,
        measured_runs: int,
    ) -> tuple[str, ...]:
        """Validate and freeze a benchmark workload before any timed work."""
        if isinstance(prompts, str):
            raise TypeError("prompts must be a sequence of strings, not one string")
        if not prompts:
            raise ValueError("prompts must contain at least one prompt")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("prompts must contain only strings")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            isinstance(warmup_runs, bool)
            or not isinstance(warmup_runs, int)
            or warmup_runs < 0
        ):
            raise ValueError("warmup_runs must be a non-negative integer")
        if (
            isinstance(measured_runs, bool)
            or not isinstance(measured_runs, int)
            or measured_runs <= 0
        ):
            raise ValueError("measured_runs must be a positive integer")
        return tuple(prompts)

    @staticmethod
    def _process_peak_rss_bytes() -> int | None:
        """Return this process's lifetime peak RSS when the OS exposes it."""
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except (ImportError, OSError):
            return None
        # Linux reports KiB; macOS and the BSDs report bytes.
        return int(peak * 1024 if sys.platform.startswith("linux") else peak)

    @classmethod
    def benchmark_from_path(
        cls,
        path: str | Path,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 100,
        warmup_runs: int = 3,
        measured_runs: int = 10,
    ) -> dict[str, object]:
        """Measure runtime loading, the first request, and warmed inference.

        ``cold_start_seconds`` covers reading the inference bundle and creating
        the runtime. It does not clear the operating system's file cache, so the
        result describes the current process and machine state only.
        """
        workload = cls._validate_benchmark_inputs(
            prompts,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )

        started = time.perf_counter()
        runtime = cls.load(path)
        cold_start_seconds = time.perf_counter() - started

        # The first request is measured before warmup and kept separate from
        # the warmed latency distribution returned by benchmark().
        started = time.perf_counter()
        first_result = runtime.generate(workload[0], max_new_tokens=max_new_tokens)
        first_request_seconds = time.perf_counter() - started
        first_request_generated_tokens = first_result.generated_token_count

        warmed = runtime.benchmark(
            workload,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        return {
            "model_path": str(path),
            "cold_start_seconds": cold_start_seconds,
            "first_request_seconds": first_request_seconds,
            "first_request_generated_tokens": first_request_generated_tokens,
            "warmed": warmed,
        }

    def benchmark(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 100,
        warmup_runs: int = 3,
        measured_runs: int = 10,
    ) -> dict[str, object]:
        """Measure warmed inference for an already loaded runtime.

        The memory value is the process lifetime peak resident set size. It is
        useful as an environment observation, but it is not an isolated model
        allocation measurement.
        """
        prompts = self._validate_benchmark_inputs(
            prompts,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )

        # Warmup happens after loading and is deliberately excluded from every
        # recorded duration. It is not a cold-start measurement.
        for _ in range(warmup_runs):
            self.generate_batch(prompts, max_new_tokens=max_new_tokens)

        durations: list[float] = []
        generated_tokens = 0

        for _ in range(measured_runs):
            start = time.perf_counter()
            results = self.generate_batch(prompts, max_new_tokens=max_new_tokens)
            duration = time.perf_counter() - start
            durations.append(duration)
            generated_tokens += sum(
                result.generated_token_count for result in results
            )

        sorted_durations = sorted(durations)

        def nearest_rank(percentile: float) -> float:
            rank = max(1, math.ceil(percentile * len(sorted_durations)))
            return sorted_durations[rank - 1]

        total_duration = sum(durations)
        if total_duration == 0:
            tokens_per_second = float("inf") if generated_tokens else 0.0
        else:
            tokens_per_second = generated_tokens / total_duration

        parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )

        return {
            "device": str(next(self.model.parameters()).device),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "model_parameter_count": parameter_count,
            "model_parameter_bytes": parameter_bytes,
            "process_peak_rss_bytes": self._process_peak_rss_bytes(),
            "memory_measurement": "process lifetime peak RSS; includes non-model memory",
            "prompt_count": len(prompts),
            "prompt_token_counts": [
                len(self.tokenizer.encode(prompt, add_bos=True)) for prompt in prompts
            ],
            "max_new_tokens": max_new_tokens,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "latencies_seconds": durations,
            "latency_mean_seconds": statistics.mean(durations),
            "latency_p50_seconds": nearest_rank(0.50),
            "latency_p95_seconds": nearest_rank(0.95),
            "latency_std_seconds": (
                statistics.stdev(durations) if len(durations) > 1 else 0.0
            ),
            "generated_tokens": generated_tokens,
            "tokens_per_second": tokens_per_second,
        }
