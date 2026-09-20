"""Generate the reproducible evaluation and inference report for stage 10."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from projects.stage07_char_transformer.data import load_corpus, make_text_splits
from projects.stage08_pytorch_transformer.model import FORMAT_NAME, FORMAT_VERSION

from .runtime import DEFAULT_MODEL_PATH, InferenceRuntime

DEFAULT_REPORT_PATH = Path(__file__).with_name("artifacts") / "inference_report.json"
DEFAULT_PROMPTS = ("We hold", "Governments")


def build_report(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Evaluate fixed splits, check streaming, and measure a fixed workload."""
    corpus = load_corpus()
    train_text, valid_text, test_text = make_text_splits(corpus)
    runtime = InferenceRuntime.load(model_path)

    generation_options = {
        "max_new_tokens": 40,
        "seed": 7,
        "temperature": 1.0,
        "top_k": 5,
    }
    samples = []
    for prompt in DEFAULT_PROMPTS:
        result = runtime.generate(prompt, **generation_options)
        streamed_text = "".join(runtime.stream_generate(prompt, **generation_options))
        samples.append(
            {
                **asdict(result),
                "streamed_text": streamed_text,
                "stream_matches_generated_text": streamed_text == result.generated_text,
            }
        )

    report: dict[str, Any] = {
        "artifact": {
            "path": str(model_path),
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "config": asdict(runtime.model.config),
        },
        "data": {
            "source": "data/declaration_excerpt.txt",
            "split": "sequential 80/10/10",
            "character_counts": {
                "train": len(train_text),
                "validation": len(valid_text),
                "test": len(test_text),
            },
        },
        "evaluation": {
            "validation_perplexity": runtime.evaluate_perplexity([valid_text]),
            "test_perplexity": runtime.evaluate_perplexity([test_text]),
            "accounting": "each split gets BOS/EOS and each target token is counted once",
        },
        "generation": {
            "options": generation_options,
            "samples": samples,
        },
        "performance": InferenceRuntime.benchmark_from_path(
            model_path,
            DEFAULT_PROMPTS,
            max_new_tokens=16,
            warmup_runs=3,
            measured_runs=10,
        ),
        "scope": (
            "Results describe this tiny CPU teaching model on the recorded environment; "
            "they do not estimate large-model, GPU, network, or concurrent service performance."
        ),
    }
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))
