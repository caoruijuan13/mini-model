"""Run the fixed SGD/Adam comparison used by the stage 06 report."""

from __future__ import annotations

import json
from pathlib import Path

from train import ARTIFACTS_DIR, TrainingConfig, train_and_evaluate


def compare_optimizers(output_dir: str | Path = ARTIFACTS_DIR) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shared = {
        "batch_size": 32,
        "max_steps": 1_500,
        "eval_interval": 25,
        "patience_evaluations": 16,
        "seed": 7,
    }
    settings = {
        "sgd": TrainingConfig(optimizer="sgd", learning_rate=0.2, **shared),
        "adam": TrainingConfig(optimizer="adam", learning_rate=0.002, **shared),
    }
    results = {}
    for name, config in settings.items():
        results[name] = train_and_evaluate(
            training_config=config,
            checkpoint_path=output_dir / f"{name}_training_checkpoint.npz",
            model_path=output_dir / f"{name}_token_mlp.npz",
            tokenizer_path=output_dir / "tokenizer.json",
            report_path=output_dir / f"{name}_experiment_report.json",
        )
    comparison = {
        "controlled_variables": [
            "model architecture",
            "corpus split",
            "mini-batch order",
            "batch size",
            "initialization seed",
            "evaluation interval",
            "early-stopping patience",
        ],
        "optimizer_specific_learning_rates": {name: config.learning_rate for name, config in settings.items()},
        "results": results,
    }
    path = output_dir / "optimizer_comparison.json"
    path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return comparison


if __name__ == "__main__":
    comparison = compare_optimizers()
    for name, result in comparison["results"].items():
        print(
            f"{name}: best_step={result['best_step']}, "
            f"valid_perplexity={result['valid_perplexity']:.6f}, "
            f"test_perplexity={result['test_perplexity']:.6f}"
        )
