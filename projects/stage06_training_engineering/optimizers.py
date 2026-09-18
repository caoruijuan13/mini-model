"""Small, explicit SGD and Adam implementations."""

from __future__ import annotations

from typing import Protocol

import numpy as np


Parameters = dict[str, np.ndarray]


class Optimizer(Protocol):
    name: str
    step_count: int

    def step(self, params: Parameters, gradients: Parameters) -> None: ...
    def state_dict(self) -> dict[str, np.ndarray]: ...
    def load_state_dict(self, state: dict[str, np.ndarray], params: Parameters) -> None: ...


def _validate(params: Parameters, gradients: Parameters) -> None:
    if set(params) != set(gradients):
        raise ValueError("gradient keys must match parameter keys")
    for name, parameter in params.items():
        gradient = gradients[name]
        if gradient.shape != parameter.shape:
            raise ValueError(f"gradient shape mismatch for {name}")
        if not np.all(np.isfinite(gradient)):
            raise ValueError(f"gradient for {name} contains non-finite values")


class SGD:
    name = "sgd"

    def __init__(self, learning_rate: float) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.learning_rate = float(learning_rate)
        self.step_count = 0

    def step(self, params: Parameters, gradients: Parameters) -> None:
        _validate(params, gradients)
        for name in params:
            params[name] -= gradients[name] * self.learning_rate
        self.step_count += 1
            
    def state_dict(self) -> dict[str, np.ndarray]:
        return {"step_count": np.asarray(self.step_count, dtype=np.int64)}

    def load_state_dict(self, state: dict[str, np.ndarray], params: Parameters) -> None:
        del params
        if set(state) != {"step_count"}:
            raise ValueError("invalid SGD state")
        self.step_count = int(state["step_count"].item())


class Adam:
    name = "adam"

    def __init__(
        self,
        learning_rate: float,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta values must be in [0, 1)")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.step_count = 0
        self.first_moment: Parameters = {}
        self.second_moment: Parameters = {}

    def step(self, params: Parameters, gradients: Parameters) -> None:
        _validate(params, gradients)
        if not self.first_moment:
            self.first_moment = {name: np.zeros_like(value) for name, value in params.items()}
            self.second_moment = {name: np.zeros_like(value) for name, value in params.items()}
        self.step_count += 1
        for name, gradient in gradients.items():
            self.first_moment[name] = (
                self.beta1 * self.first_moment[name] + (1.0 - self.beta1) * gradient
            )
            self.second_moment[name] = (
                self.beta2 * self.second_moment[name]
                + (1.0 - self.beta2) * gradient**2
            )
            corrected_first = self.first_moment[name] / (1.0 - self.beta1**self.step_count)
            corrected_second = self.second_moment[name] / (1.0 - self.beta2**self.step_count)
            params[name] -= self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + self.epsilon
            )

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {"step_count": np.asarray(self.step_count, dtype=np.int64)}
        for name, value in self.first_moment.items():
            state[f"first_moment.{name}"] = value.copy()
            state[f"second_moment.{name}"] = self.second_moment[name].copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray], params: Parameters) -> None:
        self.step_count = int(state["step_count"].item())
        expected = {"step_count"}
        if self.step_count > 0:
            expected |= {f"first_moment.{name}" for name in params}
            expected |= {f"second_moment.{name}" for name in params}
        if set(state) != expected:
            raise ValueError("invalid Adam state")
        self.first_moment = {}
        self.second_moment = {}
        if self.step_count > 0:
            for name, parameter in params.items():
                first = state[f"first_moment.{name}"]
                second = state[f"second_moment.{name}"]
                if first.shape != parameter.shape or second.shape != parameter.shape:
                    raise ValueError(f"Adam state shape mismatch for {name}")
                self.first_moment[name] = first.copy()
                self.second_moment[name] = second.copy()


def make_optimizer(name: str, learning_rate: float) -> Optimizer:
    if name == "sgd":
        return SGD(learning_rate)
    if name == "adam":
        return Adam(learning_rate)
    raise ValueError(f"unsupported optimizer: {name}")
