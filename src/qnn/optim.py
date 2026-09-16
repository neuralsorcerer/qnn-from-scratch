"""Small first-order optimizers implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qnn._validation import real_array, real_scalar


def _positive_finite(name: str, value: float) -> float:
    value = real_scalar(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")
    return value


def _validate_step_inputs(params: np.ndarray, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    params = real_array("params", params)
    grad = real_array("grad", grad)
    if params.shape != grad.shape:
        raise ValueError(
            f"params and grad must have the same shape, got {params.shape} and {grad.shape}"
        )
    if params.size == 0:
        raise ValueError("params and grad must be non-empty")
    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(grad)):
        raise ValueError("params and grad must contain only finite values")
    return params, grad


@dataclass
class SGD:
    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        self.learning_rate = _positive_finite("learning_rate", self.learning_rate)

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        params, grad = _validate_step_inputs(params, grad)
        updated = params - self.learning_rate * grad
        if not np.all(np.isfinite(updated)):
            raise FloatingPointError("non-finite SGD update encountered")
        return updated


@dataclass
class Adam:
    learning_rate: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = field(default=0, init=False)
    m: np.ndarray | None = field(default=None, init=False)
    v: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.learning_rate = _positive_finite("learning_rate", self.learning_rate)
        for name in ("beta1", "beta2"):
            value = real_scalar(name, getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1)")
            setattr(self, name, value)
        self.eps = _positive_finite("eps", self.eps)

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        params, grad = _validate_step_inputs(params, grad)
        if self.m is None or self.v is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        elif self.m.shape != params.shape or self.v.shape != params.shape:
            raise ValueError("optimizer state shape does not match params shape")

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        updated = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        if not np.all(np.isfinite(updated)):
            raise FloatingPointError("non-finite Adam update encountered")
        return updated
