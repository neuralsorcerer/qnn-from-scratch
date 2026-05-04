"""Small optimizers implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SGD:
    learning_rate: float = 0.1

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return one SGD update step for the provided parameters."""
        return params - self.learning_rate * grad


@dataclass
class Adam:
    learning_rate: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: np.ndarray | None = field(default=None, init=False)
    v: np.ndarray | None = field(default=None, init=False)

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return one Adam update step, maintaining first/second moments."""
        if self.m is None or self.v is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)

        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
