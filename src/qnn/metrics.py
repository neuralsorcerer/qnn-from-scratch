"""Losses and metrics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_binary_labels(y_true: np.ndarray, expected_len: int) -> npt.NDArray[np.float64]:
    """Validate shape and binary values of labels, returning a flat float array."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if len(y) != expected_len:
        raise ValueError(f"y_true must contain {expected_len} elements, got {len(y)}")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y_true must contain only binary labels 0 or 1")
    return y


def binary_cross_entropy(proba: np.ndarray, y_true: np.ndarray, eps: float = 1e-9) -> float:
    """Binary cross entropy averaged over samples."""
    p = np.asarray(proba, dtype=np.float64).reshape(-1)
    y = _validate_binary_labels(y_true, len(p))
    p = np.clip(p, eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def binary_cross_entropy_derivative(
    proba: np.ndarray, y_true: np.ndarray, eps: float = 1e-9
) -> npt.NDArray[np.float64]:
    """Derivative of mean BCE with respect to probabilities."""
    p = np.asarray(proba, dtype=np.float64).reshape(-1)
    y = _validate_binary_labels(y_true, len(p))
    p = np.clip(p, eps, 1.0 - eps)
    n = max(1, len(y))
    grad: npt.NDArray[np.float64] = np.asarray((p - y) / (p * (1.0 - p)) / n, dtype=np.float64)
    return grad


def accuracy_score(proba: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    """Binary classification accuracy."""
    p = np.asarray(proba, dtype=np.float64).reshape(-1)
    y = _validate_binary_labels(y_true, len(p)).astype(np.int64)
    return float(np.mean((p >= threshold).astype(np.int64) == y))


def confusion_matrix_binary(
    proba: np.ndarray, y_true: np.ndarray, threshold: float = 0.5
) -> dict[str, int]:
    """Return a small binary confusion matrix as a dictionary."""
    p = np.asarray(proba, dtype=np.float64).reshape(-1)
    y = _validate_binary_labels(y_true, len(p)).astype(np.int64)
    pred = (p >= threshold).astype(np.int64)
    return {
        "tn": int(np.sum((pred == 0) & (y == 0))),
        "fp": int(np.sum((pred == 1) & (y == 0))),
        "fn": int(np.sum((pred == 0) & (y == 1))),
        "tp": int(np.sum((pred == 1) & (y == 1))),
    }
