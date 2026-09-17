"""Binary losses and metrics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qnn._validation import real_array, real_scalar

FloatArray = npt.NDArray[np.float64]


def _validate_binary_labels(y_true: np.ndarray, expected_len: int) -> FloatArray:
    y = real_array("y_true", y_true).reshape(-1)
    if len(y) != expected_len:
        raise ValueError(f"y_true must contain {expected_len} elements, got {len(y)}")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_true must contain only finite values")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y_true must contain only binary labels 0 or 1")
    return y


def _validate_probabilities(proba: np.ndarray) -> FloatArray:
    p = real_array("proba", proba).reshape(-1)
    if len(p) == 0:
        raise ValueError("proba must contain at least one element")
    if not np.all(np.isfinite(p)):
        raise ValueError("proba must contain only finite values")
    if not np.all((0.0 <= p) & (p <= 1.0)):
        raise ValueError("proba values must lie in [0, 1]")
    return p


def _validate_eps(eps: float) -> float:
    eps = real_scalar("eps", eps)
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be a finite value in (0, 0.5)")
    return eps


def _smooth_probabilities(p: FloatArray, eps: float) -> FloatArray:
    """Affine map from ``[0,1]`` to ``[eps,1-eps]``.

    This is used instead of hard clipping so the implemented derivative is the
    derivative of the exact stabilized loss everywhere on the closed interval.
    """
    return np.asarray(eps + (1.0 - 2.0 * eps) * p, dtype=np.float64)


def binary_cross_entropy(proba: np.ndarray, y_true: np.ndarray, eps: float = 1e-9) -> float:
    """Mean epsilon-smoothed Bernoulli negative log-likelihood."""
    p = _validate_probabilities(proba)
    y = _validate_binary_labels(y_true, len(p))
    eps = _validate_eps(eps)
    p_safe = _smooth_probabilities(p, eps)
    loss = -(y * np.log(p_safe) + (1.0 - y) * np.log1p(-p_safe))
    return float(np.mean(loss))


def binary_cross_entropy_derivative(
    proba: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-9,
) -> FloatArray:
    """Derivative of the implemented mean BCE with respect to raw probabilities."""
    p = _validate_probabilities(proba)
    y = _validate_binary_labels(y_true, len(p))
    eps = _validate_eps(eps)
    p_safe = _smooth_probabilities(p, eps)
    scale = 1.0 - 2.0 * eps
    grad = scale * (p_safe - y) / (p_safe * (1.0 - p_safe)) / len(y)
    return np.asarray(grad, dtype=np.float64)


def _validate_threshold(threshold: float) -> float:
    threshold = real_scalar("threshold", threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite value in [0, 1]")
    return threshold


def accuracy_score(proba: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    """Binary classification accuracy at ``threshold``."""
    p = _validate_probabilities(proba)
    y = _validate_binary_labels(y_true, len(p)).astype(np.int64)
    threshold = _validate_threshold(threshold)
    pred = (p >= threshold).astype(np.int64)
    return float(np.mean(pred == y))


def confusion_matrix_binary(
    proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Return ``tn``, ``fp``, ``fn``, and ``tp`` counts."""
    p = _validate_probabilities(proba)
    y = _validate_binary_labels(y_true, len(p)).astype(np.int64)
    threshold = _validate_threshold(threshold)
    pred = (p >= threshold).astype(np.int64)
    return {
        "tn": int(np.sum((pred == 0) & (y == 0))),
        "fp": int(np.sum((pred == 1) & (y == 0))),
        "fn": int(np.sum((pred == 0) & (y == 1))),
        "tp": int(np.sum((pred == 1) & (y == 1))),
    }
