"""Deterministic two-dimensional toy datasets for QNN experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnn._validation import integer_scalar, real_numeric_array, real_scalar


@dataclass(frozen=True)
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    raw_train: np.ndarray
    raw_test: np.ndarray


def make_classification_dataset(
    num_samples: int = 96,
    seed: int = 7,
    kind: str = "linear",
    noise: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic two-feature binary classification dataset.

    ``vertical`` denotes the vertical boundary ``x0 = 0`` and therefore labels
    by the sign of ``x0``. ``horizontal`` denotes ``x1 = 0`` and labels by the
    sign of ``x1``. Gaussian noise is added to the separating score before the
    binary label is formed.
    """
    num_samples = integer_scalar("num_samples", num_samples, minimum=8)
    seed = integer_scalar("seed", seed, minimum=0)
    if not isinstance(kind, str):
        raise TypeError("kind must be a string")
    noise = real_scalar("noise", noise)
    if noise < 0.0:
        raise ValueError("noise must be a finite non-negative value")

    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.0, 1.0, size=(num_samples, 2))
    jitter = rng.normal(0.0, noise, size=num_samples)

    if kind == "linear":
        score = raw[:, 0] + raw[:, 1] + jitter
        y = (score > 0.0).astype(np.int64)
    elif kind == "vertical":
        score = raw[:, 0] + jitter
        y = (score > 0.0).astype(np.int64)
    elif kind == "horizontal":
        score = raw[:, 1] + jitter
        y = (score > 0.0).astype(np.int64)
    elif kind == "circle":
        radius = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)
        score = radius - 0.68 + jitter
        y = (score > 0.0).astype(np.int64)
    elif kind == "sine":
        score = np.sin(np.pi * raw[:, 0]) + np.cos(np.pi * raw[:, 1]) + jitter
        y = (score > 0.0).astype(np.int64)
    else:
        raise ValueError("kind must be one of: linear, vertical, horizontal, circle, sine")

    X_angles = raw * np.pi
    return X_angles.astype(np.float64), y, raw.astype(np.float64)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    raw: np.ndarray,
    test_size: float = 0.25,
    seed: int = 7,
    shuffle: bool = True,
) -> DatasetBundle:
    """Split aligned arrays into non-empty train and test partitions.

    This helper is intentionally simple and does not perform stratification.
    """
    test_size = real_scalar("test_size", test_size)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be a finite value between 0 and 1")
    seed = integer_scalar("seed", seed, minimum=0)
    if not isinstance(shuffle, (bool, np.bool_)):
        raise TypeError("shuffle must be boolean")

    X = real_numeric_array("X", X)
    y = real_numeric_array("y", y)
    raw = real_numeric_array("raw", raw)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if raw.ndim != 2:
        raise ValueError(f"raw must be 2-D, got shape {raw.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(raw)) or not np.all(np.isfinite(y)):
        raise ValueError("X, y, and raw must contain only finite values")

    n = len(X)
    if n < 2:
        raise ValueError("at least two samples are required for a train/test split")
    if not (len(y) == n and len(raw) == n):
        raise ValueError("X, y, and raw must have the same first dimension")

    indices = np.arange(n)
    if bool(shuffle):
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_test = int(round(n * test_size))
    n_test = min(n - 1, max(1, n_test))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return DatasetBundle(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        raw_train=raw[train_idx],
        raw_test=raw[test_idx],
    )
