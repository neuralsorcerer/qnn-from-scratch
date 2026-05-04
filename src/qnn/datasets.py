"""Deterministic toy datasets for QNN experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    Args:
        num_samples: Number of rows to generate. Must be at least ``8``.
        seed: Random seed used for deterministic generation.
        kind: Decision-boundary family. One of ``linear``, ``vertical``,
            ``horizontal``, ``circle``, or ``sine``.
        noise: Standard deviation of Gaussian noise injected into the
            separating score.

    Returns:
        A tuple ``(X_angles, y, raw)`` where:
        - ``X_angles``: feature matrix scaled by ``pi`` for rotation gates.
        - ``y``: binary labels in ``{0, 1}``.
        - ``raw``: original features in approximately ``[-1, 1]``.

    Raises:
        ValueError: If ``num_samples < 8`` or ``kind`` is unsupported.
    """
    if num_samples < 8:
        raise ValueError("num_samples must be at least 8")
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.0, 1.0, size=(num_samples, 2))

    if kind == "linear":
        score = raw[:, 0] + raw[:, 1] + rng.normal(0.0, noise, size=num_samples)
        y = (score > 0.0).astype(np.int64)
    elif kind == "vertical":
        score = raw[:, 1] + rng.normal(0.0, noise, size=num_samples)
        y = (score > 0.0).astype(np.int64)
    elif kind == "horizontal":
        score = raw[:, 0] + rng.normal(0.0, noise, size=num_samples)
        y = (score > 0.0).astype(np.int64)
    elif kind == "circle":
        radius = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)
        y = (radius > 0.68 + rng.normal(0.0, noise, size=num_samples)).astype(np.int64)
    elif kind == "sine":
        score = np.sin(np.pi * raw[:, 0]) + np.cos(np.pi * raw[:, 1])
        score += rng.normal(0.0, noise, size=num_samples)
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
    """Split aligned feature/label arrays into train and test partitions.

    Args:
        X: Angle-encoded features, shape ``(n_samples, n_features)``.
        y: Binary labels, shape ``(n_samples,)``.
        raw: Unscaled raw features aligned with ``X``.
        test_size: Fraction of rows to allocate to the test split.
        seed: Random seed used when ``shuffle`` is enabled.
        shuffle: Whether to shuffle rows before splitting.

    Returns:
        A :class:`DatasetBundle` containing train/test slices for all arrays.

    Raises:
        ValueError: If ``test_size`` is outside ``(0, 1)`` or input lengths
            are inconsistent.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    n = len(X)
    if not (len(y) == n and len(raw) == n):
        raise ValueError("X, y, and raw must have the same first dimension")

    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_test = max(1, int(round(n * test_size)))
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
