from pathlib import Path

import numpy as np

from qnn.datasets import make_classification_dataset, train_test_split
from qnn.qnn import DataReuploadingQNN
from qnn.trainer import Trainer, TrainingConfig


def test_training_saves_artifacts(tmp_path: Path):
    """Ensure fit plus save_artifacts produces the expected output files."""
    X, y, raw = make_classification_dataset(num_samples=24, seed=5)
    split = train_test_split(X, y, raw, test_size=0.25, seed=5)
    model = DataReuploadingQNN(num_layers=1, seed=5)
    trainer = Trainer(
        model,
        TrainingConfig(epochs=2, learning_rate=0.05, output_dir=str(tmp_path), log_every=1),
    )
    history = trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    assert len(history) == 2
    out = trainer.save_artifacts(split.X_train, split.y_train, split.X_test, split.y_test)
    assert (out / "metrics.csv").exists()
    assert (out / "trained_params.npy").exists()
    assert (out / "summary.json").exists()


def test_fit_rejects_empty_training_set(tmp_path: Path):
    """Verify fit rejects empty training batches with a clear ValueError."""
    model = DataReuploadingQNN(num_layers=1, seed=5)
    trainer = Trainer(model, TrainingConfig(epochs=1, output_dir=str(tmp_path), log_every=1))

    try:
        trainer.fit(
            X_train=np.empty((0, 2), dtype=np.float64),
            y_train=np.empty((0,), dtype=np.float64),
            X_test=np.array([[0.0, 0.0]], dtype=np.float64),
            y_test=np.array([0], dtype=np.float64),
        )
    except ValueError as exc:
        assert "X_train must contain at least one sample" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty training set")


def test_fit_rejects_invalid_log_every(tmp_path: Path):
    """Verify fit rejects non-positive log intervals."""
    X, y, raw = make_classification_dataset(num_samples=12, seed=2)
    split = train_test_split(X, y, raw, test_size=0.25, seed=2)
    model = DataReuploadingQNN(num_layers=1, seed=2)
    trainer = Trainer(model, TrainingConfig(epochs=1, output_dir=str(tmp_path), log_every=0))

    try:
        trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    except ValueError as exc:
        assert "log_every must be at least 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid log_every")


def test_save_artifacts_requires_history(tmp_path: Path):
    """Verify save_artifacts requires at least one completed training run."""
    X, y, raw = make_classification_dataset(num_samples=12, seed=3)
    split = train_test_split(X, y, raw, test_size=0.25, seed=3)
    model = DataReuploadingQNN(num_layers=1, seed=3)
    trainer = Trainer(model, TrainingConfig(epochs=1, output_dir=str(tmp_path), log_every=1))

    try:
        trainer.save_artifacts(split.X_train, split.y_train, split.X_test, split.y_test)
    except ValueError as exc:
        assert "call fit() before save_artifacts()" in str(exc)
    else:
        raise AssertionError("Expected ValueError when history is missing")


def test_fit_rejects_negative_grad_clip(tmp_path: Path):
    """Verify fit rejects a negative gradient clipping threshold."""
    X, y, raw = make_classification_dataset(num_samples=12, seed=8)
    split = train_test_split(X, y, raw, test_size=0.25, seed=8)
    model = DataReuploadingQNN(num_layers=1, seed=8)
    trainer = Trainer(
        model, TrainingConfig(epochs=1, output_dir=str(tmp_path), log_every=1, grad_clip=-1.0)
    )

    try:
        trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    except ValueError as exc:
        assert "grad_clip must be non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative grad_clip")
