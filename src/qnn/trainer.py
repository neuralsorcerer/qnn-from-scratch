"""Training loop for the QNN."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from qnn.metrics import accuracy_score, binary_cross_entropy, confusion_matrix_binary
from qnn.optim import Adam
from qnn.qnn import DataReuploadingQNN


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    epochs: int = 90
    learning_rate: float = 0.12
    batch_size: int = 0
    grad_clip: float = 2.0
    log_every: int = 10
    output_dir: str = "outputs/default_run"


@dataclass(frozen=True)
class TrainingRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    grad_norm: float


class Trainer:
    """Production-style training utility with deterministic logging and saving."""

    def __init__(self, model: DataReuploadingQNN, config: TrainingConfig) -> None:
        """Create a trainer with Adam optimizer and empty in-memory history."""
        self.model = model
        self.config = config
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.history: list[TrainingRecord] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> list[TrainingRecord]:
        """Train the model and return logged metric snapshots."""
        if self.config.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.config.log_every < 1:
            raise ValueError("log_every must be at least 1")
        if self.config.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")
        if len(X_train) == 0:
            raise ValueError("X_train must contain at least one sample")

        rng = np.random.default_rng(self.config.seed)
        batch_size = self.config.batch_size or len(X_train)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        batch_size = min(batch_size, len(X_train))

        for epoch in range(1, self.config.epochs + 1):
            indices = np.arange(len(X_train))
            rng.shuffle(indices)

            epoch_grad_norms: list[float] = []
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                grad = self.model.parameter_shift_gradient(X_train[batch_idx], y_train[batch_idx])
                grad = np.clip(grad, -self.config.grad_clip, self.config.grad_clip)
                epoch_grad_norms.append(float(np.linalg.norm(grad)))
                self.model.params = self.optimizer.step(self.model.params, grad)

            if self._should_log(epoch):
                record = self.evaluate(epoch, X_train, y_train, X_test, y_test)
                record = TrainingRecord(
                    epoch=record.epoch,
                    train_loss=record.train_loss,
                    train_accuracy=record.train_accuracy,
                    test_loss=record.test_loss,
                    test_accuracy=record.test_accuracy,
                    grad_norm=float(np.mean(epoch_grad_norms)),
                )
                self.history.append(record)

        return self.history

    def evaluate(
        self,
        epoch: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainingRecord:
        """Evaluate train/test losses and accuracies for one epoch."""
        train_proba = self.model.predict_proba(X_train)
        test_proba = self.model.predict_proba(X_test)
        return TrainingRecord(
            epoch=epoch,
            train_loss=binary_cross_entropy(train_proba, y_train),
            train_accuracy=accuracy_score(train_proba, y_train),
            test_loss=binary_cross_entropy(test_proba, y_test),
            test_accuracy=accuracy_score(test_proba, y_test),
            grad_norm=0.0,
        )

    def save_artifacts(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        raw_test: np.ndarray | None = None,
    ) -> Path:
        """Save metrics, config, params, and predictions to output_dir."""
        if not self.history:
            raise ValueError("No training history found; call fit() before save_artifacts()")

        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with (out / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        np.save(out / "trained_params.npy", self.model.params)

        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.history[0]).keys()))
            writer.writeheader()
            for record in self.history:
                writer.writerow(asdict(record))

        test_proba = self.model.predict_proba(X_test)
        pred = (test_proba >= 0.5).astype(np.int64)
        predictions = np.column_stack([y_test, pred, test_proba])
        if raw_test is not None:
            predictions = np.column_stack([raw_test, predictions])
            header = "raw_x0,raw_x1,y_true,y_pred,p_class_1"
        else:
            header = "y_true,y_pred,p_class_1"
        np.savetxt(
            out / "test_predictions.csv", predictions, delimiter=",", header=header, comments=""
        )

        summary = {
            "final": asdict(self.history[-1]),
            "confusion_matrix_test": confusion_matrix_binary(test_proba, y_test),
        }
        with (out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return out

    def _should_log(self, epoch: int) -> bool:
        """Return whether metrics should be recorded for ``epoch``."""
        return epoch == 1 or epoch == self.config.epochs or epoch % self.config.log_every == 0
