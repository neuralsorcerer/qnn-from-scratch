"""Training loop and artifact persistence for the QNN."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qnn._validation import integer_scalar, real_array, real_scalar
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

    def __post_init__(self) -> None:
        seed = integer_scalar("seed", self.seed, minimum=0)
        epochs = integer_scalar("epochs", self.epochs, minimum=1)
        batch_size = integer_scalar("batch_size", self.batch_size, minimum=0)
        log_every = integer_scalar("log_every", self.log_every, minimum=1)
        learning_rate = real_scalar("learning_rate", self.learning_rate)
        grad_clip = real_scalar("grad_clip", self.grad_clip)

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be a finite positive value")
        if grad_clip < 0.0:
            raise ValueError("grad_clip must be finite and non-negative; 0 disables clipping")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "log_every", log_every)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "grad_clip", grad_clip)
        object.__setattr__(self, "output_dir", self.output_dir.strip())


@dataclass(frozen=True)
class TrainingRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    grad_norm: float


class Trainer:
    """Deterministic Adam training with explicit parameter-shift gradients."""

    def __init__(self, model: DataReuploadingQNN, config: TrainingConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.history: list[TrainingRecord] = []
        self._fit_fingerprints: dict[str, str] | None = None
        self._fit_params_digest: str | None = None
        self._fit_architecture: dict[str, int | float] | None = None
        self._fit_history_digest: str | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> list[TrainingRecord]:
        """Train and return logged metric snapshots.

        ``grad_norm`` records the mean *pre-clipping* L2 gradient norm for the
        batches in each logged epoch. ``grad_clip`` is an elementwise magnitude
        cap; a value of zero disables clipping.
        """
        X_train, y_train = self._validate_supervised_data(X_train, y_train, "train")
        X_test, y_test = self._validate_supervised_data(X_test, y_test, "test")

        self.optimizer = Adam(learning_rate=self.config.learning_rate)
        self.history = []
        self._fit_fingerprints = None
        self._fit_params_digest = None
        self._fit_architecture = None
        self._fit_history_digest = None

        rng = np.random.default_rng(int(self.config.seed))
        batch_size = self.config.batch_size or len(X_train)
        batch_size = min(batch_size, len(X_train))

        for epoch in range(1, self.config.epochs + 1):
            indices = np.arange(len(X_train))
            rng.shuffle(indices)
            raw_grad_norms: list[float] = []

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                grad = self.model.parameter_shift_gradient(X_train[batch_idx], y_train[batch_idx])
                raw_grad_norms.append(float(np.linalg.norm(grad)))

                if self.config.grad_clip > 0.0:
                    grad = np.clip(grad, -self.config.grad_clip, self.config.grad_clip)
                self.model.params = self.optimizer.step(self.model.params, grad)

            if self._should_log(epoch):
                evaluated = self.evaluate(epoch, X_train, y_train, X_test, y_test)
                self.history.append(
                    TrainingRecord(
                        epoch=epoch,
                        train_loss=evaluated.train_loss,
                        train_accuracy=evaluated.train_accuracy,
                        test_loss=evaluated.test_loss,
                        test_accuracy=evaluated.test_accuracy,
                        grad_norm=float(np.mean(raw_grad_norms)),
                    )
                )

        self._fit_fingerprints = {
            "X_train": self._array_digest(X_train),
            "y_train": self._array_digest(y_train),
            "X_test": self._array_digest(X_test),
            "y_test": self._array_digest(y_test),
        }
        self._fit_params_digest = self._array_digest(self.model.params)
        self._fit_architecture = dict(self.model.architecture())
        self._fit_history_digest = self._history_digest(self.history)
        return list(self.history)

    def evaluate(
        self,
        epoch: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainingRecord:
        """Evaluate current parameters without mutating the model."""
        epoch = integer_scalar("epoch", epoch, minimum=0)
        X_train, y_train = self._validate_supervised_data(X_train, y_train, "train")
        X_test, y_test = self._validate_supervised_data(X_test, y_test, "test")
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
        experiment_config: dict[str, Any] | None = None,
    ) -> Path:
        """Persist model parameters, provenance, metrics, and predictions."""
        if not self.history:
            raise ValueError("No training history found; call fit() before save_artifacts()")

        X_train, y_train = self._validate_supervised_data(X_train, y_train, "train")
        X_test, y_test = self._validate_supervised_data(X_test, y_test, "test")
        self._validate_save_consistency(X_train, y_train, X_test, y_test)

        validated_raw_test: np.ndarray | None = None
        if raw_test is not None:
            validated_raw_test = real_array("raw_test", raw_test)
            if validated_raw_test.shape != X_test.shape:
                raise ValueError(
                    "raw_test must have exactly the same shape as X_test so rows and features align"
                )
            if not np.all(np.isfinite(validated_raw_test)):
                raise ValueError("raw_test must contain only finite values")
            if not np.allclose(validated_raw_test * np.pi, X_test, atol=1e-12, rtol=1e-12):
                raise ValueError(
                    "raw_test must be the unscaled feature matrix corresponding to X_test "
                    "(X_test = raw_test * pi)"
                )

        experiment = dict(experiment_config or {})
        try:
            json.dumps(experiment, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("experiment_config must contain JSON-serializable values") from exc
        self._validate_experiment_config(experiment, len(X_train), len(X_test))

        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for stale_name in ("loss_curve.png", "accuracy_curve.png", "decision_boundary.png"):
            stale_path = out / stale_name
            if stale_path.is_file():
                stale_path.unlink()

        resolved = {
            "experiment": experiment,
            "model": self.model.architecture(),
            "training": asdict(self.config),
            "data": {
                "train_rows": len(X_train),
                "monitor_rows": len(X_test),
                "evaluation_role": "monitoring_validation",
                "fingerprints": dict(self._fit_fingerprints or {}),
                "raw_monitoring": (
                    self._array_digest(validated_raw_test)
                    if validated_raw_test is not None
                    else None
                ),
            },
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "platform": sys.platform,
            },
        }
        self._write_json(out / "config.json", resolved)
        np.save(out / "trained_params.npy", self.model.params, allow_pickle=False)

        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.history[0]).keys()))
            writer.writeheader()
            for record in self.history:
                writer.writerow(asdict(record))

        train_proba = self.model.predict_proba(X_train)
        test_proba = self.model.predict_proba(X_test)
        pred = (test_proba >= 0.5).astype(np.int64)
        predictions = np.column_stack([y_test, pred, test_proba])
        header = "y_true,y_pred,p_class_1"

        if validated_raw_test is not None:
            predictions = np.column_stack([validated_raw_test, predictions])
            raw_header = ",".join(f"raw_x{i}" for i in range(validated_raw_test.shape[1]))
            header = f"{raw_header},{header}"

        np.savetxt(
            out / "test_predictions.csv",
            predictions,
            delimiter=",",
            header=header,
            comments="",
        )

        summary = {
            "final": asdict(self.history[-1]),
            "confusion_matrix_train": confusion_matrix_binary(train_proba, y_train),
            "confusion_matrix_test": confusion_matrix_binary(test_proba, y_test),
            "model": self.model.architecture(),
        }
        self._write_json(out / "summary.json", summary)
        return out

    def _validate_supervised_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = self.model._validate_batch(X)
        y = real_array(f"y_{name}", y).reshape(-1)
        if len(y) != len(X):
            raise ValueError(f"X_{name} and y_{name} must have matching row counts")
        accuracy_score(np.full(len(y), 0.5, dtype=np.float64), y)
        return X, y

    def _validate_save_consistency(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        if (
            self._fit_fingerprints is None
            or self._fit_params_digest is None
            or self._fit_architecture is None
            or self._fit_history_digest is None
        ):
            raise ValueError("fit provenance is unavailable; call fit() before save_artifacts()")

        current = {
            "X_train": self._array_digest(X_train),
            "y_train": self._array_digest(y_train),
            "X_test": self._array_digest(X_test),
            "y_test": self._array_digest(y_test),
        }
        if current != self._fit_fingerprints:
            raise ValueError(
                "save_artifacts data must match the arrays used by the most recent fit()"
            )
        if self._array_digest(self.model.params) != self._fit_params_digest:
            raise ValueError("model parameters changed after fit(); saved history would be stale")
        if self.model.architecture() != self._fit_architecture:
            raise ValueError(
                "model architecture changed after fit(); saved provenance would be stale"
            )
        if self._history_digest(self.history) != self._fit_history_digest:
            raise ValueError("training history changed after fit(); saved metrics would be stale")

        final = self.evaluate(self.history[-1].epoch, X_train, y_train, X_test, y_test)
        recorded = self.history[-1]
        for field in ("train_loss", "train_accuracy", "test_loss", "test_accuracy"):
            if not np.isclose(
                getattr(final, field), getattr(recorded, field), atol=1e-12, rtol=1e-12
            ):
                raise ValueError("training history does not match the current model and data")

    def _validate_experiment_config(
        self,
        experiment: dict[str, Any],
        train_rows: int,
        monitor_rows: int,
    ) -> None:
        """Reject experiment metadata that contradicts the realized run."""
        architecture = self.model.architecture()
        integer_expectations = {
            "num_qubits": int(architecture["num_qubits"]),
            "num_layers": int(architecture["num_layers"]),
            "num_features": int(architecture["num_features"]),
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "log_every": self.config.log_every,
            "num_samples": train_rows + monitor_rows,
        }
        for key, expected_int in integer_expectations.items():
            if key not in experiment:
                continue
            actual_int = integer_scalar(f"experiment_config[{key!r}]", experiment[key])
            if actual_int != expected_int:
                raise ValueError(
                    f"experiment_config field {key!r}={actual_int} "
                    f"contradicts realized value {expected_int}"
                )

        float_expectations = {
            "init_scale": float(architecture["init_scale"]),
            "learning_rate": self.config.learning_rate,
            "grad_clip": self.config.grad_clip,
        }
        for key, expected_float in float_expectations.items():
            if key not in experiment:
                continue
            actual_float = real_scalar(f"experiment_config[{key!r}]", experiment[key])
            if actual_float != expected_float:
                raise ValueError(
                    f"experiment_config field {key!r}={actual_float} "
                    f"contradicts realized value {expected_float}"
                )

        if "observable_wire" in experiment and experiment["observable_wire"] is not None:
            actual_wire = integer_scalar(
                "experiment_config['observable_wire']", experiment["observable_wire"], minimum=0
            )
            expected_wire = int(architecture["observable_wire"])
            if actual_wire != expected_wire:
                raise ValueError(
                    "experiment_config field 'observable_wire' contradicts the realized model"
                )

        if "output_dir" in experiment:
            output_dir = experiment["output_dir"]
            if not isinstance(output_dir, str):
                raise TypeError("experiment_config['output_dir'] must be a string")
            if output_dir.strip() != self.config.output_dir:
                raise ValueError(
                    "experiment_config field 'output_dir' contradicts the training configuration"
                )

        if "test_size" in experiment:
            test_size = real_scalar("experiment_config['test_size']", experiment["test_size"])
            if not 0.0 < test_size < 1.0:
                raise ValueError("experiment_config['test_size'] must lie in (0, 1)")
            total_rows = train_rows + monitor_rows
            expected_monitor_rows = min(
                total_rows - 1,
                max(1, int(round(total_rows * test_size))),
            )
            if monitor_rows != expected_monitor_rows:
                raise ValueError(
                    "experiment_config field 'test_size' contradicts the realized split row counts"
                )

    @staticmethod
    def _history_digest(history: list[TrainingRecord]) -> str:
        payload = json.dumps(
            [asdict(record) for record in history],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _array_digest(array: np.ndarray) -> str:
        canonical = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(str(canonical.dtype).encode("utf-8"))
        digest.update(repr(canonical.shape).encode("utf-8"))
        digest.update(canonical.tobytes(order="C"))
        return digest.hexdigest()

    def _should_log(self, epoch: int) -> bool:
        return epoch == 1 or epoch == self.config.epochs or epoch % self.config.log_every == 0

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
