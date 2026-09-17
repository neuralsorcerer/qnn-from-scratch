import json
from pathlib import Path

import numpy as np
import pytest

from qnn.datasets import make_classification_dataset, train_test_split
from qnn.qnn import DataReuploadingQNN
from qnn.trainer import Trainer, TrainingConfig


def _small_problem(seed=5):
    X, y, raw = make_classification_dataset(num_samples=24, seed=seed, kind="vertical", noise=0.02)
    return train_test_split(X, y, raw, test_size=0.25, seed=seed)


def test_training_saves_complete_artifacts(tmp_path: Path):
    split = _small_problem()
    model = DataReuploadingQNN(num_layers=1, seed=5)
    trainer = Trainer(
        model,
        TrainingConfig(epochs=2, learning_rate=0.05, output_dir=str(tmp_path), log_every=1),
    )
    history = trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    assert len(history) == 2
    out = trainer.save_artifacts(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        raw_test=split.raw_test,
        experiment_config={"dataset": "vertical", "num_samples": 24},
    )
    expected = {
        "metrics.csv",
        "trained_params.npy",
        "summary.json",
        "config.json",
        "test_predictions.csv",
    }
    assert expected.issubset({p.name for p in out.iterdir()})

    config = json.loads((out / "config.json").read_text())
    assert config["experiment"]["dataset"] == "vertical"
    assert config["model"]["num_layers"] == 1
    assert config["training"]["epochs"] == 2
    assert "python" in config["runtime"] and "numpy" in config["runtime"]

    summary = json.loads((out / "summary.json").read_text())
    assert "confusion_matrix_train" in summary
    assert "confusion_matrix_test" in summary


def test_fit_records_preclip_gradient_norm(tmp_path: Path):
    split = _small_problem(seed=6)
    model = DataReuploadingQNN(num_layers=1, seed=6)
    raw_grad = model.parameter_shift_gradient(split.X_train, split.y_train)
    raw_norm = float(np.linalg.norm(raw_grad))
    trainer = Trainer(
        model,
        TrainingConfig(
            epochs=1,
            learning_rate=0.01,
            grad_clip=1e-8,
            output_dir=str(tmp_path),
            log_every=1,
        ),
    )
    history = trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    np.testing.assert_allclose(history[0].grad_norm, raw_norm, rtol=1e-12, atol=1e-12)


def test_fit_resets_optimizer_and_history_between_calls(tmp_path: Path):
    split = _small_problem(seed=7)
    model = DataReuploadingQNN(num_layers=1, seed=7)
    trainer = Trainer(model, TrainingConfig(epochs=1, output_dir=str(tmp_path), log_every=1))
    first = trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    second = trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    assert len(first) == len(second) == 1
    assert trainer.optimizer.t == 1


def test_save_artifacts_requires_history(tmp_path: Path):
    split = _small_problem(seed=8)
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=8),
        TrainingConfig(epochs=1, output_dir=str(tmp_path)),
    )
    with pytest.raises(ValueError, match="call fit"):
        trainer.save_artifacts(split.X_train, split.y_train, split.X_test, split.y_test)


def test_training_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError):
        TrainingConfig(grad_clip=-1.0)


def test_training_config_normalizes_numpy_scalars_for_json(tmp_path: Path):
    cfg = TrainingConfig(
        seed=np.int64(2),
        epochs=np.int64(1),
        learning_rate=np.float64(0.05),
        batch_size=np.int64(0),
        grad_clip=np.float64(2.0),
        log_every=np.int64(1),
        output_dir=f"  {tmp_path}  ",
    )
    assert type(cfg.seed) is int
    assert type(cfg.epochs) is int
    assert type(cfg.learning_rate) is float
    assert cfg.output_dir == str(tmp_path)


def test_save_artifacts_validates_raw_before_creating_output_directory(tmp_path: Path):
    split = _small_problem(seed=9)
    out = tmp_path / "should_not_exist"
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=9),
        TrainingConfig(epochs=1, output_dir=str(out), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    bad_raw = np.full((len(split.y_test), 2), np.nan)
    with pytest.raises(ValueError, match="raw_test"):
        trainer.save_artifacts(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            raw_test=bad_raw,
        )
    assert not out.exists()


def test_save_artifacts_rejects_data_different_from_fit(tmp_path: Path):
    split = _small_problem(seed=10)
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=10),
        TrainingConfig(epochs=1, output_dir=str(tmp_path / "out"), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    changed = split.X_test.copy()
    changed[0, 0] += 1e-3
    with pytest.raises(ValueError, match="must match"):
        trainer.save_artifacts(split.X_train, split.y_train, changed, split.y_test)


def test_save_artifacts_rejects_parameters_changed_after_fit(tmp_path: Path):
    split = _small_problem(seed=11)
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=11),
        TrainingConfig(epochs=1, output_dir=str(tmp_path / "out"), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    trainer.model.params[0, 0, 0] += 1e-3
    with pytest.raises(ValueError, match="changed after fit"):
        trainer.save_artifacts(split.X_train, split.y_train, split.X_test, split.y_test)


def test_fit_rejects_complex_labels_without_silent_coercion(tmp_path: Path):
    model = DataReuploadingQNN(num_layers=1, seed=4)
    trainer = Trainer(model, TrainingConfig(epochs=1, output_dir=str(tmp_path), log_every=1))
    X = np.array([[0.0, 0.0], [0.1, -0.1]], dtype=np.float64)
    with pytest.raises(TypeError, match="y_train must be real-valued"):
        trainer.fit(X, np.array([0 + 0j, 1 + 0.1j]), X, np.array([0, 1]))


def test_training_config_rejects_numeric_strings(tmp_path: Path):
    with pytest.raises(TypeError):
        TrainingConfig(learning_rate="0.1", output_dir=str(tmp_path))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TrainingConfig(grad_clip="2", output_dir=str(tmp_path))  # type: ignore[arg-type]


def test_save_artifacts_removes_stale_managed_plots(tmp_path: Path):
    split = _small_problem(seed=10)
    for name in ("loss_curve.png", "accuracy_curve.png", "decision_boundary.png"):
        (tmp_path / name).write_bytes(b"stale")
    unrelated = tmp_path / "keep-me.txt"
    unrelated.write_text("keep", encoding="utf-8")

    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=10),
        TrainingConfig(epochs=1, output_dir=str(tmp_path), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    trainer.save_artifacts(split.X_train, split.y_train, split.X_test, split.y_test)

    for name in ("loss_curve.png", "accuracy_curve.png", "decision_boundary.png"):
        assert not (tmp_path / name).exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_save_artifacts_rejects_nan_in_experiment_config(tmp_path: Path):
    split = _small_problem(seed=11)
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=11),
        TrainingConfig(epochs=1, output_dir=str(tmp_path), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    with pytest.raises(TypeError, match="JSON-serializable"):
        trainer.save_artifacts(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            experiment_config={"bad": float("nan")},
        )


def test_save_artifacts_rejects_raw_rows_not_corresponding_to_angle_features(tmp_path: Path):
    split = _small_problem(seed=12)
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=12),
        TrainingConfig(epochs=1, output_dir=str(tmp_path / "out"), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    wrong_raw = split.raw_test[::-1].copy()
    with pytest.raises(ValueError, match="raw_test must be the unscaled feature matrix"):
        trainer.save_artifacts(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            raw_test=wrong_raw,
        )


def test_save_artifacts_rejects_history_mutation_after_fit(tmp_path: Path):
    split = _small_problem(seed=13)
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=13),
        TrainingConfig(epochs=1, output_dir=str(tmp_path / "out"), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    trainer.history.clear()
    with pytest.raises(ValueError, match="No training history"):
        trainer.save_artifacts(split.X_train, split.y_train, split.X_test, split.y_test)


def test_save_artifacts_rejects_replaced_history_after_fit(tmp_path: Path):
    split = _small_problem(seed=14)
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=14),
        TrainingConfig(epochs=1, output_dir=str(tmp_path / "out"), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    record = trainer.history[0]
    trainer.history[0] = type(record)(
        record.epoch,
        record.train_loss,
        record.train_accuracy,
        record.test_loss,
        record.test_accuracy,
        record.grad_norm + 1.0,
    )
    with pytest.raises(ValueError, match="training history changed after fit"):
        trainer.save_artifacts(split.X_train, split.y_train, split.X_test, split.y_test)


def test_save_artifacts_rejects_architecture_mutation_after_fit(tmp_path: Path):
    split = _small_problem(seed=15)
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=15),
        TrainingConfig(epochs=1, output_dir=str(tmp_path / "out"), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    trainer.model.init_scale += 0.1
    with pytest.raises(ValueError, match="architecture changed after fit"):
        trainer.save_artifacts(split.X_train, split.y_train, split.X_test, split.y_test)


def test_save_artifacts_rejects_contradictory_experiment_metadata(tmp_path: Path):
    split = _small_problem(seed=16)
    out = tmp_path / "out"
    trainer = Trainer(
        DataReuploadingQNN(num_layers=1, seed=16),
        TrainingConfig(epochs=1, output_dir=str(out), log_every=1),
    )
    trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)

    bad_configs = [
        ({"num_layers": 99}, "num_layers"),
        ({"epochs": 2}, "epochs"),
        ({"num_samples": 25}, "num_samples"),
        ({"test_size": 0.5}, "test_size"),
        ({"output_dir": str(tmp_path / "wrong")}, "output_dir"),
    ]
    for experiment_config, field in bad_configs:
        with pytest.raises(ValueError, match=field):
            trainer.save_artifacts(
                split.X_train,
                split.y_train,
                split.X_test,
                split.y_test,
                experiment_config=experiment_config,
            )

    assert not out.exists()
