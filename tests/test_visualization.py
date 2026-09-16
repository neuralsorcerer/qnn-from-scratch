from pathlib import Path

import numpy as np
import pytest

from qnn.qnn import DataReuploadingQNN
from qnn.trainer import TrainingRecord
from qnn.visualization import plot_decision_boundary, plot_training_history


def test_plot_training_history_rejects_empty(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_training_history([], tmp_path)


def test_plot_training_history_writes_files(tmp_path: Path):
    history = [
        TrainingRecord(1, 0.7, 0.5, 0.8, 0.5, 0.2),
        TrainingRecord(2, 0.6, 0.6, 0.7, 0.6, 0.1),
    ]
    plot_training_history(history, tmp_path)
    assert (tmp_path / "loss_curve.png").exists()
    assert (tmp_path / "accuracy_curve.png").exists()


def test_decision_boundary_validates_shape(tmp_path: Path):
    model = DataReuploadingQNN(num_layers=1)
    with pytest.raises(ValueError):
        plot_decision_boundary(model, np.zeros((3, 3)), np.array([0, 1, 0]), tmp_path)


def test_decision_boundary_rejects_complex_raw_values(tmp_path: Path):
    model = DataReuploadingQNN(num_qubits=1, num_layers=1, num_features=2, seed=1)
    raw = np.array([[0.0 + 0.1j, 0.0], [0.2, -0.1]], dtype=np.complex128)
    with pytest.raises(TypeError, match="raw must be real-valued"):
        plot_decision_boundary(model, raw, np.array([0, 1]), tmp_path)


def test_training_history_rejects_complex_metrics(tmp_path: Path):
    history = [TrainingRecord(1, 0.5 + 0.1j, 0.5, 0.5, 0.5, 0.1)]  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="history metrics must be real-valued"):
        plot_training_history(history, tmp_path)


def test_decision_boundary_writes_file(tmp_path: Path):
    model = DataReuploadingQNN(num_qubits=1, num_layers=1, num_features=2, seed=2)
    raw = np.array([[-0.5, -0.5], [-0.2, 0.4], [0.3, -0.1], [0.7, 0.6]])
    y = np.array([0, 0, 1, 1])
    out = plot_decision_boundary(model, raw, y, tmp_path)
    assert out == tmp_path / "decision_boundary.png"
    assert out.is_file()
