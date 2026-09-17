"""Visualization helpers for two-feature QNN experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from qnn._validation import real_array
from qnn.qnn import DataReuploadingQNN
from qnn.trainer import TrainingRecord


def _new_agg_figure(figsize: tuple[float, float]) -> Figure:
    """Create a figure backed explicitly by Agg for headless file rendering."""
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure


def plot_training_history(history: list[TrainingRecord], output_dir: str | Path) -> Path:
    """Save train/test loss and accuracy curves."""
    if not history:
        raise ValueError("history must contain at least one training record")
    epochs = [record.epoch for record in history]
    if any(b <= a for a, b in zip(epochs, epochs[1:])):
        raise ValueError("history epochs must be strictly increasing")
    values = real_array(
        "history metrics",
        [
            [r.train_loss, r.train_accuracy, r.test_loss, r.test_accuracy, r.grad_norm]
            for r in history
        ],
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("history metrics must contain only finite values")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loss_path = out / "loss_curve.png"
    loss_figure = _new_agg_figure((7.0, 4.5))
    loss_ax = loss_figure.subplots()
    loss_ax.plot(epochs, [r.train_loss for r in history], marker="o", label="train")
    loss_ax.plot(epochs, [r.test_loss for r in history], marker="o", label="test")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Binary cross entropy")
    loss_ax.set_title("QNN training loss")
    loss_ax.legend()
    loss_figure.tight_layout()
    loss_figure.savefig(loss_path, dpi=160)

    acc_path = out / "accuracy_curve.png"
    acc_figure = _new_agg_figure((7.0, 4.5))
    acc_ax = acc_figure.subplots()
    acc_ax.plot(epochs, [r.train_accuracy for r in history], marker="o", label="train")
    acc_ax.plot(epochs, [r.test_accuracy for r in history], marker="o", label="test")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_ylim(0.0, 1.05)
    acc_ax.set_title("QNN classification accuracy")
    acc_ax.legend()
    acc_figure.tight_layout()
    acc_figure.savefig(acc_path, dpi=160)
    return out


def plot_decision_boundary(
    model: DataReuploadingQNN,
    raw: np.ndarray,
    y: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Save a two-dimensional probability surface and data overlay."""
    if model.num_features != 2:
        raise ValueError("decision-boundary plotting requires exactly two model features")
    raw = real_array("raw", raw)
    y = real_array("y", y).reshape(-1)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"raw must have shape (n_samples, 2), got {raw.shape}")
    if len(y) != len(raw):
        raise ValueError("raw and y must contain the same number of rows")
    if len(y) == 0:
        raise ValueError("raw and y must contain at least one sample")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(y)):
        raise ValueError("raw and y must contain only finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only binary labels 0 or 1")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    boundary_path = out / "decision_boundary.png"

    xs = np.linspace(-1.1, 1.1, 70)
    ys = np.linspace(-1.1, 1.1, 70)
    grid = np.array([[a, b] for b in ys for a in xs], dtype=np.float64)
    proba = model.predict_proba(grid * np.pi).reshape(len(ys), len(xs))

    figure = _new_agg_figure((6.0, 5.0))
    ax = figure.subplots()
    contour = ax.contourf(xs, ys, proba, levels=20, alpha=0.8)
    figure.colorbar(contour, ax=ax, label="p(class=1)")
    ax.scatter(raw[:, 0], raw[:, 1], c=y, edgecolors="k", s=35)
    ax.set_xlabel("raw x0")
    ax.set_ylabel("raw x1")
    ax.set_title("QNN decision boundary")
    figure.tight_layout()
    figure.savefig(boundary_path, dpi=160)
    return boundary_path
