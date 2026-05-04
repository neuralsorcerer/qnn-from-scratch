"""Visualization helpers for QNN experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qnn.trainer import TrainingRecord


def plot_training_history(history: list[TrainingRecord], output_dir: str | Path) -> Path:
    """Plot and save loss/accuracy curves from training history.

    Args:
        history: Chronological metric snapshots produced during training.
        output_dir: Directory where PNG files will be written.

    Returns:
        The created output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    epochs = [r.epoch for r in history]

    loss_path = out / "loss_curve.png"
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [r.train_loss for r in history], marker="o", label="train")
    plt.plot(epochs, [r.test_loss for r in history], marker="o", label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross entropy")
    plt.title("QNN training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()

    acc_path = out / "accuracy_curve.png"
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [r.train_accuracy for r in history], marker="o", label="train")
    plt.plot(epochs, [r.test_accuracy for r in history], marker="o", label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("QNN classification accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=160)
    plt.close()

    return out


def plot_decision_boundary(model, raw: np.ndarray, y: np.ndarray, output_dir: str | Path) -> Path:
    """Plot and save a 2D decision boundary over raw feature space.

    Args:
        model: Trained model exposing ``predict_proba`` on angle-encoded input.
        raw: Raw feature matrix with two columns, used for scatter overlay.
        y: Binary labels aligned with ``raw``.
        output_dir: Directory where ``decision_boundary.png`` is written.

    Returns:
        Path to the generated decision-boundary image.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    boundary_path = out / "decision_boundary.png"

    xs = np.linspace(-1.1, 1.1, 70)
    ys = np.linspace(-1.1, 1.1, 70)
    grid = np.array([[a, b] for b in ys for a in xs], dtype=np.float64)
    proba = model.predict_proba(grid * np.pi).reshape(len(ys), len(xs))

    plt.figure(figsize=(6, 5))
    plt.contourf(xs, ys, proba, levels=20, alpha=0.8)
    plt.colorbar(label="p(class=1)")
    plt.scatter(raw[:, 0], raw[:, 1], c=y, edgecolors="k", s=35)
    plt.xlabel("raw x0")
    plt.ylabel("raw x1")
    plt.title("QNN decision boundary")
    plt.tight_layout()
    plt.savefig(boundary_path, dpi=160)
    plt.close()

    return boundary_path
