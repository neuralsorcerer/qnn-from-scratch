"""Command-line interface for qnn-from-scratch."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from qnn.datasets import make_classification_dataset, train_test_split
from qnn.io import load_json
from qnn.qnn import DataReuploadingQNN
from qnn.trainer import Trainer, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="qnn",
        description="Train a from-scratch NumPy Quantum Neural Network.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the QNN on a deterministic toy dataset.")
    train.add_argument(
        "--config", type=str, default="configs/default.json", help="Path to JSON config."
    )
    train.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    train.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    train.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    train.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")

    predict = sub.add_parser("predict", help="Predict class probabilities from saved parameters.")
    predict.add_argument("--params", type=str, required=True, help="Path to trained_params.npy.")
    predict.add_argument(
        "--x0", type=float, required=True, help="Raw feature x0 in approximately [-1, 1]."
    )
    predict.add_argument(
        "--x1", type=float, required=True, help="Raw feature x1 in approximately [-1, 1]."
    )
    predict.add_argument("--num-qubits", type=int, default=2)
    predict.add_argument("--num-layers", type=int, default=3)

    return parser


def train_command(args: argparse.Namespace) -> int:
    """Run end-to-end training from CLI arguments and persist artifacts."""
    cfg = load_json(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate

    X, y, raw = make_classification_dataset(
        num_samples=int(cfg.get("num_samples", 96)),
        seed=int(cfg.get("seed", 7)),
        kind=str(cfg.get("dataset", "linear")),
        noise=float(cfg.get("noise", 0.08)),
    )
    split = train_test_split(
        X,
        y,
        raw,
        test_size=float(cfg.get("test_size", 0.25)),
        seed=int(cfg.get("seed", 7)),
    )

    model = DataReuploadingQNN(
        num_qubits=int(cfg.get("num_qubits", 2)),
        num_layers=int(cfg.get("num_layers", 3)),
        num_features=2,
        observable_wire=min(1, int(cfg.get("num_qubits", 2)) - 1),
        seed=int(cfg.get("seed", 7)),
    )

    train_cfg = TrainingConfig(
        seed=int(cfg.get("seed", 7)),
        epochs=int(cfg.get("epochs", 90)),
        learning_rate=float(cfg.get("learning_rate", 0.12)),
        batch_size=int(cfg.get("batch_size", 0)),
        output_dir=str(cfg.get("output_dir", "outputs/default_run")),
        log_every=int(cfg.get("log_every", 10)),
    )

    trainer = Trainer(model, train_cfg)
    history = trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    output_dir = trainer.save_artifacts(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        raw_test=split.raw_test,
    )

    if not args.no_plots:
        from qnn.visualization import plot_decision_boundary, plot_training_history

        plot_training_history(history, output_dir)
        plot_decision_boundary(model, raw, y, output_dir)

    print("Training complete")
    print(f"Output directory: {output_dir}")
    print(json.dumps(asdict(history[-1]), indent=2))
    return 0


def predict_command(args: argparse.Namespace) -> int:
    """Load saved parameters and print one-sample prediction JSON."""
    model = DataReuploadingQNN(
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        num_features=2,
        observable_wire=min(1, args.num_qubits - 1),
    )
    model.params = np.load(args.params)
    x = np.array([[args.x0, args.x1]], dtype=np.float64) * np.pi
    proba = float(model.predict_proba(x)[0])
    pred = int(proba >= 0.5)
    print(json.dumps({"p_class_1": proba, "prediction": pred}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint that dispatches to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return train_command(args)
    if args.command == "predict":
        return predict_command(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
