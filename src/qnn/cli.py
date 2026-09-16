"""Command-line interface for qnn-from-scratch."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from qnn.config import ExperimentConfig
from qnn.datasets import make_classification_dataset, train_test_split
from qnn.io import load_json
from qnn.qnn import DataReuploadingQNN
from qnn.trainer import Trainer, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qnn",
        description="Train and inspect a from-scratch NumPy QNN.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the QNN on a deterministic toy dataset.")
    train.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional path to a flat experiment JSON config; "
            "built-in defaults are used otherwise."
        ),
    )
    train.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    train.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    train.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override Adam learning rate.",
    )
    train.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")

    predict = sub.add_parser("predict", help="Predict from a saved parameter tensor.")
    predict.add_argument("--params", type=str, required=True, help="Path to trained_params.npy.")
    predict.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional saved config.json. If omitted, a sibling config.json is used when present.",
    )
    predict.add_argument("--x0", type=float, required=True, help="Raw feature x0.")
    predict.add_argument("--x1", type=float, required=True, help="Raw feature x1.")
    predict.add_argument(
        "--num-qubits",
        type=int,
        default=None,
        help="Optional architecture assertion; normally inferred from params.",
    )
    predict.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Optional architecture assertion; normally inferred from params.",
    )
    predict.add_argument(
        "--observable-wire",
        type=int,
        default=None,
        help=(
            "Measurement wire when saved metadata is unavailable; otherwise it is treated "
            "as an architecture assertion."
        ),
    )
    return parser


def train_command(args: argparse.Namespace) -> int:
    cfg = (
        ExperimentConfig.from_mapping(load_json(args.config))
        if args.config is not None
        else ExperimentConfig()
    )
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    if args.learning_rate is not None:
        cfg = replace(cfg, learning_rate=args.learning_rate)

    X, y, raw = make_classification_dataset(
        num_samples=cfg.num_samples,
        seed=cfg.seed,
        kind=cfg.dataset,
        noise=cfg.noise,
    )
    split = train_test_split(
        X,
        y,
        raw,
        test_size=cfg.test_size,
        seed=cfg.seed,
    )
    model = DataReuploadingQNN(
        num_qubits=cfg.num_qubits,
        num_layers=cfg.num_layers,
        num_features=cfg.num_features,
        observable_wire=cfg.observable_wire,
        seed=cfg.seed,
        init_scale=cfg.init_scale,
    )
    train_cfg = TrainingConfig(
        seed=cfg.seed,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        grad_clip=cfg.grad_clip,
        log_every=cfg.log_every,
        output_dir=cfg.output_dir,
    )

    trainer = Trainer(model, train_cfg)
    history = trainer.fit(split.X_train, split.y_train, split.X_test, split.y_test)
    output_dir = trainer.save_artifacts(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        raw_test=split.raw_test,
        experiment_config=cfg.to_dict(),
    )

    if not args.no_plots:
        from qnn.visualization import plot_decision_boundary, plot_training_history

        plot_training_history(history, output_dir)
        plot_decision_boundary(model, raw, y, output_dir)

    print("Training complete")
    print(f"Output directory: {output_dir}")
    print(json.dumps(history[-1].__dict__, indent=2))
    return 0


def _load_saved_model_metadata(config_path: Path) -> dict[str, Any]:
    data = load_json(config_path)
    model = data.get("model")
    if isinstance(model, dict):
        return model

    # Also accept a flat experiment config for convenience, but never ignore
    # malformed or unrelated metadata silently.
    try:
        cfg = ExperimentConfig.from_mapping(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "prediction config must be a saved artifact config or a valid flat experiment config"
        ) from exc
    return {
        "num_qubits": cfg.num_qubits,
        "num_layers": cfg.num_layers,
        "num_features": cfg.num_features,
        "observable_wire": cfg.observable_wire,
    }


def _metadata_integer(metadata: dict[str, Any], name: str, default: int) -> int:
    value = metadata.get(name, default)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"saved model metadata field {name!r} must be an integer")
    return int(value)


def predict_command(args: argparse.Namespace) -> int:
    if not np.isfinite(args.x0) or not np.isfinite(args.x1):
        raise ValueError("x0 and x1 must be finite")

    params_path = Path(args.params)
    loaded = np.load(params_path, allow_pickle=False)
    if np.iscomplexobj(loaded):
        raise TypeError("parameter tensor must be real-valued")
    params = np.asarray(loaded, dtype=np.float64)
    if params.ndim != 3 or params.shape[2] != 3:
        raise ValueError(
            "parameter tensor must have shape (num_layers, num_qubits, 3); "
            f"got {params.shape}"
        )
    inferred_layers, inferred_qubits, _ = params.shape
    if inferred_layers < 1 or inferred_qubits < 1:
        raise ValueError("parameter tensor must contain at least one layer and one qubit")

    config_path: Path | None = Path(args.config) if args.config is not None else None
    if config_path is None:
        sibling = params_path.parent / "config.json"
        if sibling.exists():
            config_path = sibling
    metadata = _load_saved_model_metadata(config_path) if config_path is not None else {}

    num_layers = _metadata_integer(metadata, "num_layers", inferred_layers)
    num_qubits = _metadata_integer(metadata, "num_qubits", inferred_qubits)
    num_features = _metadata_integer(metadata, "num_features", 2)
    observable_wire = metadata.get("observable_wire", None)
    saved_parameter_count = metadata.get("parameter_count")
    if saved_parameter_count is not None:
        if isinstance(saved_parameter_count, (bool, np.bool_)) or not isinstance(
            saved_parameter_count, (int, np.integer)
        ):
            raise TypeError("saved model metadata field 'parameter_count' must be an integer")
        if int(saved_parameter_count) != params.size:
            raise ValueError("saved model parameter_count does not match the parameter tensor")

    if num_features != 2:
        raise ValueError(
            "the qnn predict CLI accepts exactly two raw features (x0, x1); "
            f"saved model metadata requires {num_features}"
        )

    if args.num_layers is not None and args.num_layers != inferred_layers:
        raise ValueError(
            f"--num-layers={args.num_layers} does not match parameter tensor ({inferred_layers})"
        )
    if args.num_qubits is not None and args.num_qubits != inferred_qubits:
        raise ValueError(
            f"--num-qubits={args.num_qubits} does not match parameter tensor ({inferred_qubits})"
        )
    if num_layers != inferred_layers or num_qubits != inferred_qubits:
        raise ValueError("saved model metadata does not match the parameter tensor shape")
    if args.observable_wire is not None:
        if observable_wire is not None and args.observable_wire != observable_wire:
            raise ValueError(
                f"--observable-wire={args.observable_wire} does not match saved model metadata "
                f"({observable_wire})"
            )
        observable_wire = args.observable_wire

    model = DataReuploadingQNN(
        num_qubits=num_qubits,
        num_layers=num_layers,
        num_features=num_features,
        observable_wire=observable_wire,
    )
    model.set_parameters(params)

    x = np.array([[args.x0, args.x1]], dtype=np.float64) * np.pi
    proba = float(model.predict_proba(x)[0])
    pred = int(proba >= 0.5)
    print(json.dumps({"p_class_1": proba, "prediction": pred}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return train_command(args)
        if args.command == "predict":
            return predict_command(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
