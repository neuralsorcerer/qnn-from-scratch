"""Validated experiment configuration used by the CLI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from qnn._validation import integer_scalar, real_scalar

_ALLOWED_DATASETS = {"linear", "vertical", "horizontal", "circle", "sine"}


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete built-in two-feature experiment configuration."""

    seed: int = 7
    num_samples: int = 56
    test_size: float = 0.25
    dataset: str = "vertical"
    noise: float = 0.08
    num_qubits: int = 2
    num_layers: int = 2
    num_features: int = 2
    observable_wire: int | None = 1
    init_scale: float = 0.25
    epochs: int = 35
    learning_rate: float = 0.12
    batch_size: int = 0
    grad_clip: float = 2.0
    output_dir: str = "outputs/default_run"
    log_every: int = 5

    def __post_init__(self) -> None:
        seed = integer_scalar("seed", self.seed, minimum=0)
        num_samples = integer_scalar("num_samples", self.num_samples, minimum=8)
        num_qubits = integer_scalar("num_qubits", self.num_qubits, minimum=1)
        num_layers = integer_scalar("num_layers", self.num_layers, minimum=1)
        num_features = integer_scalar("num_features", self.num_features, minimum=1)
        epochs = integer_scalar("epochs", self.epochs, minimum=1)
        batch_size = integer_scalar("batch_size", self.batch_size, minimum=0)
        log_every = integer_scalar("log_every", self.log_every, minimum=1)

        if num_features != 2:
            raise ValueError("the built-in CLI dataset currently has exactly two features")
        if num_features > num_qubits + 1:
            raise ValueError("num_features exceeds the model's current encoding coverage")

        test_size = real_scalar("test_size", self.test_size)
        noise = real_scalar("noise", self.noise)
        init_scale = real_scalar("init_scale", self.init_scale)
        learning_rate = real_scalar("learning_rate", self.learning_rate)
        grad_clip = real_scalar("grad_clip", self.grad_clip)

        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must lie in (0, 1)")
        if noise < 0.0:
            raise ValueError("noise must be non-negative")
        if init_scale < 0.0:
            raise ValueError("init_scale must be non-negative")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if grad_clip < 0.0:
            raise ValueError("grad_clip must be non-negative; 0 disables clipping")

        if not isinstance(self.dataset, str) or self.dataset not in _ALLOWED_DATASETS:
            raise ValueError(f"dataset must be one of: {', '.join(sorted(_ALLOWED_DATASETS))}")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")

        observable_wire = self.observable_wire
        if observable_wire is not None:
            observable_wire = integer_scalar("observable_wire", observable_wire, minimum=0)
            if observable_wire >= num_qubits:
                raise ValueError("observable_wire must be smaller than num_qubits")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "num_samples", num_samples)
        object.__setattr__(self, "test_size", test_size)
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "num_layers", num_layers)
        object.__setattr__(self, "num_features", num_features)
        object.__setattr__(self, "observable_wire", observable_wire)
        object.__setattr__(self, "init_scale", init_scale)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "grad_clip", grad_clip)
        object.__setattr__(self, "log_every", log_every)
        object.__setattr__(self, "output_dir", self.output_dir.strip())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ExperimentConfig":
        """Build a config while rejecting unknown keys and silent integer truncation."""
        if not isinstance(mapping, Mapping):
            raise TypeError("mapping must implement collections.abc.Mapping")
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
        return cls(**dict(mapping))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
