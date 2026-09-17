"""Data-reuploading Quantum Neural Network implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnn._validation import integer_scalar, real_array, real_scalar
from qnn.gates import rx, ry, rz
from qnn.metrics import binary_cross_entropy_derivative
from qnn.statevector import StateVectorSimulator


@dataclass
class DataReuploadingQNN:
    """Small trainable QNN using angle encoding and a project-specific CNOT ansatz.

    Each layer uploads two cyclic feature angles per wire, applies trainable
    ``RY-RZ`` rotations, applies the directed CNOT neighbor pattern, and then
    applies a final trainable ``RY`` rotation per wire.

    The current feature mapping guarantees that every input feature is used
    only when ``num_features <= num_qubits + 1``. Larger feature vectors are
    rejected rather than silently ignored.
    """

    num_qubits: int = 2
    num_layers: int = 3
    num_features: int = 2
    observable_wire: int | None = None
    seed: int = 7
    init_scale: float = 0.25

    def __post_init__(self) -> None:
        self.num_qubits = integer_scalar("num_qubits", self.num_qubits, minimum=1)
        self.num_layers = integer_scalar("num_layers", self.num_layers, minimum=1)
        self.num_features = integer_scalar("num_features", self.num_features, minimum=1)
        if self.num_features > self.num_qubits + 1:
            raise ValueError(
                "num_features must be <= num_qubits + 1 for the current cyclic encoder; "
                "otherwise some features would never be uploaded"
            )

        self.seed = integer_scalar("seed", self.seed, minimum=0)

        self.init_scale = real_scalar("init_scale", self.init_scale)
        if self.init_scale < 0.0:
            raise ValueError("init_scale must be a finite non-negative value")

        if self.observable_wire is None:
            self.observable_wire = min(1, self.num_qubits - 1)
        else:
            self.observable_wire = integer_scalar(
                "observable_wire", self.observable_wire, minimum=0
            )
        if not 0 <= self._measured_wire() < self.num_qubits:
            raise ValueError("observable_wire must be a valid qubit index")

        self.sim = StateVectorSimulator(self.num_qubits)
        rng = np.random.default_rng(self.seed)
        self.params = rng.normal(
            loc=0.0,
            scale=self.init_scale,
            size=(self.num_layers, self.num_qubits, 3),
        ).astype(np.float64)

    @property
    def parameter_count(self) -> int:
        return int(self.params.size)

    def architecture(self) -> dict[str, int | float]:
        return {
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "num_features": self.num_features,
            "observable_wire": self._measured_wire(),
            "init_scale": self.init_scale,
            "parameter_count": self.parameter_count,
        }

    def copy(self) -> "DataReuploadingQNN":
        other = DataReuploadingQNN(
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            num_features=self.num_features,
            observable_wire=self.observable_wire,
            seed=self.seed,
            init_scale=self.init_scale,
        )
        other.params = self.params.copy()
        return other

    def set_parameters(self, params: np.ndarray) -> None:
        """Validate and copy a complete parameter tensor into the model."""
        candidate = real_array("params", params)
        self._validate_params(candidate)
        self.params = candidate.copy()

    def forward_state(self, x: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Return the final statevector for one feature vector."""
        if params is None:
            params = self.params
        params = real_array("params", params)
        self._validate_params(params)
        x = self._validate_input(x)

        state = self.sim.zero_state()
        for layer in range(self.num_layers):
            for wire in range(self.num_qubits):
                first_feature = x[wire % self.num_features]
                second_feature = x[(wire + 1) % self.num_features]
                state = self.sim.apply_one_qubit_gate(state, rx(first_feature), wire)
                state = self.sim.apply_one_qubit_gate(state, ry(second_feature), wire)

            for wire in range(self.num_qubits):
                state = self.sim.apply_one_qubit_gate(state, ry(params[layer, wire, 0]), wire)
                state = self.sim.apply_one_qubit_gate(state, rz(params[layer, wire, 1]), wire)

            state = self.sim.apply_ring_entanglement(state)

            for wire in range(self.num_qubits):
                state = self.sim.apply_one_qubit_gate(state, ry(params[layer, wire, 2]), wire)

        # All applied matrices are checked unitary, so normalization is preserved.
        self.sim.probabilities(state)
        return state

    def expectation(self, x: np.ndarray, params: np.ndarray | None = None) -> float:
        """Return ``<Z>`` on the configured observable wire."""
        state = self.forward_state(x, params=params)
        return self.sim.expectation_z(state, self._measured_wire())

    def expectations(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Return ``<Z>`` for each row of ``X``."""
        X = self._validate_batch(X)
        return np.asarray([self.expectation(row, params=params) for row in X], dtype=np.float64)

    def predict_proba(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Return the Born probability of measuring 1 on the observable wire."""
        z = self.expectations(X, params=params)
        p = (1.0 - z) / 2.0
        # Clip only roundoff-sized excursions from the physical interval.
        return np.asarray(np.clip(p, 0.0, 1.0), dtype=np.float64)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        threshold = real_scalar("threshold", threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite value in [0, 1]")
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def parameter_shift_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the mean BCE gradient using exact two-term gate shifts.

        Every trainable scalar controls exactly one ``RY`` or ``RZ`` gate with
        generator ``P/2`` for a Pauli operator ``P``. Therefore

        ``d<Z>/dtheta = 0.5 * (<Z>_{theta+pi/2} - <Z>_{theta-pi/2})``.

        The nonlinear BCE objective is differentiated with the ordinary chain
        rule. Parameter-shift is applied to the quantum expectation, not to BCE
        itself.
        """
        X = self._validate_batch(X)
        self._validate_params(self.params)
        y = real_array("y", y).reshape(-1)
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of rows")

        base_proba = self.predict_proba(X)
        dloss_dp = binary_cross_entropy_derivative(base_proba, y)
        gradient = np.zeros_like(self.params)
        shift = np.pi / 2.0

        for index in np.ndindex(self.params.shape):
            plus_params = self.params.copy()
            minus_params = self.params.copy()
            plus_params[index] += shift
            minus_params[index] -= shift

            z_plus = self.expectations(X, params=plus_params)
            z_minus = self.expectations(X, params=minus_params)
            dz_dtheta = 0.5 * (z_plus - z_minus)
            dp_dtheta = -0.5 * dz_dtheta
            gradient[index] = np.sum(dloss_dp * dp_dtheta)

        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("non-finite parameter gradient encountered")
        return gradient

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        x = real_array("input features", x)
        if x.shape != (self.num_features,):
            raise ValueError(f"Expected input shape {(self.num_features,)}, got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("input features must contain only finite values")
        return x

    def _validate_batch(self, X: np.ndarray) -> np.ndarray:
        X = real_array("X", X)
        if X.ndim != 2 or X.shape[1] != self.num_features:
            raise ValueError(f"Expected X shape (n_samples, {self.num_features}), got {X.shape}")
        if len(X) == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _validate_params(self, params: np.ndarray) -> None:
        expected = (self.num_layers, self.num_qubits, 3)
        if params.shape != expected:
            raise ValueError(f"Expected params shape {expected}, got {params.shape}")
        if not np.all(np.isfinite(params)):
            raise ValueError("params must contain only finite values")

    def _measured_wire(self) -> int:
        wire = self.observable_wire
        if wire is None:
            raise RuntimeError("observable_wire has not been initialized")
        return wire
