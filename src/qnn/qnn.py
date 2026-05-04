"""Data-reuploading Quantum Neural Network implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnn.gates import rx, ry, rz
from qnn.metrics import binary_cross_entropy_derivative
from qnn.statevector import StateVectorSimulator


@dataclass
class DataReuploadingQNN:
    """Small trainable QNN using angle encoding and ring entanglement.

    Parameters
    ----------
    num_qubits:
        Number of simulated qubits.
    num_layers:
        Number of data-reuploading variational layers.
    num_features:
        Number of classical input features. Features are reused cyclically if
        ``num_qubits`` is larger than ``num_features``.
    observable_wire:
        Wire used for Pauli-Z expectation measurement. The binary probability
        is ``p(class=1) = (1 - <Z>) / 2``.
    seed:
        Random seed for reproducible parameter initialization.
    """

    num_qubits: int = 2
    num_layers: int = 3
    num_features: int = 2
    observable_wire: int = 1
    seed: int = 7
    init_scale: float = 0.25

    def __post_init__(self) -> None:
        """Validate settings and initialize simulator plus trainable parameters."""
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.num_features < 1:
            raise ValueError("num_features must be at least 1")
        if not 0 <= self.observable_wire < self.num_qubits:
            raise ValueError("observable_wire must be a valid qubit index")

        self.sim = StateVectorSimulator(self.num_qubits)
        rng = np.random.default_rng(self.seed)
        # 3 trainable angles per qubit per layer: RY, RZ, RY.
        self.params = rng.normal(
            loc=0.0,
            scale=self.init_scale,
            size=(self.num_layers, self.num_qubits, 3),
        ).astype(np.float64)

    def copy(self) -> "DataReuploadingQNN":
        """Return a deep-ish copy that shares no trainable parameter storage."""
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

    def forward_state(self, x: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Return final statevector for one input row."""
        if params is None:
            params = self.params
        self._validate_params(params)
        x = self._validate_input(x)

        state = self.sim.zero_state()

        for layer in range(self.num_layers):
            # Angle encoding / data re-uploading.
            for wire in range(self.num_qubits):
                first_feature = x[wire % self.num_features]
                second_feature = x[(wire + 1) % self.num_features]
                state = self.sim.apply_one_qubit_gate(state, rx(float(first_feature)), wire)
                state = self.sim.apply_one_qubit_gate(state, ry(float(second_feature)), wire)

            # Trainable variational block.
            for wire in range(self.num_qubits):
                state = self.sim.apply_one_qubit_gate(
                    state, ry(float(params[layer, wire, 0])), wire
                )
                state = self.sim.apply_one_qubit_gate(
                    state, rz(float(params[layer, wire, 1])), wire
                )

            # Entanglement layer.
            state = self.sim.apply_ring_entanglement(state)

            # Additional trainable rotations after entanglement.
            for wire in range(self.num_qubits):
                state = self.sim.apply_one_qubit_gate(
                    state, ry(float(params[layer, wire, 2])), wire
                )

        return state

    def expectation(self, x: np.ndarray, params: np.ndarray | None = None) -> float:
        """Return Pauli-Z expectation for one input row."""
        state = self.forward_state(x, params=params)
        return self.sim.expectation_z(state, self.observable_wire)

    def expectations(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Return Pauli-Z expectations for a batch."""
        X = self._validate_batch(X)
        return np.array([self.expectation(row, params=params) for row in X], dtype=np.float64)

    def predict_proba(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Return p(class=1) for each input row."""
        z = self.expectations(X, params=params)
        return np.clip((1.0 - z) / 2.0, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions."""
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def parameter_shift_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute BCE gradient with parameter-shift derivatives.

        For each trainable gate parameter theta_i, the model computes the
        derivative of the measured expectation value using
        0.5 * [f(theta_i + pi/2) - f(theta_i - pi/2)] and applies the chain
        rule through p = (1 - <Z>) / 2 and binary cross entropy.
        """
        X = self._validate_batch(X)
        y = y.astype(np.float64)
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

        return gradient

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        """Validate and return one feature vector with expected shape."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.num_features,):
            raise ValueError(f"Expected input shape {(self.num_features,)}, got {x.shape}")
        return x

    def _validate_batch(self, X: np.ndarray) -> np.ndarray:
        """Validate and return a 2-D feature batch array."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.num_features:
            raise ValueError(f"Expected X shape (n_samples, {self.num_features}), got {X.shape}")
        return X

    def _validate_params(self, params: np.ndarray) -> None:
        """Ensure parameter tensor has the model's canonical shape."""
        expected = (self.num_layers, self.num_qubits, 3)
        if params.shape != expected:
            raise ValueError(f"Expected params shape {expected}, got {params.shape}")
