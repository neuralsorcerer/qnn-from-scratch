"""Small, explicit statevector simulator.

Qubit convention
----------------
Qubit 0 is the most-significant bit. For two qubits the basis order is
``|00>, |01>, |10>, |11>``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnn._validation import integer_scalar, numeric_array
from qnn.gates import ComplexArray, Z


@dataclass(frozen=True)
class StateVectorSimulator:
    """Minimal n-qubit statevector simulator with explicit state arrays."""

    num_qubits: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "num_qubits", integer_scalar("num_qubits", self.num_qubits, minimum=1)
        )

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension ``2**num_qubits``."""
        return 1 << int(self.num_qubits)

    def zero_state(self) -> ComplexArray:
        """Return ``|00...0>``."""
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        return state

    def apply_one_qubit_gate(
        self,
        state: ComplexArray,
        gate: ComplexArray,
        wire: int,
    ) -> ComplexArray:
        """Apply a finite unitary 2x2 matrix to one wire."""
        self._validate_wire(wire)
        state = self._validate_state(state)
        gate = numeric_array("gate", gate)
        if gate.shape != (2, 2):
            raise ValueError(f"Expected a 2x2 gate, got shape {gate.shape}")
        if not np.all(np.isfinite(gate)):
            raise ValueError("gate must contain only finite values")
        if not np.allclose(gate.conj().T @ gate, np.eye(2), atol=1e-12, rtol=1e-12):
            raise ValueError("gate must be unitary")

        tensor = state.reshape([2] * self.num_qubits)
        tensor = np.moveaxis(tensor, wire, 0)
        original_shape = tensor.shape
        tensor = tensor.reshape(2, -1)
        updated = gate @ tensor
        updated = updated.reshape(original_shape)
        updated = np.moveaxis(updated, 0, wire)
        return np.asarray(updated.reshape(self.dimension), dtype=np.complex128)

    def apply_cnot(self, state: ComplexArray, control: int, target: int) -> ComplexArray:
        """Apply CNOT with the supplied control and target wires."""
        self._validate_wire(control)
        self._validate_wire(target)
        if control == target:
            raise ValueError("control and target must be different wires")
        state = self._validate_state(state)

        # In the tensor view, axis ``q`` is qubit ``q`` because wire 0 is
        # the most-significant bit. Move control/target to the first two
        # axes, swap target slices only inside the control=1 sector, then
        # restore the original axis order. This is O(2**n) and avoids the
        # O(n) bit-list conversion previously done for every basis state.
        tensor = state.reshape([2] * self.num_qubits)
        moved = np.moveaxis(tensor, (int(control), int(target)), (0, 1))
        updated = moved.copy()
        updated[1, 0, ...] = moved[1, 1, ...]
        updated[1, 1, ...] = moved[1, 0, ...]
        restored = np.moveaxis(updated, (0, 1), (int(control), int(target)))
        return np.asarray(restored.reshape(self.dimension), dtype=np.complex128)

    def apply_ring_entanglement(self, state: ComplexArray) -> ComplexArray:
        """Apply the project's directed CNOT neighbor pattern.

        For two qubits the only neighbor pair is ``0 -> 1``. For three or
        more qubits, the chain ``0 -> 1 -> ... -> n-1`` is closed with
        ``n-1 -> 0``.
        """
        state = self._validate_state(state)
        if self.num_qubits == 1:
            return state.copy()

        updated = state
        for control in range(self.num_qubits - 1):
            updated = self.apply_cnot(updated, control=control, target=control + 1)
        if self.num_qubits > 2:
            updated = self.apply_cnot(updated, control=self.num_qubits - 1, target=0)
        return updated

    def expectation_z(self, state: ComplexArray, wire: int) -> float:
        """Return ``<Z_wire>`` for a normalized state."""
        self._validate_wire(wire)
        state = self._validate_state(state, require_normalized=True)
        z_state = self.apply_one_qubit_gate(state, Z, wire)
        value = float(np.real(np.vdot(state, z_state)))
        if value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
            raise FloatingPointError(f"invalid Pauli-Z expectation {value}")
        return float(np.clip(value, -1.0, 1.0))

    def probabilities(self, state: ComplexArray) -> np.ndarray:
        """Return computational-basis Born probabilities."""
        state = self._validate_state(state, require_normalized=True)
        return np.asarray(np.abs(state) ** 2, dtype=np.float64)

    def probability_one(self, state: ComplexArray, wire: int) -> float:
        """Return the Born probability of measuring ``1`` on ``wire``."""
        self._validate_wire(wire)
        probs = self.probabilities(state)
        total = 0.0
        for basis_index, probability in enumerate(probs):
            bit = (basis_index >> (self.num_qubits - 1 - wire)) & 1
            if bit == 1:
                total += float(probability)
        return total

    def _validate_state(
        self,
        state: ComplexArray,
        *,
        require_normalized: bool = False,
    ) -> ComplexArray:
        state = numeric_array("state", state)
        if state.shape != (self.dimension,):
            raise ValueError(f"Expected state shape {(self.dimension,)}, got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("state must contain only finite amplitudes")
        if require_normalized:
            norm_sq = float(np.real(np.vdot(state, state)))
            if not np.isclose(norm_sq, 1.0, atol=1e-10, rtol=1e-10):
                raise ValueError(f"state must be normalized, got squared norm {norm_sq}")
        return state

    def _validate_wire(self, wire: int) -> None:
        wire = integer_scalar("wire", wire, minimum=0)
        if not 0 <= wire < self.num_qubits:
            raise ValueError(f"wire must be in [0, {self.num_qubits - 1}], got {wire}")
