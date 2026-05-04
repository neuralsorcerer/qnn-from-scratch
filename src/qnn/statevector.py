"""Small, readable statevector simulator.

Qubit convention
----------------
Qubit 0 is the most significant bit. For two qubits, the basis order is:
|00>, |01>, |10>, |11>.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnn.gates import ComplexArray, Z


@dataclass(frozen=True)
class StateVectorSimulator:
    """Minimal n-qubit statevector simulator.

    The class is intentionally immutable and stateless. Methods accept and
    return explicit state arrays so QNN forward passes are easy to test and
    reason about.
    """

    num_qubits: int

    def __post_init__(self) -> None:
        """Validate constructor arguments after dataclass initialization."""
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")

    @property
    def dimension(self) -> int:
        """Return the Hilbert-space dimension for ``num_qubits``."""
        return int(2**self.num_qubits)

    def zero_state(self) -> ComplexArray:
        """Return the computational basis ground state ``|00...0>``."""
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        return state

    def apply_one_qubit_gate(
        self, state: ComplexArray, gate: ComplexArray, wire: int
    ) -> ComplexArray:
        """Apply a 2x2 gate to one wire."""
        self._validate_wire(wire)
        if gate.shape != (2, 2):
            raise ValueError(f"Expected a 2x2 gate, got shape {gate.shape}")
        if state.shape != (self.dimension,):
            raise ValueError(f"Expected state shape {(self.dimension,)}, got {state.shape}")

        tensor = state.reshape([2] * self.num_qubits)
        tensor = np.moveaxis(tensor, wire, 0)
        original_shape = tensor.shape
        tensor = tensor.reshape(2, -1)
        updated = gate @ tensor
        updated = updated.reshape(original_shape)
        updated = np.moveaxis(updated, 0, wire)
        return np.asarray(updated.reshape(self.dimension), dtype=np.complex128)

    def apply_cnot(self, state: ComplexArray, control: int, target: int) -> ComplexArray:
        """Apply a controlled-X gate."""
        self._validate_wire(control)
        self._validate_wire(target)
        if control == target:
            raise ValueError("control and target must be different wires")
        if state.shape != (self.dimension,):
            raise ValueError(f"Expected state shape {(self.dimension,)}, got {state.shape}")

        updated = np.zeros_like(state)
        for basis_index, amplitude in enumerate(state):
            if amplitude == 0:
                continue
            bits = self._index_to_bits(basis_index)
            if bits[control] == 1:
                bits[target] ^= 1
            updated[self._bits_to_index(bits)] += amplitude
        return updated

    def apply_ring_entanglement(self, state: ComplexArray) -> ComplexArray:
        """Apply a nearest-neighbor CNOT chain with a closing ring when possible."""
        if self.num_qubits == 1:
            return state
        updated = state
        for control in range(self.num_qubits - 1):
            updated = self.apply_cnot(updated, control=control, target=control + 1)
        if self.num_qubits > 2:
            updated = self.apply_cnot(updated, control=self.num_qubits - 1, target=0)
        return updated

    def expectation_z(self, state: ComplexArray, wire: int) -> float:
        """Return <Z_wire>."""
        self._validate_wire(wire)
        z_state = self.apply_one_qubit_gate(state, Z, wire)
        return float(np.real(np.vdot(state, z_state)))

    def probabilities(self, state: ComplexArray) -> np.ndarray:
        """Return computational-basis probabilities."""
        probs = np.abs(state) ** 2
        normalized = probs / probs.sum()
        return np.asarray(normalized, dtype=np.float64)

    def _validate_wire(self, wire: int) -> None:
        """Raise ``ValueError`` if ``wire`` is outside the valid range."""
        if not 0 <= wire < self.num_qubits:
            raise ValueError(f"wire must be in [0, {self.num_qubits - 1}], got {wire}")

    def _index_to_bits(self, index: int) -> list[int]:
        """Convert a basis-state index to its big-endian bit representation."""
        return [(index >> (self.num_qubits - 1 - i)) & 1 for i in range(self.num_qubits)]

    def _bits_to_index(self, bits: list[int]) -> int:
        """Convert a big-endian list of bits back to a basis-state index."""
        index = 0
        for bit in bits:
            index = (index << 1) | int(bit)
        return index
