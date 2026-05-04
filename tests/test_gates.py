import numpy as np

from qnn.gates import rx, ry, rz
from qnn.statevector import StateVectorSimulator


def test_rotation_gates_are_unitary():
    """Verify elementary rotation gates are unitary matrices."""
    for gate_fn in [rx, ry, rz]:
        gate = gate_fn(0.73)
        np.testing.assert_allclose(gate.conj().T @ gate, np.eye(2), atol=1e-12)


def test_cnot_maps_basis_state_10_to_11():
    """Verify CNOT flips the target when control qubit is set."""
    sim = StateVectorSimulator(num_qubits=2)
    state = np.array([0, 0, 1, 0], dtype=np.complex128)  # |10>
    updated = sim.apply_cnot(state, control=0, target=1)
    expected = np.array([0, 0, 0, 1], dtype=np.complex128)  # |11>
    np.testing.assert_allclose(updated, expected)


def test_expectation_z_on_zero_and_one():
    """Verify Z expectation values for |0> and |1> are +1 and -1."""
    sim = StateVectorSimulator(num_qubits=1)
    zero = np.array([1, 0], dtype=np.complex128)
    one = np.array([0, 1], dtype=np.complex128)
    assert sim.expectation_z(zero, 0) == 1.0
    assert sim.expectation_z(one, 0) == -1.0
