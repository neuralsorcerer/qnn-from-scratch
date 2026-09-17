import numpy as np
import pytest

from qnn.gates import H, X, ry
from qnn.statevector import StateVectorSimulator


def _explicit_one_qubit(gate, wire, n):
    factors = [gate if q == wire else np.eye(2) for q in range(n)]
    out = factors[0]
    for factor in factors[1:]:
        out = np.kron(out, factor)
    return out


def _random_state(n, seed=123):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return state / np.linalg.norm(state)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_zero_state(n):
    sim = StateVectorSimulator(n)
    state = sim.zero_state()
    assert state.shape == (2**n,)
    assert state[0] == 1.0
    np.testing.assert_allclose(np.linalg.norm(state), 1.0)


def test_big_endian_wire_order():
    sim = StateVectorSimulator(2)
    zero = sim.zero_state()
    assert np.argmax(np.abs(sim.apply_one_qubit_gate(zero, X, 0))) == 2  # |10>
    assert np.argmax(np.abs(sim.apply_one_qubit_gate(zero, X, 1))) == 1  # |01>


@pytest.mark.parametrize("wire", [0, 1, 2])
def test_one_qubit_application_matches_kronecker_reference(wire):
    sim = StateVectorSimulator(3)
    state = _random_state(3, seed=wire + 4)
    gate = ry(0.37 - 0.1 * wire)
    actual = sim.apply_one_qubit_gate(state, gate, wire)
    expected = _explicit_one_qubit(gate, wire, 3) @ state
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("control,target", [(0, 1), (1, 0), (0, 2), (2, 1)])
def test_cnot_matches_basis_reference(control, target):
    sim = StateVectorSimulator(3)
    for basis in range(8):
        state = np.zeros(8, dtype=np.complex128)
        state[basis] = 1.0
        bits = [(basis >> (2 - i)) & 1 for i in range(3)]
        expected_bits = bits.copy()
        if expected_bits[control]:
            expected_bits[target] ^= 1
        expected_index = 0
        for bit in expected_bits:
            expected_index = (expected_index << 1) | bit
        actual = sim.apply_cnot(state, control, target)
        assert np.argmax(np.abs(actual)) == expected_index


def test_ring_pattern_three_qubits():
    sim = StateVectorSimulator(3)
    state = np.zeros(8, dtype=np.complex128)
    state[4] = 1.0  # |100>
    # 0->1: 110, 1->2: 111, 2->0: 011
    actual = sim.apply_ring_entanglement(state)
    assert np.argmax(np.abs(actual)) == 3


def test_expectation_and_probability_one_agree():
    sim = StateVectorSimulator(2)
    state = sim.apply_one_qubit_gate(sim.zero_state(), H, 0)
    for wire in (0, 1):
        p1 = sim.probability_one(state, wire)
        z = sim.expectation_z(state, wire)
        np.testing.assert_allclose(p1, (1.0 - z) / 2.0, atol=1e-12)


def test_probabilities_do_not_silently_renormalize():
    sim = StateVectorSimulator(1)
    with pytest.raises(ValueError, match="normalized"):
        sim.probabilities(np.array([2.0, 0.0], dtype=np.complex128))


def test_nonunitary_gate_is_rejected():
    sim = StateVectorSimulator(1)
    with pytest.raises(ValueError, match="unitary"):
        sim.apply_one_qubit_gate(sim.zero_state(), np.array([[1, 0], [0, 2]]), 0)


@pytest.mark.parametrize("wire", [-1, 2])
def test_invalid_wire_is_rejected(wire):
    sim = StateVectorSimulator(2)
    with pytest.raises(ValueError):
        sim.expectation_z(sim.zero_state(), wire)


def test_num_qubits_numpy_integer_is_normalized_to_builtin_int():
    sim = StateVectorSimulator(np.int64(3))
    assert type(sim.num_qubits) is int
    assert sim.num_qubits == 3
