import numpy as np
import pytest

from qnn.gates import H, I2, X, Y, Z, rx, ry, rz


@pytest.mark.parametrize("gate_fn", [rx, ry, rz])
@pytest.mark.parametrize("theta", [0.0, 0.73, -2.1, np.pi, 11.5])
def test_rotation_gates_are_unitary(gate_fn, theta):
    gate = gate_fn(theta)
    np.testing.assert_allclose(gate.conj().T @ gate, np.eye(2), atol=1e-12)


def test_fixed_gates_are_unitary_and_immutable():
    for gate in (I2, X, Y, Z, H):
        np.testing.assert_allclose(gate.conj().T @ gate, np.eye(2), atol=1e-12)
        assert not gate.flags.writeable


def test_rotation_special_values():
    np.testing.assert_allclose(rx(0.0), I2)
    np.testing.assert_allclose(ry(0.0), I2)
    np.testing.assert_allclose(rz(0.0), I2)
    np.testing.assert_allclose(rx(2 * np.pi), -I2, atol=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rotations_reject_nonfinite_angles(bad):
    for gate_fn in (rx, ry, rz):
        with pytest.raises(ValueError):
            gate_fn(bad)


def test_rotation_gates_reject_boolean_and_complex_angles():
    import pytest

    for gate_fn in [rx, ry, rz]:
        with pytest.raises(TypeError):
            gate_fn(True)
        with pytest.raises(TypeError):
            gate_fn(0.2 + 0.1j)
