"""Elementary one-qubit gates used by the local statevector simulator."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qnn._validation import real_scalar

ComplexArray = npt.NDArray[np.complex128]

I2: ComplexArray = np.eye(2, dtype=np.complex128)
X: ComplexArray = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y: ComplexArray = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z: ComplexArray = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H: ComplexArray = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

# Prevent accidental mutation of module-level gate constants.
for _gate in (I2, X, Y, Z, H):
    _gate.setflags(write=False)


def _as_finite_angle(theta: float) -> float:
    """Validate one finite real scalar angle."""
    return real_scalar("theta", theta)


def rx(theta: float) -> ComplexArray:
    """Return :math:`R_X(theta) = exp(-i theta X / 2)`."""
    theta = _as_finite_angle(theta)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(theta: float) -> ComplexArray:
    """Return :math:`R_Y(theta) = exp(-i theta Y / 2)`."""
    theta = _as_finite_angle(theta)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> ComplexArray:
    """Return :math:`R_Z(theta) = exp(-i theta Z / 2)`."""
    theta = _as_finite_angle(theta)
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )
