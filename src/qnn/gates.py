"""Core one-qubit gates used by the local statevector simulator."""

from __future__ import annotations

import numpy as np

ComplexArray = np.ndarray

I2: ComplexArray = np.eye(2, dtype=np.complex128)
X: ComplexArray = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y: ComplexArray = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z: ComplexArray = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H: ComplexArray = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def rx(theta: float) -> ComplexArray:
    """Build the one-qubit :math:`R_X(\\theta)` rotation matrix.

    Args:
        theta: Rotation angle in radians.

    Returns:
        A ``(2, 2)`` complex unitary matrix implementing
        :math:`\\exp(-i \\theta X / 2)`.
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(theta: float) -> ComplexArray:
    """Build the one-qubit :math:`R_Y(\\theta)` rotation matrix.

    Args:
        theta: Rotation angle in radians.

    Returns:
        A ``(2, 2)`` complex unitary matrix implementing
        :math:`\\exp(-i \\theta Y / 2)`.
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> ComplexArray:
    """Build the one-qubit :math:`R_Z(\\theta)` rotation matrix.

    Args:
        theta: Rotation angle in radians.

    Returns:
        A ``(2, 2)`` complex unitary matrix implementing
        :math:`\\exp(-i \\theta Z / 2)`.
    """
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )
