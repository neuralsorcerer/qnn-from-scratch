"""Shared strict scalar/array validation helpers.

The public APIs in this project intentionally reject booleans, strings, object
arrays, and complex values when a real numeric value is required. Silent
coercion makes malformed experiment data difficult to detect and can discard
information (for example, converting complex values to real floats).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def integer_scalar(name: str, value: Any, *, minimum: int | None = None) -> int:
    """Return a built-in ``int`` after strict integer validation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def real_scalar(
    name: str,
    value: Any,
    *,
    finite: bool = True,
) -> float:
    """Return a built-in ``float`` without accepting strings or booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if finite and not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def real_numeric_array(name: str, value: Any) -> np.ndarray:
    """Return a NumPy array after strict real numeric dtype validation."""
    raw = np.asarray(value)
    if raw.dtype.kind == "c":
        raise TypeError(f"{name} must be real-valued")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    return raw


def real_array(name: str, value: Any) -> FloatArray:
    """Convert genuine real numeric array-like input to ``float64``.

    Integer, unsigned-integer, and floating dtypes are accepted. Boolean,
    string, bytes, object, datetime, and complex dtypes are rejected.
    """
    return np.asarray(real_numeric_array(name, value), dtype=np.float64)


def numeric_array(name: str, value: Any) -> ComplexArray:
    """Convert genuine real/complex numeric array-like input to ``complex128``."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iufc":
        raise TypeError(f"{name} must contain numeric values")
    return np.asarray(raw, dtype=np.complex128)
