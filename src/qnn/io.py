"""I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load strict UTF-8 JSON and require a top-level object.

    Duplicate object keys and non-standard ``NaN``/``Infinity`` constants are
    rejected instead of being silently accepted by Python's permissive JSON
    decoder.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(
            f, object_pairs_hook=_strict_object, parse_constant=_reject_nonfinite_constant
        )
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object")
    return data
