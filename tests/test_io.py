import json
from pathlib import Path

import pytest

from qnn.io import load_json


def test_load_json_rejects_duplicate_keys(tmp_path: Path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"epochs": 1, "epochs": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_json(path)


def test_load_json_rejects_nonfinite_constants(tmp_path: Path):
    for token in ("NaN", "Infinity", "-Infinity"):
        path = tmp_path / f"bad-{token.replace('-', 'minus')}.json"
        path.write_text('{"value": ' + token + "}", encoding="utf-8")
        with pytest.raises(ValueError, match="Non-finite JSON number"):
            load_json(path)


def test_load_json_accepts_strict_object(tmp_path: Path):
    path = tmp_path / "ok.json"
    payload = {"epochs": 3, "nested": {"enabled": True}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_json(path) == payload
