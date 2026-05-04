"""Run the default QNN training experiment programmatically."""

from __future__ import annotations

from qnn.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", "--config", "configs/default.json"]))
