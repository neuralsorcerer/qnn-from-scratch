#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e .
qnn train --config configs/default.json
python -m pytest -q
