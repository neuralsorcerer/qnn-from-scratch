.PHONY: install test lint demo clean

install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest -q

lint:
	black --check .
	isort --check-only .
	mypy

demo:
	qnn train --config configs/default.json

clean:
	rm -rf outputs .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
