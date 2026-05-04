.PHONY: install test demo clean

install:
	python -m pip install -e .[dev]

test:
	python -m pytest -q

demo:
	qnn train --config configs/default.json

clean:
	rm -rf outputs .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
