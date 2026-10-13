FOLDER := $(shell pwd)
MAIN = main.py
VENV = $(FOLDER)/.venv
REQ = requirements.txt
PY = $(VENV)/bin/python

install:
	uv sync

run:
	uv run $(PY) -m src

debug:
	python3 -m pdb $(MAIN)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name ".cache" -exec rm -rf {} +

lint:
	python3 -m flake8 . --exclude=.venv,llm_sdk
	python3 -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


.PHONY: install run debug clean lint