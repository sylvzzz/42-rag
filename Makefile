FOLDER := $(shell pwd)
MAIN = main.py
VENV = $(FOLDER)/.venv
REQ = requirements.txt
PY = $(VENV)/bin/python

install:
	uv sync

run:
	uv run $(PY) -m src

index:
	uv run python3 -m src index --max_chunk_size 2000

search:
	uv run python3 -m src search --query "Whats the project of the codebase" --k 1

search_dataset:
	uv run $(PY) -m src search_dataset --dataset_path data/datasets/datasets_public/public/UnansweredQuestions/dataset_docs_public.json --k 2 --save_directory data/output/search_results/

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