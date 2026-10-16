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

search_dataset_docs:
	uv run $(PY) -m src search_dataset --dataset_path data/datasets/datasets_public/public/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/

search_dataset_code:
	uv run $(PY) -m src search_dataset --dataset_path data/datasets/datasets_public/public/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/

evaluate_code:
	uv run $(PY) -m src evaluate --student_search_results_path data/output/search_results/dataset_code_public.json --dataset_path data/datasets/datasets_public/public/AnsweredQuestions/dataset_code_public.json

evaluate_docs:
	uv run $(PY) -m src evaluate --student_search_results_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets/datasets_public/public/AnsweredQuestions/dataset_docs_public.json

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