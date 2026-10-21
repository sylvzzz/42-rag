FOLDER := $(shell pwd)
VENV = $(FOLDER)/.venv
REQ = requirements.txt
PY = $(VENV)/bin/python
MOULINETTE := $(shell if [ -x ./moulinette ]; then echo ./moulinette; elif [ -x ./moulinette-ubuntu ]; then echo ./moulinette-ubuntu; else echo ./moulinette-fedora; fi)

install:
	uv sync

run:
	uv run $(PY) -m src

index:
	uv run python3 -m src index --max_chunk_size 2000

search:
	uv run python3 -m src search --query "Whats is this codebase about?" --k 1

search_dataset_docs:
	uv run $(PY) -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

search_dataset_code:
	uv run $(PY) -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

answer:
	uv run $(PY) -m src answer --query "What is this codebase about?" --k 3

answer_dataset_docs:
	uv run $(PY) -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions

answer_dataset_code:
	uv run $(PY) -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions

evaluate_code:
	uv run $(PY) -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json

evaluate_docs:
	uv run $(PY) -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

moulinette_docs:
	$(MOULINETTE) evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000

moulinette_code:
	$(MOULINETTE) evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000

api:
	uv run uvicorn api:app --reload

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".cache" -exec rm -rf {} +

lint:
	python3 -m flake8 .
	python3 -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


.PHONY: install run debug clean lint search answer search_dataset_docs search_dataset_code answer_dataset_docs answer_dataset_code evaluate_docs evaluate_code moulinette_docs moulinette_code