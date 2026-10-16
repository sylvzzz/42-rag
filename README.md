# RAG aggainst the machine

# Instructions

## Installation

#### Installing in a 42 School Computer
If you are running this in a 42 school computer, please run this in the `sgoinfre` directory, since in 42 we dont have a lot of space in home directory

```bash
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/sgoinfre/$(USER)/.local/bin sh

# run this when you're in the project directory

export HF_HOME="$PWD/.cache/huggingface"
export UV_CACHE_DIR="$PWD/.cache/uv"
export PIP_CACHE_DIR="$PWD/.cache/pip"

make
```

#### Installing in your personal computer
```bash
curl -LsSf https://astral.sh/uv/install.sh

make
```

## Commands

| Action | What it does | Arguments | Usage |
| --- | --- | --- | --- |
| index | Creates the chunks and saves the embeddings from the code base | ``` <max_chunk_size>``` | ```uv run python3 -m src index --max_chunk_size 2000``` |
| search | Gets the top k sources from a query/prompt | ``` <query> <k>``` | ``` uv run python3 -m src search --query "What is this code base about?" --k 5``` |
| search_dataset | Runs a search over a whole dataset and saves a `StudentSearchResults` JSON (same file name) under `--save_directory` | ```<path> <int> <dir>``` | ``` uv run python3 -m src search_dataset -–dataset_path <path> -–k <int> -–save_directory <dir>``` |
| evaluate | Measures how good the retrieval was, by comparing your search results against the correct answers | ```<student_search_results> <dataset>``` | ```uv run python3 -m src evaluate --student_search_results_path <path> --dataset_path <path>``` |

## What does `evaluate` do?

`evaluate` does not search anything. It is only a **scoreboard**: it takes the
results your `search_dataset` produced and compares them against the correct
answers (the *ground-truth* sources), then prints how many of them you found.

### How the score is computed

For every question:

1. Take the correct source(s) from the ground-truth dataset.
2. Look at your top-k results for that question.
3. A correct source counts as **found** when one of your results points to the
   **same file** and covers **roughly the same region** of it (the character
   ranges overlap enough — we do not need an exact match, only to land in the
   right area of the right file).
4. `recall@k = found correct sources / total correct sources` for that question.
5. The final number is the **average over all questions**, reported for
   `k = 1, 3, 5, 10`.

A result in a different file never counts, which is why the file path has to be
exact.

### What the results mean

`Recall@k` is the share of correct sources you managed to retrieve within your
top `k` results.

| Score | Meaning |
| --- | --- |
| `1.000` | every correct source was retrieved within the top-k |
| `0.500` | half of the correct sources were retrieved |
| `0.000` | none of the correct sources were retrieved |

`Recall@1` is the strictest (only the very first result counts); `Recall@10` is
the most forgiving. The project thresholds are **80% on docs** and **50% on
code** at `Recall@5`.

### Example

Say a question has one correct source, `lora.md`, lines 3833–5714:

- If your **first** result is exactly that region of `lora.md`, the source is
  found within the top-1, so `Recall@1` through `Recall@10` are all `1.0`.
- If that region only shows up as your **7th** result, `Recall@5` is `0.0` but
  `Recall@10` is `1.0`.

For a dataset, the score is the average of these per-question values.

> The official score used during the defense is computed by the provided
> `moulinette`. This `evaluate` command mirrors the same idea so you can iterate
> quickly on your own, without the moulinette.

## Running the moulinette

The moulinette is the official scorer (the provided binary).

**1. Prepare the binary** (once) — pick the one for your distro and rename it to
`moulinette`:

```bash
mv moulinette-ubuntu moulinette     # or: mv moulinette-fedora moulinette
```

**2. Score your results** — the **student results come first**, the
**ground-truth `AnsweredQuestions` dataset second**:

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000 -t 0.8
```

| Flag | Meaning |
| --- | --- |
| `--k` | max sources per question (default 10) |
| `--max_context_length` | max context characters per source (default 2000) |
| `-t` | if set, checks Recall@5 against this value and prints `PASS`/`FAIL` |

The threshold is `-t 0.8` on docs and `-t 0.5` on code. The Makefile wraps the
whole sequence (it also runs `search_dataset` first and auto-detects the binary,
so the rename is optional there):

```bash
make search_dataset_docs && make moulinette_docs   # docs
make search_dataset_code && make moulinette_code   # code
```

The datasets live under `data/datasets/` (`UnansweredQuestions/` for the
questions, `AnsweredQuestions/` for the ground truth) and every output is scoped
by dataset under `data/output/search_results/<DatasetScope>/`, as required by
the subject.


