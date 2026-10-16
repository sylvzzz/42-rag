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
| search_dataset | Gets the top k sources from a set of queries/prompts and saves it in ```data/datasets/datasets_public/public/AnsweredQuestions``` in a json file of same name as the input file | ```<path> <int> <dir>``` | ``` uv run python3 -m src search_dataset -–dataset_path <path> -–k <int> -–save_directory <dir>``` |
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

Rename the binary for your distro first (e.g. `mv moulinette-ubuntu moulinette`),
then run:

```bash
./moulinette evaluate_student_search_results \
  <student_search_results.json> <ground_truth_dataset.json> \
  --k 10 --max_context_length 2000 -t 0.8
```

- The **first** file is your `search_dataset` output; the **second** is the
  ground-truth dataset with the correct sources.
- `-t 0.8` (docs) or `-t 0.5` (code) makes it print `PASS`/`FAIL` against the
  threshold.

With the **public** datasets you need a small pre-step, because they ship in a
different shape and use unrelated question ids: build a ground-truth file that
uses **your** question ids, pairing the two datasets by order.

```bash
DS=docs    # or: DS=code
TH=0.8     # use 0.5 when DS=code
uv run python - "$DS" <<'PY'
import json, sys
ds = sys.argv[1]
base = "data/datasets/datasets_public/public"
st = json.load(open(f"data/output/search_results/dataset_{ds}_public.json"))["search_results"]
a  = json.load(open(f"{base}/AnsweredQuestions/dataset_{ds}_public.json"))["search_results"]
gt = {"rag_questions": [
        {"question_id": st[i]["question_id"], "question": st[i]["question"], "answer": "",
         "sources": [{"file_path": s["file_path"],
                      "first_character_index": s["first_character_index"],
                      "last_character_index": s["last_character_index"]}
                     for s in a[i]["retrieved_sources"]]}
        for i in range(len(st))]}
json.dump(gt, open(f"/tmp/{ds}_gt.json", "w"))
PY

./moulinette evaluate_student_search_results \
  data/output/search_results/dataset_${DS}_public.json \
  /tmp/${DS}_gt.json --k 10 --max_context_length 2000 -t ${TH}
```

During the defense no pre-step is needed: the official ground-truth already
matches the questions you were given.

