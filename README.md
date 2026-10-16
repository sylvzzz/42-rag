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