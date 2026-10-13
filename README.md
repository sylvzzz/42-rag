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