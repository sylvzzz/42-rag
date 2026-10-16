"""Command-line entry point (Python Fire) for the RAG pipeline."""
import sys

import fire

from .logger import Logger
from .rag import RAG

if __name__ == "__main__":
    try:
        fire.Fire(RAG)
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        Logger.error(f"Error: {e}")
        sys.exit(1)
