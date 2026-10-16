"""Chunk the raw vLLM corpus into MinimalSource pieces."""
from pathlib import Path
from typing import List, cast

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from .models import MinimalSource

RAW_DIR = "data/raw/vllm-0.10.1"


def _chunks_from_file(doc: Path,
                      splitter: RecursiveCharacterTextSplitter
                      ) -> List[MinimalSource]:
    """Split one file into MinimalSource chunks with character offsets."""
    try:
        text = doc.read_text()
    except UnicodeDecodeError:
        return []

    results: List[MinimalSource] = []
    for chunk in splitter.create_documents([text]):
        start = int(chunk.metadata["start_index"])
        results.append(
            MinimalSource(
                file_path=str(doc),
                first_character_index=start,
                last_character_index=len(chunk.page_content) + start,
                content=chunk.page_content,
            )
        )
    return results


def create_chunks_for_text(chunk_size: int) -> List[MinimalSource]:
    """Chunk every .txt/.md file of the corpus."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=0,
        chunk_size=chunk_size,
        add_start_index=True,
        strip_whitespace=False,
    )
    valid = {".txt", ".md"}
    return [
        chunk
        for doc in Path(RAW_DIR).rglob("*")
        if doc.is_file() and doc.suffix in valid
        for chunk in _chunks_from_file(doc, splitter)
    ]


def create_chunks_for_code(chunk_size: int) -> List[MinimalSource]:
    """Chunk every .py file of the corpus.

    The splitter is built with ``Language.PYTHON`` as the first positional
    argument to reproduce the reference chunking exactly; the ground-truth
    character offsets depend on it.
    """
    results: List[MinimalSource] = []
    for doc in Path(RAW_DIR).rglob("*"):
        if doc.is_file() and doc.suffix == ".py":
            splitter = RecursiveCharacterTextSplitter(
                cast(List[str], Language.PYTHON),
                chunk_overlap=0,
                chunk_size=chunk_size,
                add_start_index=True,
                strip_whitespace=False,
            )
            results.extend(_chunks_from_file(doc, splitter))
    return results
