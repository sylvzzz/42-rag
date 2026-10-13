from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, Language
from .models import MinimalSource


def create_chunks_for_text(chunk_size: int):
    path = Path("data/raw/vllm-0.10.1")
    valid_files = [".txt", ".md"]
    chunks_created = []

    for doc in path.rglob("*"):
        if doc.is_file() and doc.suffix in valid_files:
            documents = RecursiveCharacterTextSplitter(
                chunk_overlap=0,
                chunk_size=chunk_size,
                add_start_index=True,
                strip_whitespace=False
            )

            chunks = documents.create_documents([doc.read_text()])
            for chunk in chunks:
                start_index = chunk.metadata["start_index"]
                result = MinimalSource(file_path=str(doc),
                                       first_character_index=start_index,
                                       last_character_index=len(chunk.page_content) + start_index,
                                       content=chunk.page_content)
                chunks_created.append(result)

    return chunks_created


def create_chunks_for_code(chunk_size: int):
    path = Path("data/raw/vllm-0.10.1")
    chunks_created = []
    for doc in path.rglob("*"):
        if doc.is_file() and doc.suffix == ".py":
            documents = RecursiveCharacterTextSplitter(
                Language.PYTHON,
                chunk_overlap=0,
                chunk_size=chunk_size,
                add_start_index=True,
                strip_whitespace=False,
            )

            chunks = documents.create_documents([doc.read_text()])
            for chunk in chunks:
                start_index = chunk.metadata["start_index"]
                result = MinimalSource(file_path=str(doc),
                                       first_character_index=start_index,
                                       last_character_index=len(chunk.page_content) + start_index,
                                       content=chunk.page_content)
                chunks_created.append(result)

    return chunks_created
