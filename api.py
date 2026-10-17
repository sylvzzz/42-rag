"""Local HTTP API exposing the RAG pipeline (bonus part)."""
from fastapi import FastAPI, HTTPException

from src.logger import FatalError
from src.rag import RAG

app = FastAPI()
rag = RAG()


@app.get("/status")
def status() -> dict[str, int]:
    """Health check.

    Returns:
        A fixed 200 status payload.
    """
    return {"status": 200}


@app.get("/")
def root() -> dict[str, str]:
    """Welcome message.

    Returns:
        A greeting payload.
    """
    return {"value": "Hello world from rag against the machine project!"}


@app.get("/index")
def index(max_chunk_size: int = 2000) -> dict[str, str]:
    """(Re)build the corpus index.

    Args:
        max_chunk_size: Maximum characters per chunk.

    Returns:
        A success payload once the index is rebuilt.

    Raises:
        HTTPException: If indexing fails.
    """
    try:
        rag.index(max_chunk_size)
    except (FatalError, OSError, ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail="Error encountered while indexing...",
        ) from e
    return {"status": "Successfully indexed!"}


@app.get("/answer")
def answer(prompt: str = "", k: int = 5) -> dict[str, str]:
    """Answer a question using the RAG pipeline.

    Args:
        prompt: Question to answer, must not be empty.
        k: Maximum number of sources to retrieve.

    Returns:
        A payload with the generated answer.

    Raises:
        HTTPException: If the input is invalid or answering fails.
    """
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be positive")
    try:
        text = rag.answer(query=prompt, k=k)
    except (FatalError, OSError, ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail="Error encountered while answering...",
        ) from e
    return {"answer": text}
