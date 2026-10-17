"""Build and persist the BM25 index over the chunks."""
import bm25s

from .models import MinimalSource

INDEX_DIR = "data/processed/indexs"


def index_chunks(chunks: list[MinimalSource]) -> None:
    """Tokenize the chunks and build/save a bm25l index.

    Args:
        chunks: Chunks to index.

    Raises:
        OSError: If the index cannot be persisted under ``INDEX_DIR``.
    """
    texts = [chunk.content for chunk in chunks]
    tokens = bm25s.tokenize(texts, leave=True)
    bm25 = bm25s.BM25(method="bm25l", k1=1, b=0.5)
    bm25.index(tokens, leave_progress=True)
    bm25.save(INDEX_DIR, leave_progress=True)
