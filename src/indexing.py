import bm25s
import numpy as np
from .models import MinimalSource


def index_chunks(chunks: list[MinimalSource]) -> None:
    texts = []
    
    for chunk in chunks:
        texts.append(chunk.content)

    vectorizer = bm25s.tokenize(texts, leave=True)
    bm25 = bm25s.BM25(method="bm25l", k1=1, b=0.5)
    bm25.index(vectorizer, leave_progress=True)
    bm25.save("data/processed/indexs", leave_progress=True)
