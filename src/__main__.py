from .chunking import create_chunks_for_code, create_chunks_for_text
from .indexing import index_chunks
from .models import MinimalSearchResults, MinimalSource
import fire
import json
from pathlib import Path
import bm25s
import uuid


INDEX_DIR = "data/processed/"
CHUNKS_FILE = INDEX_DIR + "index.json"
INDEX_DIR = "data/processed/indexs"

class RAG:
    def __init__(self) -> None:
        self.bm25 = bm25s.BM25(method="bm25l", k1=1, b=0.5)

    def index(self, max_chunk_size: int) -> None:
        text_chunks = create_chunks_for_text(max_chunk_size)
        code_chunks = create_chunks_for_code(max_chunk_size)
        total_chunks = [*text_chunks, *code_chunks]
        self._save_chunks(total_chunks)
        index_chunks(total_chunks)

    def _save_chunks(self, chunks) -> None:
        path = Path(CHUNKS_FILE)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(CHUNKS_FILE, mode="w") as file:
            saved = [m.model_dump() for m in chunks]

            json.dump(saved, file, indent=2)

    def search(self, query: str, k: int) -> MinimalSearchResults:
        sources = self._load_sources()
        if not sources:
            return []

        self.bm25 = bm25s.BM25.load(INDEX_DIR)

        k = min(k, len(sources))

        query_tokens = bm25s.tokenize(query, show_progress=False)
        indices, scores = self.bm25.retrieve(query_tokens, k=k, show_progress=False)

        results_texts = [
            sources[i]
            for i, score in zip(indices[0], scores[0])
            if score > 0
        ]

        results = MinimalSearchResults(question_id=str(uuid.uuid4()),
                                       question=query,
                                       retrieved_sources=results_texts)


        # print(f"QUESTION: {results.question}")
        # print(f"Results: {results.retrieved_sources}")

        return results

    def _load_sources(self) -> list[MinimalSource]:
        with open(CHUNKS_FILE, "r") as file:
            return [MinimalSource.model_validate(m) for m in json.load(file)]



if __name__ == "__main__":
    fire.Fire(RAG)

