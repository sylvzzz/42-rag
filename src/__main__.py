from .chunking import create_chunks_for_code, create_chunks_for_text
from .indexing import index_chunks
from .models import MinimalSearchResults, MinimalSource
import fire
import json
from pathlib import Path


CHUNKS_FILE = "data/processed/index.json"


class RAG:
    def index(self, max_chunk_size: int) -> None:
        text_chunks = create_chunks_for_text(max_chunk_size)
        code_chunks = create_chunks_for_code(max_chunk_size)
        total_chunks = [*text_chunks, *code_chunks]
        self._save_chunks(total_chunks)
        index_chunks(total_chunks)

    def _save_chunks(self, chunks) -> None:
        path =  Path(CHUNKS_FILE)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(CHUNKS_FILE, mode="w") as file:
            saved = [m.model_dump() for m in chunks]

            json.dump(saved, file, indent=2)


    def search(self, query: str, k: int) -> MinimalSearchResults:
        sources = self._load_sources()
        return sources

    def _load_sources(self) -> list[MinimalSource]:
        with open(CHUNKS_FILE, "r") as file:
            return [MinimalSource.model_validate(m) for m in json.load(file)]



if __name__ == "__main__":
    fire.Fire(RAG)

