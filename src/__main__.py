from .chunking import create_chunks_for_code, create_chunks_for_text
from .evaluate import report_recall
from .indexing import index_chunks
from .models import MinimalSearchResults, MinimalSource, RagDataset, UnansweredQuestion, AnsweredQuestion, StudentSearchResults
import fire
import json
from pathlib import Path
import bm25s
import uuid
from tqdm import tqdm


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

    def search(self, query: str, k: int,
               question_id: str | None = None) -> MinimalSearchResults:
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

        results = MinimalSearchResults(
            question_id=question_id or str(uuid.uuid4()),
            question=query,
            retrieved_sources=results_texts,
        )


        # print(f"QUESTION: {results.question}")
        # print(f"Results: {results.retrieved_sources}")

        return results

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        with open(dataset_path, "r") as file:
            data_questions = json.load(file).get("rag_questions", "")

        rag_questions = self._dict_to_unanswered_object(data_questions)
        data = RagDataset(rag_questions=rag_questions)

        results = [self.search(q.question, k, q.question_id)
                   for q in tqdm(data.rag_questions)]

        path = Path(save_directory)
        path.mkdir(exist_ok=True, parents=True)
        out_file = Path(dataset_path).name

        with open(path / out_file, mode="w", encoding="utf-8") as file:
            output = StudentSearchResults(search_results=results, k=k)
            json.dump(output.model_dump(), file, indent=2, ensure_ascii=False)

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        report_recall(student_search_results_path, dataset_path)

    def _dict_to_unanswered_object(self, dataset: dict) -> RagDataset:
        return [UnansweredQuestion.model_validate(m) for m in dataset]

    def _load_sources(self) -> list[MinimalSource]:
        with open(CHUNKS_FILE, "r") as file:
            return [MinimalSource.model_validate(m) for m in json.load(file)]



if __name__ == "__main__":
    fire.Fire(RAG)

