"""Retrieval-Augmented Generation pipeline."""
import json
import threading
import uuid
from pathlib import Path
from typing import Any, List

import bm25s
from pydantic import ValidationError
from tqdm import tqdm

from .chunking import create_chunks_for_code, create_chunks_for_text
from .evaluate import report_recall
from .indexing import INDEX_DIR, index_chunks
from .logger import Logger
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from .qwen import Qwen

CHUNKS_FILE = "data/processed/index.json"

# Concrete, catchable failures from file/JSON/model libraries we depend on.
RUNTIME_ERRORS = (OSError, ValueError, RuntimeError)


class RAG:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self) -> None:
        """Set up the pipeline with a lazily-loaded index and sources."""
        self.bm25: bm25s.BM25 | None = None
        self._sources: List[MinimalSource] | None = None
        self._engine: Qwen | None = None
        self._engine_lock = threading.Lock()

    def index(self, max_chunk_size: int) -> None:
        """Chunk data/raw and build the index under data/processed/."""
        if max_chunk_size <= 0:
            Logger.error("max_chunk_size must be a positive integer.")
            return
        try:
            text_chunks = create_chunks_for_text(max_chunk_size)
            code_chunks = create_chunks_for_code(max_chunk_size)
        except OSError as e:
            Logger.error(f"Could not read the corpus: {e}")
            return

        total_chunks = [*text_chunks, *code_chunks]
        if not total_chunks:
            Logger.error("No files found under data/raw/vllm-0.10.1.")
            return

        self._save_chunks(total_chunks)
        try:
            index_chunks(total_chunks)
        except RUNTIME_ERRORS as e:
            Logger.error(f"Indexing failed: {e}")
            return

        self._sources = None
        self.bm25 = None
        Logger.success(f"Indexed {len(total_chunks)} chunks.")

    def search(self, query: str, k: int,
               question_id: str | None = None) -> MinimalSearchResults:
        """Return the top-k sources for a single query."""
        empty = MinimalSearchResults(
            question_id=question_id or str(uuid.uuid4()),
            question=query,
            retrieved_sources=[],
        )
        if k <= 0 or not query.strip():
            return empty

        sources = self._load_sources()
        if not sources:
            return empty

        k = min(k, len(sources))
        try:
            bm25 = self._load_index()
            if bm25 is None:
                return empty
            query_tokens = bm25s.tokenize(query, show_progress=False)
            indices, scores = bm25.retrieve(
                query_tokens, k=k, show_progress=False
            )
            found = [
                sources[i]
                for i, score in zip(indices[0], scores[0])
                if score > 0
            ]
        except RUNTIME_ERRORS + (KeyError, IndexError) as e:
            Logger.error(f"Search failed: {e}")
            return empty

        return MinimalSearchResults(
            question_id=empty.question_id,
            question=query,
            retrieved_sources=found,
        )

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        """Search a whole dataset and save a StudentSearchResults JSON."""
        data = self._read_json(dataset_path)
        if data is None:
            return
        raw = data.get("rag_questions") if isinstance(data, dict) else None
        if raw is None:
            Logger.error(f"Dataset '{dataset_path}' has no 'rag_questions'.")
            return

        questions = self._to_questions(raw)
        results = [self.search(q.question, k, q.question_id)
                   for q in tqdm(questions, desc="Searching")]
        output = StudentSearchResults(search_results=results, k=k)
        self._save_json(save_directory, dataset_path, output.model_dump())

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Report recall@k against a ground-truth dataset."""
        report_recall(student_search_results_path, dataset_path)

    def answer(self, query: str, k: int) -> str:
        """Answer a single query from the retrieved sources."""
        if not query.strip():
            Logger.error("Empty question.")
            return "Please provide a non-empty question."

        results = self.search(query, k)
        if not results.retrieved_sources:
            Logger.warning("No sources found; not calling the model.")
            return "No relevant sources were found for this question."

        try:
            return self._load_engine().answer(
                query, results.retrieved_sources
            )
        except RUNTIME_ERRORS as e:
            Logger.error(f"Answer generation failed: {e}")
            return "Unable to generate an answer."

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate answers for a dataset and save them as JSON."""
        data = self._read_json(student_search_results_path)
        if data is None:
            return
        try:
            student = StudentSearchResults.model_validate(data)
        except ValidationError as e:
            Logger.error(f"Invalid search results: {e}")
            return

        try:
            engine = self._load_engine()
        except RUNTIME_ERRORS as e:
            Logger.error(f"Could not load the model: {e}")
            return

        answers = [
            self._answer_one(engine, result)
            for result in tqdm(student.search_results, desc="Answering")
        ]
        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=student.k
        )
        self._save_json(
            save_directory, student_search_results_path, output.model_dump()
        )

    def _save_chunks(self, chunks: List[MinimalSource]) -> None:
        """Persist the chunks as JSON."""
        try:
            path = Path(CHUNKS_FILE)
            path.parent.mkdir(exist_ok=True, parents=True)
            with open(CHUNKS_FILE, mode="w") as file:
                json.dump([m.model_dump() for m in chunks], file, indent=2)
        except OSError as e:
            Logger.error(f"Could not write the chunks: {e}")

    @staticmethod
    def _answer_one(engine: Qwen,
                    result: MinimalSearchResults) -> MinimalAnswer:
        """Build a MinimalAnswer for one search result, never raising."""
        try:
            text = engine.answer(result.question, result.retrieved_sources)
        except RUNTIME_ERRORS as e:
            Logger.error(f"Answer generation failed: {e}")
            text = "Unable to generate an answer."
        return MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=text,
        )

    @staticmethod
    def _save_json(save_directory: str, source_path: str,
                   payload: dict[str, Any]) -> None:
        """Write ``payload`` into ``save_directory`` with the source name."""
        try:
            path = Path(save_directory)
            path.mkdir(exist_ok=True, parents=True)
            with open(path / Path(source_path).name, mode="w",
                      encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
        except OSError as e:
            Logger.error(f"Could not write the output: {e}")

    @staticmethod
    def _read_json(path: str) -> Any:
        """Read a JSON file, returning ``None`` on any read/parse error."""
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            Logger.error(f"File not found: {path}")
        except json.JSONDecodeError as e:
            Logger.error(f"Malformed JSON in {path}: {e}")
        except OSError as e:
            Logger.error(f"Could not read {path}: {e}")
        return None

    @staticmethod
    def _to_questions(raw: Any) -> List[UnansweredQuestion]:
        """Validate raw question dicts, skipping the malformed ones."""
        questions: List[UnansweredQuestion] = []
        if not isinstance(raw, list):
            Logger.error("'rag_questions' must be a list.")
            return questions
        for item in raw:
            try:
                questions.append(UnansweredQuestion.model_validate(item))
            except ValidationError:
                Logger.warning("Skipping a malformed question entry.")
        return questions

    def _load_index(self) -> bm25s.BM25 | None:
        """Load (once) and return the persisted BM25 index, or ``None``."""
        if self.bm25 is None:
            try:
                self.bm25 = bm25s.BM25.load(INDEX_DIR)
            except RUNTIME_ERRORS as e:
                Logger.error(f"Could not load the index: {e}")
                return None
        return self.bm25

    def _load_engine(self) -> Qwen:
        """Load (once) and return the Qwen answer engine."""
        with self._engine_lock:
            if self._engine is None:
                self._engine = Qwen()
        return self._engine

    def _load_sources(self) -> List[MinimalSource]:
        """Load (once) and return the persisted chunks."""
        if self._sources is None:
            data = self._read_json(CHUNKS_FILE)
            if data is None:
                return []
            sources: List[MinimalSource] = []
            if isinstance(data, list):
                for item in data:
                    try:
                        sources.append(MinimalSource.model_validate(item))
                    except ValidationError:
                        Logger.warning("Skipping a malformed chunk.")
            self._sources = sources
        return self._sources
