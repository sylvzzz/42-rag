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
from .logger import FatalError, Logger
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

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000
MAX_K = 10
MAX_ANSWER_SOURCES = 5
MAX_QUERY_CHARS = 2000

RUNTIME_ERRORS = (OSError, ValueError, TypeError, RuntimeError)
NUMBER_ERRORS = (TypeError, ValueError, OverflowError)


class RAG:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self) -> None:
        """Set up the pipeline with a lazily-loaded index and sources."""
        self._bm25: bm25s.BM25 | None = None
        self._sources: List[MinimalSource] | None = None
        self._engine: Qwen | None = None
        self._engine_lock = threading.Lock()

    def index(self, max_chunk_size: int = 2000) -> None:
        """Chunk data/raw and build the index under data/processed/."""
        size = self._clean_chunk_size(max_chunk_size)
        try:
            text_chunks = create_chunks_for_text(size)
            code_chunks = create_chunks_for_code(size)
        except RUNTIME_ERRORS as e:
            Logger.error(f"Could not read the corpus: {e}")
            raise FatalError from e

        total_chunks = [*text_chunks, *code_chunks]
        if not total_chunks:
            Logger.error("No files found under data/raw/vllm-0.10.1.")
            raise FatalError

        self._save_chunks(total_chunks)
        try:
            index_chunks(total_chunks)
        except RUNTIME_ERRORS as e:
            Logger.error(f"Indexing failed: {e}")
            raise FatalError from e

        self._sources = None
        self._bm25 = None
        Logger.success(f"Successfully indexed {len(total_chunks)} chunks!")

    def search(self, query: str, k: int,
               question_id: str | None = None) -> MinimalSearchResults:
        """Return the top-k sources for a single query."""
        empty = MinimalSearchResults(
            question_id=question_id or str(uuid.uuid4()),
            question=query,
            retrieved_sources=[],
        )
        clean_k = self._clean_k(k)
        if not query.strip():
            Logger.warning("Empty query; returning no sources.")
            return empty

        sources = self._load_sources()
        if not sources:
            return empty

        k = min(clean_k, len(sources))
        try:
            bm25 = self._load_index()
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
            raise FatalError from e

        return MinimalSearchResults(
            question_id=empty.question_id,
            question=query,
            retrieved_sources=found,
        )

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        """Search a whole dataset and save a StudentSearchResults JSON."""
        clean_k = self._clean_k(k)
        data = self._read_json(dataset_path)
        raw = data.get("rag_questions") if isinstance(data, dict) else None
        if raw is None:
            Logger.error(f"Dataset '{dataset_path}' has no 'rag_questions'.")
            raise FatalError

        questions = self._to_questions(raw)
        results = [self.search(q.question, clean_k, q.question_id)
                   for q in tqdm(questions, desc="Searching")]
        output = StudentSearchResults(search_results=results, k=clean_k)
        self._save_json(save_directory, dataset_path, output.model_dump())

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Report recall@k against a ground-truth dataset."""
        report_recall(student_search_results_path, dataset_path)

    def answer(self, query: str, k: int) -> str:
        """Answer a single query from the retrieved sources."""
        if not query.strip():
            Logger.warning("Empty question; nothing to answer.")
            return "Please provide a non-empty question."
        clean_k = self._clean_k(k)
        query = self._truncate_query(query)

        results = self.search(query, clean_k)
        if not results.retrieved_sources:
            Logger.warning("No sources found; not calling the model.")
            return "No relevant sources were found for this question."

        sources = results.retrieved_sources[:MAX_ANSWER_SOURCES]
        try:
            return self._load_engine().answer(query, sources)
        except RUNTIME_ERRORS as e:
            Logger.error(f"Answer generation failed: {e}")
            raise FatalError from e

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate answers for a dataset and save them as JSON."""
        data = self._read_json(student_search_results_path)
        try:
            student = StudentSearchResults.model_validate(data)
        except ValidationError as e:
            Logger.error(f"Invalid search results: {e}")
            raise FatalError from e

        try:
            engine = self._load_engine()
        except RUNTIME_ERRORS as e:
            Logger.error(f"Could not load the model: {e}")
            raise FatalError from e

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

    @staticmethod
    def _clean_k(value: Any) -> int:
        """Return ``k`` as a positive int, clamped to ``MAX_K``."""
        try:
            k = int(value)
        except NUMBER_ERRORS as e:
            Logger.error(f"k must be an integer (got {value!r}).")
            raise FatalError from e
        if k <= 0:
            Logger.error(f"k must be positive (got {k}).")
            raise FatalError
        if k > MAX_K:
            Logger.warning(f"k={k} is too large; using {MAX_K}.")
            return MAX_K
        return k

    @staticmethod
    def _clean_chunk_size(value: Any) -> int:
        """Return ``max_chunk_size`` clamped to a sane range."""
        try:
            size = int(value)
        except NUMBER_ERRORS as e:
            Logger.error(
                f"max_chunk_size must be an integer (got {value!r})."
            )
            raise FatalError from e
        if size <= 0:
            Logger.error(f"max_chunk_size must be positive (got {size}).")
            raise FatalError
        if size < MIN_CHUNK_SIZE:
            Logger.warning(
                f"max_chunk_size={size} is too small; using {MIN_CHUNK_SIZE}."
            )
            return MIN_CHUNK_SIZE
        if size > MAX_CHUNK_SIZE:
            Logger.warning(
                f"max_chunk_size={size} is too large; using {MAX_CHUNK_SIZE}."
            )
            return MAX_CHUNK_SIZE
        return size

    @staticmethod
    def _truncate_query(query: str) -> str:
        """Cap the query length so it never floods the model context."""
        if len(query) > MAX_QUERY_CHARS:
            Logger.warning(
                f"Query is too long ({len(query)} chars); "
                f"truncating to {MAX_QUERY_CHARS}."
            )
            return query[:MAX_QUERY_CHARS]
        return query

    def _save_chunks(self, chunks: List[MinimalSource]) -> None:
        """Persist the chunks as JSON."""
        try:
            path = Path(CHUNKS_FILE)
            path.parent.mkdir(exist_ok=True, parents=True)
            with open(CHUNKS_FILE, mode="w") as file:
                json.dump([m.model_dump() for m in chunks], file, indent=2)
        except OSError as e:
            Logger.error(f"Could not write the chunks: {e}")
            raise FatalError from e
        Logger.success(f"Saved {len(chunks)} sources to {CHUNKS_FILE}.")

    @staticmethod
    def _answer_one(engine: Qwen,
                    result: MinimalSearchResults) -> MinimalAnswer:
        """Build a MinimalAnswer for one search result, never raising."""
        sources = result.retrieved_sources[:MAX_ANSWER_SOURCES]
        try:
            text = engine.answer(result.question, sources)
        except RUNTIME_ERRORS as e:
            Logger.warning(f"Skipping one answer: {e}")
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
            destination = path / Path(source_path).name
            with open(destination, mode="w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
        except OSError as e:
            Logger.error(f"Could not write the output: {e}")
            raise FatalError from e
        Logger.success(f"Saved {destination}.")

    @staticmethod
    def _read_json(path: str) -> Any:
        """Read a JSON file, raising ``FatalError`` on any failure."""
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError as e:
            Logger.error(f"File not found: {path}")
            raise FatalError from e
        except json.JSONDecodeError as e:
            Logger.error(f"Malformed JSON in {path}")
            raise FatalError from e
        except OSError as e:
            Logger.error(f"Could not read {path}")
            raise FatalError from e

    @staticmethod
    def _to_questions(raw: Any) -> List[UnansweredQuestion]:
        """Validate raw question dicts, skipping the malformed ones."""
        if not isinstance(raw, list):
            Logger.error("'rag_questions' must be a list.")
            raise FatalError
        questions: List[UnansweredQuestion] = []
        for item in raw:
            try:
                questions.append(UnansweredQuestion.model_validate(item))
            except ValidationError as e:
                Logger.warning(f"Skipping a malformed question entry: {e}")
        return questions

    def _load_index(self) -> bm25s.BM25:
        """Load (once) and return the persisted BM25 index."""
        bm25 = self._bm25
        if bm25 is None:
            try:
                bm25 = bm25s.BM25.load(INDEX_DIR)
            except RUNTIME_ERRORS as e:
                Logger.error(f"Could not load the index: {e}")
                raise FatalError from e
            self._bm25 = bm25
        return bm25

    def _load_engine(self) -> Qwen:
        """Load (once) and return the Qwen answer engine."""
        with self._engine_lock:
            if self._engine is None:
                self._engine = Qwen()
        return self._engine

    def _load_sources(self) -> List[MinimalSource]:
        """Load (once) and return the persisted chunks."""
        sources = self._sources
        if sources is None:
            data = self._read_json(CHUNKS_FILE)
            if not isinstance(data, list):
                Logger.error(f"{CHUNKS_FILE} must contain a JSON list.")
                raise FatalError
            sources = []
            for item in data:
                try:
                    sources.append(MinimalSource.model_validate(item))
                except ValidationError as e:
                    Logger.warning(f"Skipping a malformed chunk: {e}")
            self._sources = sources
        return sources
