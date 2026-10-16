"""Compute the recall@k of student search results against ground truth.

Mirrors the official moulinette: a correct source counts as found when a
retrieved source is in the same file and the character spans overlap with an
IoU strictly greater than 0.05.
"""
import json
from typing import Any, Dict, List, Sequence, Tuple

from pydantic import ValidationError

from .logger import Logger
from .models import MinimalSource, StudentSearchResults

K_VALUES = (1, 3, 5, 10)
IOU_THRESHOLD = 0.05


def _overlap_ratio(correct: MinimalSource, found: MinimalSource) -> float:
    """Return the IoU of the character spans of two sources."""
    start = max(correct.first_character_index, found.first_character_index)
    end = min(correct.last_character_index, found.last_character_index)
    inter = max(0, end - start)
    span = (
        max(correct.last_character_index, found.last_character_index)
        - min(correct.first_character_index, found.first_character_index)
    )
    return inter / span if span > 0 else 0.0


def _is_found(
    correct: MinimalSource, ranked: Sequence[MinimalSource]
) -> bool:
    """Tell whether one correct source is covered by the ranked results."""
    for result in ranked:
        if result.file_path != correct.file_path:
            continue
        if _overlap_ratio(correct, result) > IOU_THRESHOLD:
            return True
    return False


def _recall(
    correct: Sequence[MinimalSource],
    ranked: Sequence[MinimalSource],
    k: int,
) -> float:
    """Return the share of correct sources found in the top-k results."""
    if not correct:
        return 1.0
    if not ranked:
        return 0.0
    top = ranked[:k]
    found = sum(1 for source in correct if _is_found(source, top))
    return found / len(correct)


def _parse_source(raw: Dict[str, Any]) -> MinimalSource:
    """Build a MinimalSource from a ground-truth entry, tolerating gaps."""
    try:
        first = int(raw.get("first_character_index", 0))
        last = int(raw.get("last_character_index", 0))
    except (TypeError, ValueError):
        first = last = 0
    return MinimalSource(
        file_path=str(raw.get("file_path", "")),
        first_character_index=first,
        last_character_index=last,
        content=str(raw.get("content", "")),
    )


def _load_ground_truth(
    data: Any,
) -> Dict[str, Tuple[str, List[MinimalSource]]]:
    """Map each ground-truth question id to its text and correct sources.

    Accepts both the ``rag_questions`` (AnsweredQuestion) layout and the
    ``search_results`` reference layout. Malformed entries are skipped.
    """
    truth: Dict[str, Tuple[str, List[MinimalSource]]] = {}
    if not isinstance(data, dict):
        return truth
    items = data.get("rag_questions") or data.get("search_results") or []
    if not isinstance(items, list):
        return truth
    for item in items:
        if not isinstance(item, dict) or item.get("question_id") is None:
            continue
        sources = item.get("sources") or item.get("retrieved_sources") or []
        if not isinstance(sources, list):
            sources = []
        truth[str(item["question_id"])] = (
            str(item.get("question", "")),
            [_parse_source(s) for s in sources if isinstance(s, dict)],
        )
    return truth


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


def report_recall(student_path: str, dataset_path: str) -> None:
    """Print recall@k of student results against the ground-truth dataset."""
    student_data = _read_json(student_path)
    dataset_data = _read_json(dataset_path)
    if student_data is None or dataset_data is None:
        return

    try:
        student = StudentSearchResults.model_validate(student_data)
    except ValidationError as e:
        Logger.error(f"Invalid student search results: {e}")
        return

    truth = _load_ground_truth(dataset_data)
    by_id = {result.question_id: result for result in student.search_results}
    by_text = {result.question: result for result in student.search_results}

    if truth and all(not correct for _, correct in truth.values()):
        print("Warning: the ground truth has no sources.")
        print("Did you pass the AnsweredQuestions dataset?")

    scores: Dict[int, List[float]] = {k: [] for k in K_VALUES}
    for question_id, (question, correct) in truth.items():
        match = by_id.get(question_id) or by_text.get(question)
        ranked = match.retrieved_sources if match else []
        for k in K_VALUES:
            scores[k].append(_recall(correct, ranked, k))

    print("Evaluation Results")
    print("=" * 40)
    for k in K_VALUES:
        values = scores[k]
        mean = sum(values) / len(values) if values else 0.0
        print(f"Recall@{k}: {mean:.3f}")
