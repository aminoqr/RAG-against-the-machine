"""Recall@k evaluation against ground-truth AnswerQuestions, matching
moulinette's own definition: same file+path plus IoU >= 0.05 overlap."""

import json
from pathlib import Path

from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

IOU_THRESHOLD = 0.05
RECALL_KS = (1, 3, 5, 10)


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection-over-union of two character ranges, 0 if they're
    in different files or don't overlap at all.
    
    Args:
        a: first source location.
        b: second source location.
    
    Returns:
        A float in [0, 1]; 0 when the files differ or the ranges
        don't overlap, 1 when the ranges are identical.
    """
    if a.file_path != b.file_path:
        return 0.0
    
    inter_start = max(a.first_character_index, b.first_character_index)
    inter_end = min(a.last_character_index, b.last_character_index)
    intersection = max(0, inter_end - inter_start)
    if intersection == 0:
        return 0.0
    
    union_start = min(a.first_character_index, b.first_character_index)
    union_end = max(a.last_character_index, b.last_character_index)
    union = union_end - union_start
    return intersection / union if union > 0 else 0.0


def _is_hit(
    ground_truth: MinimalSource, retrieved: list[MinimalSource]
) -> bool:
    """Whether any retrieved source counts as a match for one
    ground-truth source: same file_path, IoU >= 0.05"""
    return any(iou(ground_truth, r) >= IOU_THRESHOLD for r in retrieved)


def _recall_at_k(
    ground_truth_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    k: int,
) -> float | None:
    """Share of ground_truth_sources found within the first k
    retrieved_sources. None if there are no ground-truth sources to
    find (recall is undefined, not zero, for such a question)."""
    if not ground_truth_sources:
        return None
    top_k = retrieved_sources[:k]
    hits = sum(1 for gt in ground_truth_sources if _is_hit(gt, top_k))
    return hits / len(ground_truth_sources)


def evaluate(
    student_search_results_path: Path, dataset_path: Path
) -> dict[int, float]:
    """Compute recall@k (k in RECALL_KS) of student_search_results
    against the ground-truth AnsweredQuestions dataset_path.
    
    Args:
        student_search_results_path: a StudentSearchResults JSON file
            (search_dataset's output).
        dataset_path: the matching ground-truth AnsweredQuestions
            dataset (must have real 'sources' per question).
    
    Returns:
        A dict mapping each k in RECALL_KS to the mean recall@k across
        every question that has at least one ground-truth source.
        
    Raises:
        FileNotFoundError: if either path doesn't exist.
        ValueError: if either file isn't valid JSON, or doesn't match
            the expected schema.
    """
    try:
        with student_search_results_path.open("r",encoding="utf-8") as f:
            results_raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{student_search_results_path} is not valid JSON: {e}"
        ) from e
    try:
        student_results = StudentSearchResults.model_validate(results_raw)
    except ValidationError as e:
        raise ValueError(
            f"{student_search_results_path} doesn't match the expected "
            f"StudentSearchResults format : {e}"
        ) from e
    
    try:
        with dataset_path.open("r", encoding="utf-8") as f:
            dataset_raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{dataset_path} is not valid JSON: {e}") from e
    try:
        dataset = RagDataset.model_validate(dataset_raw)
    except ValidationError as e:
        raise ValueError(
            f"{dataset_path} doesn't match the expected dataset format: {e}"
        ) from e
    
    ground_truth_by_id = {
        q.question_id: q.sources
        for q in dataset.rag_questions
        if isinstance(q, AnsweredQuestion)
    }
    
    totals:  dict[int, float] = {k: 0.0 for k in RECALL_KS}
    counts: dict[int, int] = {k: 0 for k in RECALL_KS}
    
    for result in student_results.search_results:
        ground_truth_sources = ground_truth_by_id.get(result.question_id)
        if ground_truth_sources is None:
            continue
        for k in RECALL_KS:
            recall = _recall_at_k(
                ground_truth_sources,result.retrieved_sources, k
            )
            if recall is None:
                continue
            totals[k] += recall
            counts[k] += 1
                
    return {
        k: (totals[k] / counts[k] if counts[k] else 0.0) for k in RECALL_KS
    }
