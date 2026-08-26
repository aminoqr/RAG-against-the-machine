"""Given a query, score it against the persisted BM25 index and return
the top-k ranked source locations. Also runs that same search in batch
over a whole dataset of questions."""

import json
from pathlib import Path

from pydantic import ValidationError
from tqdm import tqdm

from src.bm25 import Bm25Index, load_bm25_index, score_query
from src.indexer import ChunkIndex, load_index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.tokenizer import tokenize


def _search_loaded(
    query: str, k: int, chunk_index: ChunkIndex, bm25_index: Bm25Index
) -> list[MinimalSource]:
    """Score query against an already-loaded index and return the
    top-k ranked results. Shared by search() (loads once per call)
    and search_dataset() (loads onnce for the whole batch)."""
    if k <= 0 or not query or not query.strip():
        return []

    query_tokens = tokenize(query)
    scores = score_query(query_tokens, bm25_index)

    ranked = sorted(
        (
            (score, source)
            for score, source in zip(scores, chunk_index.chunks)
            if score > 0
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [source for _, source in ranked[:k]]


def search(query: str, k: int, processed_dir: Path) -> list[MinimalSource]:
    """Return the top-k chunks most relevant to query.

    Args:
        query: the natural-language or code question to search for.
        k: maximum number of results to return.
        processed_dir: directory previously passed to the 'index'
            command's save_index/save_bm25_index (data/processed).

    Returns:
        Up to k MinimalSource results, ranked by BM25 score
        descending, restricted to chunks that scored above zero.
    """
    chunk_index = load_index(processed_dir)
    bm25_index = load_bm25_index(processed_dir)
    return _search_loaded(query, k, chunk_index, bm25_index)


def search_dataset(
    dataset_path: Path, k: int, save_directory: Path, processed_dir: Path
) -> Path:
    """Run search for every question in a dataset file, and persist a
    StudentSearchResults JSON under save_directory.

    Loads the index once for the whole batch (unlike calling search()
    per question, which would reload the same multi-megabyte JSON
    files from disk on every single question).

    Args:
        dataset_path: path to an UnansweredQuestions/AnsweredQuestions
            JSON file (a RagDataSet).
        k: max number of results to return per question.
        save_directory: directory to write the output JSON into
            (created if missing); the output file keeps the input
            dataset's filename(e.g. dataset_docs_public.json).
        processed_dir: directory previously written by 'index;
            (data/processed).

    Returns:
        The path of the written StudentSearchResults JSON file.

    Raise:
        FileNotFoundError: if dataset_path or the index under
            processed_dir don't exist.
        ValueError: if dataset_path isn't valid JSON, or doesn't match
            the expected dataset schema.
    """
    try:
        with dataset_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{dataset_path} is not valid JSON: {e}") from e

    try:
        dataset = RagDataset.model_validate(raw)
    except ValidationError as e:
        raise ValueError(
            f"{dataset_path} doesn't match"
            f"the expected dataset format: {e}"
        ) from e

    chunk_index = load_index(processed_dir)
    bm25_index = load_bm25_index(processed_dir)

    results: list[MinimalSearchResults] = []
    for question in tqdm(dataset.rag_questions, desc="Searching"):
        sources = _search_loaded(
            question.question, k, chunk_index, bm25_index
        )
        results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            )
        )

    output = StudentSearchResults(search_results=results, k=k)

    save_directory.mkdir(parents=True, exist_ok=True)
    out_path = save_directory / dataset_path.name
    with out_path.open("w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    return out_path
