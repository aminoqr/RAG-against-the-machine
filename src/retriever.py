"""Given a query, score it against the persisted BM25 index and return
the top-k ranked source locations. Also runs that same search in batch
over a whole dataset of questions."""

import json
from pathlib import Path

from pydantic import ValidationError
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray

from src.bm25 import Bm25Index, load_bm25_index, score_query
from src.embeddings import load_embedding_index
from src.embeddings import score_query as embedding_score_query
from src.hybrid import BM25_WEIGHT, SEMANTIC_WEIGHT, reciprocal_rank_fusion
from src.indexer import ChunkIndex, load_index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.tokenizer import tokenize

# Bonus: hybrid retrieval widens each individual ranker's candidate
# pool before fusing, so RRF has more than k candidates per side to
# actually re-rank -- fusing two already-truncated top-k lists would
# rarely change anything.
CANDIDATE_POOL = 30


def _rank_bm25(
    query: str, k: int, chunk_index: ChunkIndex, bm25_index: Bm25Index
) -> list[MinimalSource]:
    """BM25-only ranking: tokenize, score every chunk, keep the top-k
    that scored above zero."""
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


def _rank_semantic(
    query: str,
    k: int,
    chunk_index: ChunkIndex,
    embeddings: NDArray[np.float32],
) -> list[MinimalSource]:
    """Semantic-only ranking (bonus): cosine similarity in embedding
    space, top-k above zero."""
    scores = embedding_score_query(query, embeddings)
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


def search_loaded(
    query: str,
    k: int,
    chunk_index: ChunkIndex,
    bm25_index: Bm25Index,
    embeddings: NDArray[np.float32] | None = None,
) -> list[MinimalSource]:
    """Score query against an already-loaded index and return the
    top-k ranked results. Shared by search() (loads once per call)
    and search_dataset() (loads once for the whole batch).

    Lexical-only (default, mandatory-part behavior) when embeddings is
    None. When embeddings is given (bonus: hybrid retrieval), fuses
    the BM25 ranking and the semantic ranking with Reciprocal Rank
    Fusion instead of returning BM25 alone.
    """
    if k <= 0 or not query or not query.strip():
        return []

    pool = max(k, CANDIDATE_POOL)
    bm25_ranked = _rank_bm25(query, pool, chunk_index, bm25_index)

    if embeddings is None:
        return bm25_ranked[:k]

    semantic_ranked = _rank_semantic(query, pool, chunk_index, embeddings)
    fused = reciprocal_rank_fusion(
        bm25_ranked,
        semantic_ranked,
        weights=(BM25_WEIGHT, SEMANTIC_WEIGHT),
    )
    return fused[:k]


def search(
    query: str,
    k: int,
    processed_dir: Path,
    use_semantic: bool = False,
) -> list[MinimalSource]:
    """Return the top-k chunks most relevant to query.

    Args:
        query: the natural-language or code question to search for.
        k: maximum number of results to return.
        processed_dir: directory previously passed to the 'index'
            command's save_index/save_bm25_index (data/processed).
        use_semantic: bonus flag. When True, also loads the semantic
            embedding index and fuses it with BM25 via Reciprocal Rank
            Fusion instead of using BM25 alone.

    Returns:
        Up to k MinimalSource results, ranked descending, restricted
        to chunks that scored above zero in at least one ranker.
    """
    chunk_index = load_index(processed_dir)
    bm25_index = load_bm25_index(processed_dir)
    embeddings = load_embedding_index(processed_dir) if use_semantic else None
    return search_loaded(query, k, chunk_index, bm25_index, embeddings)


def search_dataset(
    dataset_path: Path,
    k: int,
    save_directory: Path,
    processed_dir: Path,
    use_semantic: bool = False,
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
        use_semantic: bonus flag, see search().

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
    embeddings = load_embedding_index(processed_dir) if use_semantic else None

    results: list[MinimalSearchResults] = []
    for question in tqdm(dataset.rag_questions, desc="Searching"):
        sources = search_loaded(
            question.question, k, chunk_index, bm25_index, embeddings
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
