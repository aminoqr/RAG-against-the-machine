"""Bonus: local HTTP API. Exposes the exact same search/answer pipeline
the CLI uses -- every endpoint here is a thin wrapper around
retriever.search_loaded() / generator.answer_question(), the same
functions `uv run python -m src search` / `answer` call. No retrieval
or generation logic is duplicated between the two entrypoints.

Also demonstrates the caching bonus: IndexCache keeps the loaded
ChunkIndex/Bm25Index/embeddings in memory for the server's whole
lifetime (a stateless CLI invocation can't benefit from this -- it
exits after one call -- but a server handling many requests can), and
QueryCache short-circuits a repeated identical query entirely.

Run with:
    uv run uvicorn src.api:app --port 8000
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.cache import IndexCache, QueryCache
from src.generator import answer_question, load_model
from src.models import MinimalSource
from src.retriever import search_loaded

PROCESSED_DIR = Path("data/processed")

app = FastAPI(
    title="RAG against the machine",
    description="Search and answer over the vLLM corpus, same pipeline "
    "as the CLI.",
)

_index_cache = IndexCache(PROCESSED_DIR)
_query_cache = QueryCache()
_tokenizer: PreTrainedTokenizerBase | None = None
_model: PreTrainedModel | None = None


def _get_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load Qwen3-0.6B once for the server's lifetime, on first use."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer, _model = load_model()
    return _tokenizer, _model


class SearchResponse(BaseModel):
    """Response body for GET /search."""

    query: str
    k: int
    cache_hit: bool
    took_ms: float
    results: list[MinimalSource]


class AnswerResponse(BaseModel):
    """Response body for GET /answer."""

    query: str
    answer: str
    sources: list[MinimalSource]
    took_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    query: str, k: int = 10, use_semantic: bool = False
) -> SearchResponse:
    """Same ranking as `uv run python -m src search`, over HTTP.

    Args:
        query: the question to search for.
        k: maximum number of results.
        use_semantic: bonus flag -- fuse BM25 with the semantic index
            via Reciprocal Rank Fusion instead of BM25 alone.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    t0 = time.perf_counter()
    cached = _query_cache.get(query, k, use_semantic)
    if cached is not None:
        return SearchResponse(
            query=query,
            k=k,
            cache_hit=True,
            took_ms=(time.perf_counter() - t0) * 1000,
            results=cached,
        )

    try:
        chunk_index = _index_cache.chunk_index()
        bm25_index = _index_cache.bm25_index()
        embeddings = _index_cache.embeddings() if use_semantic else None
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="No index found -- run 'uv run python -m src index' first.",
        )

    results = search_loaded(query, k, chunk_index, bm25_index, embeddings)
    _query_cache.set(query, k, use_semantic, results)
    return SearchResponse(
        query=query,
        k=k,
        cache_hit=False,
        took_ms=(time.perf_counter() - t0) * 1000,
        results=results,
    )


@app.get("/answer", response_model=AnswerResponse)
def answer_endpoint(query: str, k: int = 10) -> AnswerResponse:
    """Same pipeline as `uv run python -m src answer`, over HTTP.

    Args:
        query: the question to answer.
        k: number of retrieved sources to ground the answer in.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    t0 = time.perf_counter()
    try:
        chunk_index = _index_cache.chunk_index()
        bm25_index = _index_cache.bm25_index()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="No index found -- run 'uv run python -m src index' first.",
        )

    sources: list[MinimalSource] = search_loaded(
        query, k, chunk_index, bm25_index
    )
    if not sources:
        raise HTTPException(
            status_code=404,
            detail="No relevant sources found -- cannot answer.",
        )

    tokenizer, model = _get_model()
    result = answer_question(query, sources, tokenizer, model)
    return AnswerResponse(
        query=query,
        answer=result.answer,
        sources=result.retrieved_sources,
        took_ms=(time.perf_counter() - t0) * 1000,
    )
