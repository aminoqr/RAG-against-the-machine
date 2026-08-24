"""Given a query, score it against the persisted BM25 index and return
the top-k ranked source locations."""

from pathlib import Path

from src.bm25 import load_bm25_index, score_query
from src.indexer import load_index
from src.models import MinimalSource
from src.tokenizer import tokenize


def search(query: str, k: int, processed_dir: Path) -> list[MinimalSource]:
    """Return the top-k chunks most relevant to query.    
    Args:
        query: the natural-language or code question to search for.
        k: maximum number of results to return.
        processed_dir: directory previously passed to the 'index'
            command's save_index/save_bm25_index
    
    Returns:
        Up to k MinimalSource results, ranaked by BM25 score
        descending, restricted to chunks that scored above zero. An
        empty list if k <= 0, the query is empty/has no known terms,
        or nothing scored above zero.
    """
    if k <= 0 or not query or not query.strip():
        return []
    
    chunk_index = load_index(processed_dir)
    bm25_index = load_bm25_index(processed_dir)
    
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
