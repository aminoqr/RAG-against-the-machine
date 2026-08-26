"""BM25 statistical index: fit term/document frequencies over every
chunk produced by the `index` command, persist them, and score a
tokenized query against every chunk at search time."""

import math
from collections import Counter, defaultdict
from pathlib import Path
from pydantic import BaseModel
from src.indexer import ChunkIndex
from src.tokenizer import tokenize

K1 = 1.5
B = 0.75


class Bm25Index(BaseModel):
    """A fitted BM25 index: corpus-wide statistics plus, per chunk (in
    the same order as ChunkIndex.chunks), how many times each term
    appears in it."""

    k1: float
    b: float
    n_chunks: int
    avg_chunk_length: float
    document_frequency: dict[str, int]
    chunk_term_frequencies: list[dict[str, int]]


def build_bm25_index(chunk_index: ChunkIndex) -> Bm25Index:
    """Fit a BM25 index over every chunk in chunk-index, re-reading each
    chunk's exact text slice from its source file (grouped per file so
    each file is opened once, no matter how many chunks it produced).

    Args:
        chunk_index: the ChunkIndex produced by build_inex.

    Returns:
        A fitted Bm25Index, with chunk_term_frequencies aligned
        index-for-index with chunk-index.chunks.
    """
    chunks_by_file: dict[str, list[int]] = defaultdict(list)
    for i, source in enumerate(chunk_index.chunks):
        chunks_by_file[source.file_path].append(i)

    n_chunks = len(chunk_index.chunks)
    chunk_term_frequencies: list[dict[str, int]] = [{}
                                                    for _ in range(n_chunks)]
    document_frequency: Counter[str] = Counter()
    total_length = 0

    for file_path, indices in chunks_by_file.items():
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for i in indices:
            source = chunk_index.chunks[i]
            slice_text = text[
                source.first_character_index:source.last_character_index
            ]
            tf = Counter(tokenize(slice_text))
            chunk_term_frequencies[i] = dict(tf)
            total_length += sum(tf.values())
            document_frequency.update(tf.keys())

    avg_chunk_length = total_length / n_chunks if n_chunks else 0.0

    return Bm25Index(
        k1=K1,
        b=B,
        n_chunks=n_chunks,
        avg_chunk_length=avg_chunk_length,
        document_frequency=dict(document_frequency),
        chunk_term_frequencies=chunk_term_frequencies,
    )


def _idf(term: str, index: Bm25Index) -> float:
    """BM25's IDF variant: always non-negative, unlike plain log(n/df),
    so an extremely common term contributes ~0 instead of a negative
    score that woudl penalize chunks for containing it."""
    df = index.document_frequency.get(term, 0)
    return math.log((index.n_chunks - df + 0.5) / (df + 0.5) + 1)


def score_query(query_tokens: list[str], index: Bm25Index) -> list[float]:
    """Score every chunk in index against an already-tokenized query.

    Args:
        query_tokens: the query, tokenized with the same tokenize()
            function used at index time.
        index: a fitted Bm25Index.

    Returns:
        One BM25 score per chunk, in the same order as
        index.chunk_term_frequencies (and thus ChunkIndex.chunks).
    """
    scores = [0.0] * index.n_chunks
    if not query_tokens or index.n_chunks == 0 or index.avg_chunk_length == 0:
        return scores

    unique_terms = set(query_tokens)
    idfs = {term: _idf(term, index) for term in unique_terms}

    for i, tf in enumerate(index.chunk_term_frequencies):
        chunk_length = sum(tf.values())
        norm = 1 - index.b + index.b * (chunk_length / index.avg_chunk_length)
        score = 0.0
        for term in unique_terms:
            term_freq = tf.get(term, 0)
            if term_freq == 0:
                continue
            score += (
                idfs[term]
                * (term_freq * (index.k1 + 1))
                / (term_freq + index.k1 * norm)
            )
        scores[i] = score
    return scores


def save_bm25_index(index: Bm25Index, save_directory: Path) -> Path:
    """Persist index as JSON under save_direcotry/bm25_index.json."""
    save_directory.mkdir(parents=True, exist_ok=True)
    out_path = save_directory / "bm25_index.json"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(index.model_dump_json(indent=2))
    return out_path


def load_bm25_index(save_directory: Path) -> Bm25Index:
    """Load a previously persisted Bm25Index from save_directory."""
    path = save_directory / "bm25_index.json"
    with path.open("r", encoding="utf-8") as f:
        return Bm25Index.model_validate_json(f.read())
