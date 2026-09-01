"""Semantic (embedding-based) index: a lightweight CPU sentence-transformer
turns every chunk into a vector, so retrieval can also rank by meaning
instead of only shared words -- catches a paraphrased question that BM25
would miss entirely."""

from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.indexer import ChunkIndex

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILENAME = "embeddings.npy"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer once and reuse it -- loading takes
    a couple of seconds, don't repeat it per call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def build_embedding_index(chunk_index: ChunkIndex) -> NDArray[np.float32]:
    """Encode every chunk in chunk_index into an L2-normalized embedding
    vector, re-reading each chunk's exact text slice from its source
    file (grouped per file, same pattern as bm25.build_bm25_index).

    Vectors are L2-normalized so a plain dot product at query time is
    equivalent to cosine similarity, without renormalizing every score.

    Args:
        chunk_index: the ChunkIndex produced by build_index.

    Returns:
        An (n_chunks, embedding_dim) float32 array, aligned index-for-
        index with chunk_index.chunks.
    """
    chunks_by_file: dict[str, list[int]] = defaultdict(list)
    for i, source in enumerate(chunk_index.chunks):
        chunks_by_file[source.file_path].append(i)

    n_chunks = len(chunk_index.chunks)
    texts: list[str] = [""] * n_chunks

    for file_path, indices in chunks_by_file.items():
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i in indices:
            source = chunk_index.chunks[i]
            texts[i] = text[
                source.first_character_index:source.last_character_index
            ]

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return cast(NDArray[np.float32], embeddings.astype(np.float32))


def embed_query(query: str) -> NDArray[np.float32]:
    """Encode a single query into the same normalized vector space as
    build_embedding_index, so a dot product against it is cosine
    similarity.

    Args:
        query: the raw question text (no manual tokenization -- the
            sentence-transformer has its own subword tokenizer).

    Returns:
        A (embedding_dim,) float32 vector, L2-normalized.
    """
    model = _get_model()
    vector = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    return cast(NDArray[np.float32], vector.astype(np.float32))


def score_query(query: str, embeddings: NDArray[np.float32]) -> list[float]:
    """Cosine-similarity score every chunk against query.

    Args:
        query: the raw question text.
        embeddings: the array returned by build_embedding_index /
            load_embedding_index.

    Returns:
        One similarity score per chunk (range [-1, 1], almost always
        positive for real text), aligned with the embeddings rows.
    """
    if embeddings.shape[0] == 0 or not query or not query.strip():
        return [0.0] * embeddings.shape[0]
    query_vector = embed_query(query)
    return list((embeddings @ query_vector).astype(float))


def save_embedding_index(
    embeddings: NDArray[np.float32], save_directory: Path
) -> Path:
    """Persist embeddings as a binary .npy file under save_directory."""
    save_directory.mkdir(parents=True, exist_ok=True)
    out_path = save_directory / EMBEDDINGS_FILENAME
    np.save(out_path, embeddings)
    return out_path


def load_embedding_index(save_directory: Path) -> NDArray[np.float32]:
    """Load a previously persisted embedding index from save_directory."""
    path = save_directory / EMBEDDINGS_FILENAME
    return cast(NDArray[np.float32], np.load(path).astype(np.float32))
