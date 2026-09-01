"""Bonus: caching. Two independent layers:

- IndexCache: loads ChunkIndex/Bm25Index/embeddings once per process
  and hands back the same in-memory objects afterward. Matters for a
  long-running process like the HTTP API (search()/search_dataset()
  already avoid repeated loads within a single call, but a server
  handling many separate requests needs this across calls).
- QueryCache: a small LRU mapping (query, k, use_semantic) -> results,
  so a repeated identical query skips scoring every chunk again.
  Invalidated wholesale on process restart -- a fresh `index` run
  means a fresh process, so there's no risk of ever serving a stale
  result from a rebuilt index.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.bm25 import Bm25Index, load_bm25_index
from src.embeddings import load_embedding_index
from src.indexer import ChunkIndex, load_index
from src.models import MinimalSource

QUERY_CACHE_MAX_SIZE = 256
QueryKey = tuple[str, int, bool]


class IndexCache:
    """Lazily loads and memoizes the on-disk index for one process."""

    def __init__(self, processed_dir: Path) -> None:
        self._processed_dir = processed_dir
        self._chunk_index: ChunkIndex | None = None
        self._bm25_index: Bm25Index | None = None
        self._embeddings: NDArray[np.float32] | None = None

    def chunk_index(self) -> ChunkIndex:
        """The ChunkIndex, loaded from disk once and reused."""
        if self._chunk_index is None:
            self._chunk_index = load_index(self._processed_dir)
        return self._chunk_index

    def bm25_index(self) -> Bm25Index:
        """The Bm25Index, loaded from disk once and reused."""
        if self._bm25_index is None:
            self._bm25_index = load_bm25_index(self._processed_dir)
        return self._bm25_index

    def embeddings(self) -> NDArray[np.float32]:
        """The semantic embedding matrix, loaded from disk once and
        reused (bonus: only needed when hybrid search is requested)."""
        if self._embeddings is None:
            self._embeddings = load_embedding_index(self._processed_dir)
        return self._embeddings


class QueryCache:
    """An LRU cache of (query, k, use_semantic) -> ranked results."""

    def __init__(self, max_size: int = QUERY_CACHE_MAX_SIZE) -> None:
        self._max_size = max_size
        self._store: "OrderedDict[QueryKey, list[MinimalSource]]" = (
            OrderedDict()
        )
        self.hits = 0
        self.misses = 0

    def get(
        self, query: str, k: int, use_semantic: bool
    ) -> list[MinimalSource] | None:
        """Return the cached results for this exact (query, k,
        use_semantic), or None on a cache miss."""
        key = (query, k, use_semantic)
        if key not in self._store:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return self._store[key]

    def set(
        self,
        query: str,
        k: int,
        use_semantic: bool,
        results: list[MinimalSource],
    ) -> None:
        """Store results for (query, k, use_semantic), evicting the
        least-recently-used entry once max_size is exceeded."""
        key = (query, k, use_semantic)
        self._store[key] = results
        self._store.move_to_end(key)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)
