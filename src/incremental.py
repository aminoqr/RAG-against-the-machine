"""Bonus: incremental indexing. Re-chunk and re-tokenize only the
files that changed since the last `index` run, instead of the whole
corpus every time.

Why this is genuinely harder for BM25 than for a per-chunk store:
document_frequency and avg_chunk_length are aggregates over *every*
chunk in the corpus, so they still need recomputing whenever anything
changes. What incremental indexing actually buys you is skipping the
expensive part -- walking the corpus, reading, chunking and
tokenizing unchanged files -- by reusing their already-tokenized
per-chunk term frequencies straight from the previous run's persisted
Bm25Index. Recomputing the aggregates from that already-in-memory data
is a cheap O(chunks) pass, no file I/O involved.
"""

from collections import Counter
from pathlib import Path

from src.bm25 import B, K1, Bm25Index, load_bm25_index
from src.corpus import ChunkerKind, discover_corpus_files
from src.indexer import ChunkIndex, load_index
from src.manifest import (
    build_manifest,
    diff_manifest,
    load_manifest,
    save_manifest,
)
from src.markdown_chunker import chunk_markdown
from src.models import MinimalSource
from src.python_chunker import chunk_python
from src.tokenizer import tokenize


def build_index_incremental(
    corpus_root: Path, max_chunk_size: int, processed_dir: Path
) -> tuple[ChunkIndex, Bm25Index, set[str]]:
    """Re-index only the files that changed since the last run under
    processed_dir, reusing every other chunk's data untouched.

    Args:
        corpus_root: root directory of the corpus to index.
        max_chunk_size: maximum characters per chunk.
        processed_dir: directory holding the previous run's index (if
            any); also where the new manifest gets persisted.

    Returns:
        (chunk_index, bm25_index, reprocessed_files) -- the merged,
        ready-to-save indexes, plus which file paths were actually
        re-chunked this run (empty on a no-op re-run: nothing changed).
    """
    files = discover_corpus_files(corpus_root)
    kind_by_path: dict[str, ChunkerKind] = {
        str(path): kind for path, kind in files
    }

    new_manifest = build_manifest([path for path, _ in files])
    old_manifest = load_manifest(processed_dir)
    changed_or_new, unchanged, _removed = diff_manifest(
        old_manifest, new_manifest
    )

    try:
        old_chunk_index = load_index(processed_dir)
        old_bm25 = load_bm25_index(processed_dir)
    except FileNotFoundError:
        old_chunk_index = None
        old_bm25 = None

    kept_sources: list[MinimalSource] = []
    kept_tfs: list[dict[str, int]] = []
    if old_chunk_index is not None and old_bm25 is not None:
        for source, tf in zip(
            old_chunk_index.chunks, old_bm25.chunk_term_frequencies
        ):
            if source.file_path in unchanged:
                kept_sources.append(source)
                kept_tfs.append(tf)

    new_sources: list[MinimalSource] = []
    new_tfs: list[dict[str, int]] = []
    for file_path_str in changed_or_new:
        kind = kind_by_path.get(file_path_str)
        if kind is None:
            continue
        try:
            text = Path(file_path_str).read_text(
                encoding="utf-8", errors="ignore"
            )
        except OSError:
            continue

        chunk_fn = chunk_python if kind == "python" else chunk_markdown
        path_tokens = tokenize(file_path_str)
        for chunk in chunk_fn(text, max_chunk_size):
            new_sources.append(
                MinimalSource(
                    file_path=file_path_str,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )
            tf = Counter(tokenize(chunk.text))
            tf.update(path_tokens)
            new_tfs.append(dict(tf))

    all_sources = kept_sources + new_sources
    all_tfs = kept_tfs + new_tfs

    document_frequency: Counter[str] = Counter()
    total_length = 0
    for tf in all_tfs:
        document_frequency.update(tf.keys())
        total_length += sum(tf.values())

    n_chunks = len(all_sources)
    avg_chunk_length = total_length / n_chunks if n_chunks else 0.0

    chunk_index = ChunkIndex(max_chunk_size=max_chunk_size, chunks=all_sources)
    bm25_index = Bm25Index(
        k1=K1,
        b=B,
        n_chunks=n_chunks,
        avg_chunk_length=avg_chunk_length,
        document_frequency=dict(document_frequency),
        chunk_term_frequencies=all_tfs,
    )

    save_manifest(new_manifest, processed_dir)

    return chunk_index, bm25_index, changed_or_new
