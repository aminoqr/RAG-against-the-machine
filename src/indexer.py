"""Wires corpus discovery and the two chunking strategies together into
the 'index' command: chunk the whole corpus and persist chunk offsets."""

from pathlib import Path
from pydantic import BaseModel
from tqdm import tqdm

from src.corpus import discover_corpus_files
from src.markdown_chunker import chunk_markdown
from src.models import MinimalSource
from src.python_chunker import chunk_python


class ChunkIndex(BaseModel):
    """Persisted index: every chunk's location, nothing else -- chunk
    text is re-read from the source file on demand by offset, so we
    never store it twice."""

    max_chunk_size: int
    chunks: list[MinimalSource]


def build_index(corpus_root: Path, max_chunk_size: int) -> ChunkIndex:
    """Walk corpus_root, chunk every discovered file with the chunker
    matching its extension, and return the combined index.

    Args:
        corpus_root: root directory of the corpus to index(e.g.
            ''data/raw/vllm-0.10.1'').
        max_chunk_size: maximum characters per chunk.

    Returns:
        A ChunkIndex covering every chunk of every discovered file.
    """
    files = discover_corpus_files(corpus_root)
    chunks: list[MinimalSource] = []

    for path, kind in tqdm(files, desc="Chunking"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        chunk_fn = chunk_python if kind == "python" else chunk_markdown
        for chunk in chunk_fn(text, max_chunk_size):
            chunks.append(
                MinimalSource(
                    file_path=str(path),
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )

    return ChunkIndex(max_chunk_size=max_chunk_size, chunks=chunks)


def save_index(index: ChunkIndex, save_directory: Path) -> Path:
    """Persist index as JSON under save_directory/chunks.json.

    Args:
        index: the ChunkIndex to persist.
        save_directory: directory to write into (created if missing).

    Returns:
        The path of the written file.
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    out_path = save_directory / "chunks.json"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(index.model_dump_json(indent=2))
    return out_path


def load_index(save_directory: Path) -> ChunkIndex:
    """Load a previously persisted ChunkIndex from save_directory.

    Args:
        save_directory: directory previously passed to save_index.

    Returns:
        The loaded ChunkIndex.
    """
    path = save_directory / "chunks.json"
    with path.open("r", encoding="utf-8") as f:
        return ChunkIndex.model_validate_json(f.read())
