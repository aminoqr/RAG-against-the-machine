"""Markdown/text chunking strategy: split content into pieces no larger
than max_chunk_size characters, preferring paragraph boundaries over
mid-sentence/mid-line cuts."""

from src.chunk import Chunk
from src.chunk_utils import hard_split, pack_spans, split_by_lines


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of each non-blank paragraph in text,
    where paragraphs are separated by one or more blank lines."""
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    i = 0
    while i < length:
        if text[i] == "\n" and i + 1 < length and text[i + 1] == "\n":
            j = i
            while j < length and text[j] == "\n":
                j += 1
            if text[start:i].strip():
                spans.append((start, i))
            start = j
            i = j
        else:
            i += 1
    if text[start:length].strip():
        spans.append((start, length))
    return spans


def chunk_markdown(text: str, max_chunk_size: int = 2000) -> list[Chunk]:
    """Chunk markdown/text content into pieces no larger than
    max_chunk_size characters, preferring paragraph boundaries so a
    heading isn't split from the content immediately following it.

    Args:
        text: the full file content to chunk.
        max_chunk_size: the maximum number of characters per chunk.

    Returns:
        A list of Chunks covering the whole file, in order, with exact
        character offsets relative to the start of ``text``.

    Raises:
        ValueError: if max_chunk_size is not a positive integer.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if not text or not text.strip():
        return []

    spans = _paragraph_spans(text)
    if not spans:
        return hard_split(text, 0, len(text), max_chunk_size)

    return pack_spans(text, spans, max_chunk_size, split_by_lines)
