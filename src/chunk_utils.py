"""Shared low-level splitting helpers used by every chunking strategy:
a generic greedy span packer, plus the raw fallbacks (line-aware and
hard character splits) used when a single unit is too big to fit."""

from typing import Callable

from src.chunk import Chunk

SplitOversized = Callable[[str, int, int, int], "list[Chunk]"]


def hard_split(
    text: str, start: int, end: int, max_chunk_size: int
) -> list[Chunk]:
    """Last-resort fallback: cut text[start:end] every max_chunk_size
    characters, with no attempt at boundary awareness."""
    chunks: list[Chunk] = []
    pos = start
    while pos < end:
        chunk_end = min(pos + max_chunk_size, end)
        chunks.append(
            Chunk(
                text=text[pos:chunk_end],
                first_character_index=pos,
                last_character_index=chunk_end,
            )
        )
        pos = chunk_end
    return chunks


def split_by_lines(
    text: str, start: int, end: int, max_chunk_size: int
) -> list[Chunk]:
    """A unit longer than max_chunk_size: pack whole lines of
    text[start:end] together, hard-splitting any single line that alone
    exceeds the limit."""
    chunks: list[Chunk] = []
    cur_start = start
    cur_end = start
    pos = start
    while pos < end:
        newline = text.find("\n", pos, end)
        line_end = newline + 1 if newline != -1 else end
        line_len = line_end - pos

        if line_len > max_chunk_size:
            if cur_end > cur_start:
                chunks.append(
                    Chunk(
                        text=text[cur_start:cur_end],
                        first_character_index=cur_start,
                        last_character_index=cur_end,
                    )
                )
            chunks.extend(hard_split(text, pos, line_end, max_chunk_size))
            cur_start = line_end
            cur_end = line_end
        elif cur_end - cur_start + line_len > max_chunk_size:
            chunks.append(
                Chunk(
                    text=text[cur_start:cur_end],
                    first_character_index=cur_start,
                    last_character_index=cur_end,
                )
            )
            cur_start = pos
            cur_end = line_end
        else:
            cur_end = line_end
        pos = line_end

    if cur_end > cur_start:
        chunks.append(
            Chunk(
                text=text[cur_start:cur_end],
                first_character_index=cur_start,
                last_character_index=cur_end,
            )
        )
    return chunks


def _flush(
    text: str,
    start: int,
    end: int,
    max_chunk_size: int,
    split_oversized: SplitOversized,
) -> list[Chunk]:
    """Turn one accumulated group of spans into one or more Chunks."""
    if end - start <= max_chunk_size:
        return [
            Chunk(
                text=text[start:end],
                first_character_index=start,
                last_character_index=end,
            )
        ]
    return split_oversized(text, start, end, max_chunk_size)


def pack_spans(
    text: str,
    spans: list[tuple[int, int]],
    max_chunk_size: int,
    split_oversized: SplitOversized,
) -> list[Chunk]:
    """Greedily merge contiguous (start, end) spans into Chunks no
    larger than max_chunk_size. When even a single span alone exceeds
    the limit, delegate to split_oversized(text, start, end,
    max_chunk_size) to decide how to break it down further.

    This is the shared packing mechanism behind both chunking
    strategies: only what counts as a "span" differs (paragraphs for
    markdown, top-level statements/class members for Python).
    """
    if not spans:
        return []

    chunks: list[Chunk] = []
    group_start, group_end = spans[0]

    for span_start, span_end in spans[1:]:
        if span_end - group_start <= max_chunk_size:
            group_end = span_end
            continue
        chunks.extend(
            _flush(text, group_start, group_end, max_chunk_size,
                   split_oversized)
        )
        group_start, group_end = span_start, span_end

    chunks.extend(
        _flush(text, group_start, group_end, max_chunk_size,
               split_oversized)
    )
    return chunks
