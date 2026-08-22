"""Markdown/text chunking strategy: split content into pieces no larger
than max_chunk_size characters, preferring paragraph boundaries over
mid-sentence/mid-line cuts."""

from src.chunk import Chunk


def _paragraph_spans(text:str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of each non-blank parahgraph in text,
    where paragraphs are separated by one or more blank lines."""
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    i = 0
    while i < length:
        if text[i] == "\n" and i + 1 < length and text[i+1] == "\n":
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

def _hard_split(
    text: str, start_offset: int, max_chunk_size: int
) -> list[Chunk]:
    """Last-resort fallback: cut text every max_chunk_size characters,
    with no attempt at boundary awareness."""
    
    chunks: list[Chunk] = []
    n = len(text)
    pos = 0
    while pos < n:
        end = min(pos + max_chunk_size, n)
        chunks.append(
            Chunk(
                text=text[pos:end],
                first_character_index=start_offset + pos,
                last_character_index=start_offset + end,
            )
        )
        pos = end
    return chunks

def _split_by_lines(
    text: str, start_offset: int, max_chunk_size: int
) -> list[Chunk]:
    """A paragraph longer than max-chunk-size: pack whole lines together,
    and hard-split any single line that alone sceeds the limit."""
    
    chunks: list[Chunk] = []
    cur_start = 0
    cur_end = 0
    pos = 0
    n = len(text)
    while pos < n:
        newline = text.find("\n", pos)
        line_end = newline + 1 if newline != -1 else n
        line_len = line_end - pos
        
        if line_len > max_chunk_size:
            if cur_end > cur_start:
                chunks.append(
                    Chunk(
                        text=text[cur_start:cur_end],
                        first_character_index=start_offset + cur_start,
                        last_character_index=start_offset + cur_end,
                    )
                )
            chunks.extend(
                _hard_split(
                    text[pos:line_end], start_offset + pos, max_chunk_size
                )
            )
            cur_start = line_end
            cur_end = line_end
        elif cur_end - cur_start + line_len > max_chunk_size:
            chunks.append(
                Chunk(
                    text=text[cur_start:cur_end],
                    first_character_index=start_offset + cur_start,
                    last_character_index=start_offset + cur_end,
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
                first_character_index=start_offset + cur_start,
                last_character_index=start_offset + cur_end,
            )
        )
    return chunks

def _flush_group(
    text: str, start: int, end: int, max_chunk_size: int
) -> list[Chunk]:
    """Turn one accumulated group of paragraphs into one or more Chunks."""
    if end - start <= max_chunk_size:
        return [
            Chunk(
                text=text[start:end],
                first_character_index=start,
                last_character_index=end,
            )
        ]
    return _split_by_lines(text[start:end], start, max_chunk_size)

def chunk_markdown(text: str, max_chunk_size: int = 2000) -> list[Chunk]:
    """Chunk markdwon/text content into pieces no parger than
    max_chunk_size characters, preferring paragraph boundaries so a 
    heading isn't split form the content immediately following it.
    
    Args:
        text: the full file content to chunk.
        max_chunk_size: the maximum number of characters per chunk.
    
    Returns:
        A list of Chunks covering the whole file, in order, with exact
        character offsets relative to the start of ''text''.
    
    Raises:
        ValueError: if max_chunk_size is not a positive integer.
    """
    
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if not text or not text.strip():
        return []

    spans = _paragraph_spans(text)
    if not spans:
        return _hard_split(text, 0, max_chunk_size)
    
    chunks: list[Chunk] = []
    group_start, group_end = spans[0]
    
    for span_Start, span_end in spans[1:]:
        if span_end - group_start <= max_chunk_size:
            group_end = span_end
            continue
        chunks.extend(
            _flush_group(text, group_start, group_end, max_chunk_size)
        )
        group_start, group_end = span_Start, span_end
    
    chunks.extend(_flush_group(text, group_start, group_end, max_chunk_size))
    return chunks
