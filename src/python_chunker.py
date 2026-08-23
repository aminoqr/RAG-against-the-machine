"""Python code chunking strategy: split on syntax-aware boundaries
(top-level functions/classes, falling back to a class's own methods)
rather than blind character counts, since a function is a coherent
unit of meaning in a way that 'the next N characters' isn't."""

import ast

from src.chunk import Chunk
from src.chunk_utils import pack_spans, split_by_lines


def _line_starts(text: str) -> list[int]:
    """offsets[i] = character offset where line (i + 1) starts;
    offsets[-1] always equals len(text)."""
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_span(
    node: ast.stmt, offsets: list[int], text_len: int
) -> tuple[int, int]:
    """Character span of a top-level node, including its decorators
    (a decorated function/class's lineno points at the def/class
    keyword, not the decorator above it)."""
    start_lineno = node.lineno
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        start_lineno = min(start_lineno, min(d.lineno for d in decorators))
    start = offsets[start_lineno - 1]

    end_lineno = node.end_lineno
    end = offsets[end_lineno] if end_lineno is not None else text_len
    return start, end


def _split_large_class(
    text: str,
    class_node: ast.ClassDef,
    offsets: list[int],
    text_len: int,
    max_chunk_size: int,
) -> list[Chunk]:
    """A class too large to fit in one chunk: pack its own members
    (methods, nested statements) instead of falling straight to a raw
    line split."""
    if not class_node.body:
        start, end = _node_span(class_node, offsets, text_len)
        return split_by_lines(text, start, end, max_chunk_size)

    member_spans = [
        _node_span(member, offsets, text_len) for member in class_node.body
    ]
    return pack_spans(text, member_spans, max_chunk_size, split_by_lines)


def chunk_python(text: str, max_chunk_size: int = 2000) -> list[Chunk]:
    """Chunk Python source into pieces no larger than max_chunk_size
    characters, grouping whole top-level statements (imports,
    functions, classes) together where they fit, and recursing one
    level into a class's methods when the class alone is too large.

    Falls back to a line-aware split of the whole file if the source
    doesn't parse (e.g. a syntax error, or a .py file that isn't
    actually valid Python) so indexing never crashes on one bad file.

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

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return split_by_lines(text, 0, len(text), max_chunk_size)

    offsets = _line_starts(text)
    text_len = len(text)
    top_level_spans = [_node_span(n, offsets, text_len) for n in tree.body]
    if not top_level_spans:
        return []

    span_to_node = dict(zip(top_level_spans, tree.body))

    def split_oversized(
        t: str, start: int, end: int, size: int
    ) -> list[Chunk]:
        node = span_to_node.get((start, end))
        if isinstance(node, ast.ClassDef):
            return _split_large_class(t, node, offsets, text_len, size)
        return split_by_lines(t, start, end, size)

    return pack_spans(text, top_level_spans, max_chunk_size, split_oversized)
