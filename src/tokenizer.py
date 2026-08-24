"""Shared tokenization used at both index time and query time -- must
stay identical on both sides, or BM25 scores silently go to zero for
terms that no longer line up."""

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric characters, and additionally
    emit sub-word tokens for snake_case/camelCase identifiers(e.g.
    'add_result' also yields 'add' and 'result') so a question using
    a plain English word can still match a chunk where that word only
    appears inside a compund identifier.

    Args:
        text: raw text to tokenize (source code or prose).

    Returns:
        A list of lowercase tokens, in order, possibly with repeats.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tokens.append(raw.lower())
        parts = [
            p for p in _CAMEL_BOUNDARY_RE.sub("_", raw).split("_") if p
        ]
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens
