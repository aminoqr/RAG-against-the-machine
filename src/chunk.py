"""Shared chunk representation used by every chunking strategy."""

from dataclasses import dataclass


@dataclass
class Chunk:
    """A contiguous slice of a source file, with its exact character span
    relative to the start of that file."""

    text: str
    first_character_index: int
    last_character_index: int
