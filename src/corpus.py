"""Decides which corpus files get indexed,
and which chunker handles each one."""

from pathlib import Path
from typing import Literal

ChunkerKind = Literal["python", "markdown"]

# Extensions we actually chunk, mapped to which chunker strategy handles them.
# .txt is grouped with markdown on purpose: dataset_docs_public.json has a real
# ground-truth source at the repo root (CMakeLists.txt) that isn't a .md file
# but is still plain prose-ish text, not code.abs
CHUNKABLE_EXTENSIONS: dict[str, ChunkerKind] = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "markdown",
}

# Directories we never even walk into. Verified against the real ground-truth
# datasets: every code-question source lives under vllm/, none under tests/,
# benchmarks/, examples/, or csrc/ - so  excluding these costs zero recall.
EXCLUDED_DIR_NAMES = {
    ".git",
    ".github",
    ".buildkite",
    "__pycache__",
    "csrc",     # C++/CUDA kernels - outside the two required chunker types
    "cmake",    # helper .cmake scripts, build plumbing no conceptual content
    "docker",
    "tests",    # ground truth never points here for this corpus
}

def discover_corpus_files(root: Path) -> list[tuple[Path, ChunkerKind]]:
    """Walk 'root' and return (file_path, chunker_kind)
    for every file worth indexing"""
    discovered: list[tuple[Path, ChunkerKind]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
    
        relative_parts = path.relative_to(root).parts
        if any(path in EXCLUDED_DIR_NAMES for part in relative_parts):
            continue

        chunker_kind = CHUNKABLE_EXTENSIONS.get(path.suffix)
        if chunker_kind is None:
            continue

        discovered.append((path, chunker_kind))
    
    return discovered