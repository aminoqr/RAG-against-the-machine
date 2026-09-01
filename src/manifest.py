"""Bonus: change detection for incremental indexing. Tracks each
corpus file's mtime + size across `index` runs so the next run can
tell exactly which files changed, without hashing file contents."""

from pathlib import Path

from pydantic import BaseModel

MANIFEST_FILENAME = "manifest.json"


class FileRecord(BaseModel):
    """One file's fingerprint at the time it was last indexed."""

    mtime: float
    size: int


class Manifest(BaseModel):
    """Every indexed file's fingerprint, keyed by its exact corpus path."""

    files: dict[str, FileRecord]


def build_manifest(file_paths: list[Path]) -> Manifest:
    """Snapshot the current mtime+size of every given file.

    Args:
        file_paths: files to fingerprint (the corpus's currently
            discovered, chunkable files).

    Returns:
        A Manifest covering every file that could be stat'd. A file
        that disappears between discovery and stat() (rare race) is
        silently skipped rather than crashing the whole index run.
    """
    records: dict[str, FileRecord] = {}
    for path in file_paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        records[str(path)] = FileRecord(mtime=stat.st_mtime, size=stat.st_size)
    return Manifest(files=records)


def save_manifest(manifest: Manifest, save_directory: Path) -> Path:
    """Persist manifest as JSON under save_directory/manifest.json."""
    save_directory.mkdir(parents=True, exist_ok=True)
    out_path = save_directory / MANIFEST_FILENAME
    with out_path.open("w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    return out_path


def load_manifest(save_directory: Path) -> Manifest | None:
    """Load a previously persisted Manifest, or None if there isn't
    one yet (first-ever index run)."""
    path = save_directory / MANIFEST_FILENAME
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return Manifest.model_validate_json(f.read())


def diff_manifest(
    old: Manifest | None, new: Manifest
) -> tuple[set[str], set[str], set[str]]:
    """Compare two manifests to find what changed.

    Args:
        old: the previous run's manifest, or None on a first run (in
            which case every file counts as changed/new).
        new: the current run's freshly built manifest.

    Returns:
        (changed_or_new, unchanged, removed) -- three disjoint sets of
        file paths. changed_or_new covers both genuinely new files and
        ones whose mtime or size differ from last time.
    """
    if old is None:
        return set(new.files), set(), set()

    changed_or_new: set[str] = set()
    unchanged: set[str] = set()
    for path, record in new.files.items():
        old_record = old.files.get(path)
        if old_record is None or old_record != record:
            changed_or_new.add(path)
        else:
            unchanged.add(path)

    removed = set(old.files) - set(new.files)
    return changed_or_new, unchanged, removed
