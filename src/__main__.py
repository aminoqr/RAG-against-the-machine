"""Command-line interface for the RAG against the machine project."""
import fire
from pathlib import Path
from src.indexer import build_index, save_index


class Cli:
    """Expose the project's commands as CLI subcommands via Python Fire."""
    def index(self, max_chunk_size: int = 2000) -> None:
        """Chunk the whole corpus and persist the resulting index.

        Args:
            max_chunk_size: maximum number of characters per chunk.
        """
        corpus_root = Path("data/raw/vllm-0.10.1")
        idx = build_index(corpus_root, max_chunk_size)
        out_path = save_index(idx, Path("data/processed"))
        print(f"Indexed {len(idx.chunks)} chunks from corpus into {out_path}")

    def search(self, query: str, k: int = 10) -> None:
        print(f"search called with query={query!r}, k={k}")

    def search_dataset(
        self, dataset_path: str, k: int = 10, save_directory: str = None
    ) -> None:
        print(
            f"search_dataset called with dataset_path={dataset_path!r}, "
            f"k={k}, save_directory={save_directory!r}"
        )

    def answer(self, query: str, k: int = 10) -> None:
        print(f"answer called with query={query!r}, k={k}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = ""
    ) -> None:
        print(
            f"answer_dataset called with "
            f"student_search_results_path={student_search_results_path!r}, "
            f"save_directory={save_directory!r}"
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        print(
            f"evaluate called with "
            f"student_search_results_path={student_search_results_path!r} "
            f"dataset_path={dataset_path!r}"
        )


if __name__ == "__main__":
    fire.Fire(Cli)
