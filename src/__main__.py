"""Command-line interface for the RAG against the machine project."""
import fire
from pathlib import Path
from src.indexer import build_index, save_index
from src.bm25 import build_bm25_index, save_bm25_index
from src.retriever import search as run_search
from src.retriever import search_dataset as run_search_dataset
from src.evaluator import evaluate as run_evaluate


class Cli:
    """Expose the project's commands as CLI subcommands via Python Fire."""
    def index(self, max_chunk_size: int = 2000) -> None:
        """Chunk the whole corpus, fit a BM25 index, and persist both.

        Args:
            max_chunk_size: maximum number of characters per chunk.
        """
        corpus_root = Path("data/raw/vllm-0.10.1")
        try:
            chunk_index = build_index(corpus_root, max_chunk_size)
        except ValueError as e:
            print(f"Indexing failed: {e}")
            return
        
        chunks_path = save_index(chunk_index, Path("data/processed"))
        bm25_index = build_bm25_index(chunk_index)
        bm25_path = save_bm25_index(bm25_index, Path("data/processed"))

        print(
            f"Indexed {len(chunk_index.chunks)} chunks "
            f"({len(bm25_index.document_frequency)} unique terms) "
            f"into {chunks_path} and {bm25_path}"
        )

    def search(self, query: str, k: int = 10) -> None:
        """Return the top-k sources for a single query.
        
        Args:
            query: the question to search for.
            k: maximum number of results to return.
        """
        
        try:
            results = run_search(query, k, Path("data/processed"))
        except FileNotFoundError:
            print(
                "No index found under data/processed/ -- run "
                "'uv run python -m src index' first."
            )
            return
        
        if not results:
            print("No results.")
            return
        
        for source in results:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

    def search_dataset(
        self, dataset_path: str, k: int = 10, save_directory: str | None = None
    ) -> None:
        """Run search over a whole dataset and write a
        StudentSearchResults JSON file, scoped under save_directory.
        
        Args:
            dataset_path: path to an UnansweredQuestions/
                AnsweredQUestions JSON file.
            k: max number of tesults to return per question.
            save_directory: directory to write the output JSON into.
        """
        if not save_directory:
            print("save_directory is required.")
            return
    
        try:
            out_path = run_search_dataset(
                Path(dataset_path),
                k,
                Path(save_directory),
                Path("data/processed"),
            )
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return
        except ValueError as e:
            print(f"Invalid dataset: {e}")
            return
        
        print(f"Saved student_search_results to {out_path}")
        

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
        """Report your own recall@k against a ground-truth dataset.
        
        Args:
            student_search_results_path: path to a StudentSearchResults
                JSON file (search_dataset's output).
            dataset_path: path to the matching ground-truth
                AnsweredQuestions dataset.
        """
        try:
            recall = run_evaluate(
                Path(student_search_results_path), Path(dataset_path)
            )
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return
        except ValueError as e:
            print(f"Invalid input: {e}")
            return
        
        if all(v == 0.0 for v in recall.values()):
            print(
                "Warning: every recall@k is 0.0 -- double check "
                "student_search_resi;ts_dath and dataset_path actually "
                "refer to the same set of questions."
            )
        
        print("Evaluation Results")
        print("=" * 40)
        print(" ".join(f"Recall@{k}: {recall[k]:.3f}" for k in sorted(recall)))


if __name__ == "__main__":
    fire.Fire(Cli)
