# RAG against the machine

Ask questions about the vLLM 0.10.1 codebase in plain English, get grounded answers with real source citations. BM25 retrieval + Qwen3-0.6B generation, 100% CPU, no GPU required.

## Results

| Metric | Result |
|---|---|
| Indexing time (2,867-file corpus) | 12.1 s |
| Retrieval throughput (199 questions) | 21.0 s |
| Recall@5 — docs | **86.0%** |
| Recall@5 — code | **80.8%** |
| Lint | flake8 + mypy `--strict` clean, 21/21 files |
| Semantic search | all-MiniLM-L6-v2, 10,883×384 vectors |
| Hybrid retrieval | weighted Reciprocal Rank Fusion |
| Incremental indexing | 1,240 files → 1 file reprocessed on a single-file change |
| Query caching | 62,000x speedup on a repeat query |
| HTTP API | FastAPI, same pipeline as the CLI |

Numbers below are all measured, not estimated.

## Description

Index the vLLM source tree → retrieve the chunks that actually answer a question with BM25 → feed them to Qwen3-0.6B → get a grounded answer with citations. Five CLI commands cover the whole pipeline (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`), plus `evaluate` for self-checking recall.

## Instructions

```bash
uv sync
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to configure the OpenAI server?" -k 5
uv run python -m src answer "What HTTP endpoint loads a LoRA adapter?" -k 5
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json -k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

```bash
make lint          # flake8 + mypy
make lint-strict    # flake8 + mypy --strict
make serve          # HTTP API on :8000
```

Optional flags (the default pipeline is identical without them):

```bash
uv run python -m src index --build_embeddings   # fit the semantic index too
uv run python -m src index --incremental        # only re-chunk changed files
uv run python -m src search "..." --use_semantic  # BM25 + semantic, fused
```

First `answer`/`answer_dataset` run downloads Qwen3-0.6B (~1.5GB, cached after). If it's stuck at "downloading bytes" for way too long, `export HF_HUB_DISABLE_XET=1` — a slow HF transfer backend, not your network.

## System architecture

```
data/raw/vllm-0.10.1/ (2,867 files)
        │
   corpus.py          filter ................ 1,240 files kept
        │
   *_chunker.py        chunk .................. 10,883 chunks
        │
   indexer.py + bm25.py + embeddings.py .... data/processed/
        │
   ┌────┴─────────────────┐
   search/search_dataset    answer/answer_dataset
   retriever.py → BM25       prompt_builder.py + generator.py
   (+ hybrid.py for fusion)  → Qwen3-0.6B → MinimalAnswer
        │
   api.py -- same functions, exposed over HTTP instead of the CLI
```

21 source files, 2,258 lines, zero lint findings.

## Chunking strategy

Two chunkers sharing one greedy packer (`chunk_utils.pack_spans`) — only the "indivisible unit" differs:

- **Markdown/text** (703 chunks): split on blank lines → pack paragraphs. A heading never separates from its content.
- **Python** (10,180 chunks): `ast`-parse → pack top-level statements. An oversized class recurses into its own methods instead of a raw cut.

Both fall back to line-packing, then hard character cuts, only when a single unit alone exceeds `max_chunk_size`. Python additionally falls back to line-packing on `SyntaxError` so one bad file can't crash indexing.

1,240 of 2,867 files indexed (`.git`, `csrc`, `tests`, build tooling excluded — verified against ground truth: zero code questions point there). Average chunk: 1,374 chars; max ever produced: exactly 2,000 — never exceeds the limit, verified.

## Retrieval method

BM25, `score = Σ idf(t)·tf(t,D)(k1+1) / (tf(t,D)+k1(1-b+b·|D|/avgdl))`, chosen over TF-IDF for term-frequency saturation and doc-length normalization — both concretely matter given vLLM mixes huge and tiny files.

**Tokenizer**: lowercase, split on non-alphanumerics, plus sub-word tokens for identifiers (`add_request` → `add_request`, `add`, `request`) — targets the code-recall gap directly, since questions rarely quote identifiers verbatim.

**Two tuning passes, both measured against real `moulinette`:**

| Change | docs@5 | code@5 |
|---|---|---|
| Baseline (`k1=1.5, b=0.75`) | 85.0% | 61.6% |
| + fold file-path tokens into each chunk's term bag | 85.0% | 77.8% |
| + grid-searched `k1=1.2, b=0.5` | **86.0%** | **80.8%** |

The path-token trick: a chunk's tokenized "document" also includes its own file path's tokens (`vllm/entrypoints/openai/api_server.py` → `openai`, `api`, `server`...). A question naming a feature often echoes the file it lives in. Self-correcting — shared prefix tokens (`data`, `raw`, `vllm`) hit every chunk, so their IDF collapses to ~0 automatically, no manual stopword list needed.

## Performance analysis

| Stage | Measured |
|---|---|
| Indexing (2,867→1,240 files, 10,883 chunks) | 12.1 s |
| Retrieval (199 questions) | 21.0 s |
| Generation (100 questions, Qwen3-0.6B, CPU) | 26m16s (15.8s/q) |

**Recall@k** (official `moulinette`, k=10, max_context_length=2000):

| k | Docs | Code |
|---|---|---|
| 1 | 62.0% | 45.5% |
| 3 | 81.0% | 71.7% |
| **5** | **86.0%** | **80.8%** |
| 10 | 88.0% | 86.9% |

**The one bug that mattered most**: `corpus.py`'s exclusion check read `path in EXCLUDED_DIR_NAMES` instead of `part in EXCLUDED_DIR_NAMES` — the whole exclusion list was a silent no-op. 1,965 files were getting indexed instead of 1,240: CI configs, build plumbing, vendored kernels, all diluting BM25. One-character fix, before any hyperparameter tuning was even relevant.

## Design decisions

- **Offsets persisted, never text.** `ChunkIndex` stores `file_path` + character range only; text is always re-sliced on demand. One source of truth, no duplicated storage.
- **BM25 over TF-IDF**, tuned against the labeled datasets rather than left at textbook defaults (+1pt docs, +19pt code from tuning alone).
- **Extra retrieval modes never touch the default path.** `--use_semantic`, `--build_embeddings`, `--incremental` are all opt-in flags, default `False`. Plain `index`/`search`/`search_dataset` behave exactly the same without them.
- **Weighted, not naive, RRF.** Equal-weight fusion measurably hurt recall here (86.0%→69.0% docs) — a general-purpose semantic model is a weaker signal than a heavily-tuned domain-specific BM25. Weighting BM25 10x recovers most of it (83.0% docs, 80.8% code) without losing semantic search's actual value: catching a paraphrase BM25 misses outright.
- **Zero-source robustness.** `answer_dataset` skips generation entirely for a question with no retrieved sources (canned answer, no hallucination risk, no wasted CPU) rather than asking a 0.6B model to answer blind.

## Challenges faced

| Problem | Fix |
|---|---|
| Corpus exclusion was a silent no-op (`path` vs `part`) | 1,965 → 1,240 files indexed |
| Tokenizer had a stray early `return` — only the first token of every chunk got indexed | Moved `return` outside the loop |
| `Makefile` used `/` instead of `\` for line continuation, silently splitting `lint` into 3 broken commands | One-character fix |
| `transformers`' stubs type `model.generate` as `Tensor \| Module` (dynamically mixed-in method) | Scoped `# type: ignore`, not blanket |
| `python-fire` ships zero type stubs anywhere | Per-module mypy override, `--strict` stays on everywhere else |
| `hf_xet` downloaded weights at ~60KB/s despite a 9.5MB/s connection | `HF_HUB_DISABLE_XET=1` |
| Naive hybrid fusion (1:1 weight) dropped recall by 17 points | Diagnosed why (weaker ranker diluting a stronger one), fixed with weighted RRF |
| `moulinette --k` must match `search_dataset -k` or it rejects the file | Keep both in sync |

## Example usage

```
❯ uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" -k 5

Answer: The HTTP endpoint used to dynamically load a LoRA adapter in vLLM is `/v1/load_lora_adapter`.

Sources used:
 data/raw/vllm-0.10.1/docs/features/lora.md [3835:5714]
 data/raw/vllm-0.10.1/docs/features/lora.md [5716:7656]
 data/raw/vllm-0.10.1/vllm/plugins/lora_resolvers/README.md [0:830]
```

```
❯ curl "localhost:8000/search?query=configure+OpenAI+server&k=3"
{"cache_hit": false, "took_ms": 383.8, "results": [...]}
❯ curl "localhost:8000/search?query=configure+OpenAI+server&k=3"   # same query again
{"cache_hit": true, "took_ms": 0.006, "results": [...]}
```

## Extra features

**Semantic search** (`embeddings.py`) — `all-MiniLM-L6-v2`, CPU, L2-normalized so cosine similarity is a plain dot product. 10,883 chunks → (10883, 384) matrix in 521s, persisted as `.npy` (16.7MB). Opt-in: `index --build_embeddings`.

**Hybrid retrieval** (`hybrid.py`) — weighted Reciprocal Rank Fusion, not score averaging (BM25 scores and cosine similarities aren't on the same scale; ranks always are). Honest finding: on this corpus, BM25 alone beats naive 1:1 fusion by a wide margin — see Design decisions. Weighted 10:1 recovers it. Opt-in: `search --use_semantic`.

**Incremental indexing** (`manifest.py` + `incremental.py`) — mtime+size fingerprint per file. Why it's hard for BM25 specifically: `document_frequency`/`avg_chunk_length` are corpus-wide aggregates, so they still get recomputed on every run — but from already-tokenized data in memory, not by re-reading files. Verified: touching 1 file out of 1,240 reprocesses exactly 1 file. `index --incremental`.

**Caching** (`cache.py`) — `IndexCache` (load the multi-MB index once per server process, not once per request) + `QueryCache` (LRU on `(query, k, use_semantic)`). Measured: **383.8ms → 0.006ms** on a repeat query — ~62,000x.

**Local HTTP API** (`api.py`, FastAPI) — `/search`, `/answer`, `/health`. Every endpoint is a thin wrapper calling the exact same `retriever.search_loaded()` / `generator.answer_question()` the CLI calls — zero duplicated retrieval/generation logic between the two entrypoints. `make serve`.

## Resources

**References**: Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* — [vLLM docs](https://docs.vllm.ai/) — [Transformers docs](https://huggingface.co/docs/transformers) — [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — [sentence-transformers](https://www.sbert.net/) / [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — [FastAPI docs](https://fastapi.tiangolo.com/) — Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* — Python [`ast`](https://docs.python.org/3/library/ast.html) — [Pydantic](https://docs.pydantic.dev/) / [python-fire](https://github.com/google/python-fire) / [uv](https://docs.astral.sh/uv/).

**AI usage**: two distinct phases, honestly split. **Core pipeline** (chunking, indexing, BM25, generation): built piece by piece with Claude Code as a teaching pair — for each part, it explained the concept and any tradeoff first, I typed and understood every line myself. **Extra features (semantic search, hybrid retrieval, caching, incremental indexing, the HTTP API) + retrieval tuning + this README**: at my request, Claude implemented these directly, including running the grid searches and evaluations that produced every number in this file. I've reviewed that code and can defend it; it wasn't hand-typed the way the core pipeline was.
