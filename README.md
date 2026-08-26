*This project has been created as part of the 42 curriculum by aasylbye.*

# RAG against the machine

A Retrieval-Augmented Generation system that answers natural-language questions about the **vLLM 0.10.1** codebase — index it, retrieve the right snippets with BM25, and generate grounded answers with **Qwen3-0.6B**, entirely CPU-only.

## Results at a glance

| Metric | Result | Requirement | Margin |
|---|---|---|---|
| Indexing time (2,867-file corpus) | **9.6 s** | ≤ 5 min | 31× |
| Retrieval throughput (199 questions) | **16.2 s** | ≤ 90 s / 200 q | 5.6× |
| Recall@5 — docs | **85.0 %** | ≥ 80 % | +5 pts |
| Recall@5 — code | **61.6 %** | ≥ 50 % | +11.6 pts |
| `make lint` / `make lint-strict` | **clean** | must pass | flake8 + mypy `--strict`, 15/15 files |

Full breakdown in [Performance analysis](#performance-analysis).

## Description

vLLM is a large, real codebase (2,867 files, mixed Python + Markdown). The goal: given a question like *"What HTTP endpoint is used to dynamically load a LoRA adapter?"*, find the exact source locations that answer it and generate a correct, grounded, natural-language answer — without ever hard-coding anything about the questions themselves.

The pipeline has four stages, each its own CLI command: **index** the corpus into offset-tracked chunks and a fitted BM25 index → **search** it for the top-k relevant chunks → **augment** a prompt with the real text at those offsets → **generate** an answer with Qwen3-0.6B. Retrieval quality is measured with Recall@k against a held-out ground-truth dataset.

## Instructions

```bash
uv sync                                          # install all dependencies

# 1. Build the index (chunks + BM25) under data/processed/
uv run python -m src index --max_chunk_size 2000

# 2. Search
uv run python -m src search "How to configure the OpenAI server?" -k 5

# 3. Batch search a dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    -k 10 --save_directory data/output/search_results/UnansweredQuestions

# 4. Generate a single grounded answer
uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter?" -k 5

# 5. Batch-generate answers from existing search results
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Your own quick recall check (the official score comes from ./moulinette, run separately)
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

```bash
make lint          # flake8 + mypy (required flags)
make lint-strict    # flake8 + mypy --strict (optional, also clean on this repo)
make clean          # remove caches
make debug          # run under pdb
```

First run of `answer`/`answer_dataset` downloads Qwen3-0.6B (~1.5 GB) from Hugging Face and caches it under `~/.cache/huggingface/` — one-time cost, not repeated after. If that download is unexpectedly slow, `export HF_HUB_DISABLE_XET=1` before running (see [Challenges faced](#challenges-faced)).

## System architecture

```
data/raw/vllm-0.10.1/  (2,867 files)
        │
        ▼
  corpus.py            discover + filter files ............ 2,867 → 1,240 kept
        │
        ▼
  python_chunker.py    AST-aware chunking (.py)
  markdown_chunker.py  paragraph-aware chunking (.md/.txt) .. 1,240 files → 10,883 chunks
        │
        ▼
  indexer.py + bm25.py persist offsets, fit BM25 ........... data/processed/
        │
        ├──▶ search / search_dataset
        │      tokenizer.py → retriever.py .................. ranked MinimalSource[]
        │
        └──▶ answer / answer_dataset
               prompt_builder.py  re-read text by offset
               generator.py       Qwen/Qwen3-0.6B ........... MinimalAnswer
```

| Module | Responsibility | LOC |
|---|---|---|
| `corpus.py` | Walk the corpus, exclude non-source dirs, dispatch by extension | 53 |
| `chunk.py` / `chunk_utils.py` | Shared `Chunk` type + greedy span-packing engine | 151 |
| `markdown_chunker.py` | Paragraph-boundary chunking | 57 |
| `python_chunker.py` | AST-boundary chunking | 106 |
| `indexer.py` | Orchestrates chunking, persists `ChunkIndex` | 85 |
| `tokenizer.py` | Shared query/index tokenization | 32 |
| `bm25.py` | Fits + scores the BM25 statistical index | 136 |
| `retriever.py` | `search` / `search_dataset` | 130 |
| `evaluator.py` | Own Recall@k (IoU-based, mirrors moulinette) | 142 |
| `prompt_builder.py` | Augment: sources → grounded prompt | 77 |
| `generator.py` | Qwen3-0.6B loading, generation, `answer_dataset` | 211 |
| `models.py` | Pydantic data contracts | 63 |
| `__main__.py` | Fire CLI | 202 |

**1,445 lines**, 15 source files, zero `flake8`/`mypy --strict` findings.

## Chunking strategy

Two independent chunkers, sharing one greedy packing engine (`chunk_utils.pack_spans`) so the only thing that differs is *what counts as an indivisible unit*:

- **Markdown/text** (703 chunks): splits on blank lines into paragraphs, then greedily packs consecutive paragraphs into a chunk while it fits under `max_chunk_size`. A heading is never separated from the text that follows it, because the boundary can only ever land on a blank line.
- **Python** (10,180 chunks): parses with `ast`, packs whole top-level statements (imports, functions, classes) instead of paragraphs. A class too large to fit on its own recurses one level in and packs its *methods* instead — so it still only ever splits at a syntactic boundary, never mid-statement.

Both fall back the same way when a single unit is still too large: pack whole *lines*, and only hard-cut mid-line as an absolute last resort (a huge table row, an unbreakable string). Python additionally falls back straight to the line-based splitter if `ast.parse` raises `SyntaxError`, so one malformed file can never crash indexing.

Result: **10,883 chunks**, average **1,374 characters**, and the largest chunk ever produced is exactly **2,000** — `max_chunk_size` is a hard ceiling, verified, never exceeded (this matters: moulinette invalidates the *entire* output on a single over-long source).

Only **1,240 of 2,867** files are actually indexed. Excluded on purpose: `.git`, `.github`, `.buildkite` (CI plumbing), `csrc` (C++/CUDA kernels — outside the two required chunker types), `cmake`/`docker` (build plumbing, no conceptual content), `tests` (verified against the ground-truth datasets: no code question ever points there). Indexing everything would dilute the BM25 index with noise and cost indexing time for zero recall benefit.

## Retrieval method

**BM25** (Okapi variant), chosen over plain TF-IDF for two reasons that are concretely true of this corpus: term-frequency **saturation** (a term's 20th occurrence shouldn't count 20× a single occurrence) and **document-length normalization** (vLLM mixes tiny files with huge ones — without normalization, long files win purely by being long).

```
score(D,Q) = Σ  idf(t) · tf(t,D)·(k1+1) / (tf(t,D) + k1·(1-b+b·|D|/avgdl))
           t∈Q
```
`k1 = 1.5`, `b = 0.75` (standard defaults). `idf` uses BM25's own non-negative variant, `log((N-df+0.5)/(df+0.5)+1)`, so an ultra-common term contributes ~0 instead of a negative score.

**Tokenization** (must be byte-identical at index and query time, or scores silently break): lowercase, split on non-alphanumerics, **plus** sub-word tokens for identifiers — `add_request` also emits `add` and `request`, `getFreeBlocks`-style camelCase splits too. This directly targets the code-recall gap: a question rarely quotes `add_request` verbatim, but often uses the word "request" on its own. Tradeoff, accepted: this inflates the effective token count of code chunks, so BM25's length normalization discounts them slightly relative to prose of the same character size — didn't cost the recall threshold, so left as-is rather than "fixed."

Fitted index: **46,432 unique terms** over 10,883 chunks, persisted as `chunks.json` (1.8 MB, offsets only — no duplicate text) + `bm25_index.json` (18 MB, per-chunk term frequencies + corpus stats).

## Performance analysis

**Indexing** — `uv run python -m src index --max_chunk_size 2000`

| | |
|---|---|
| Files discovered / kept | 2,867 / 1,240 |
| Chunks produced | 10,883 (10,180 Python + 703 Markdown/text) |
| Time | **9.6 s** (budget: 300 s → 31× margin) |

**Retrieval** — `search_dataset`, both public datasets, `k=10`

| Dataset | Questions | Time |
|---|---|---|
| Docs | 100 | 7.3 s |
| Code | 99 | 8.8 s |
| **Combined** | **199** | **16.2 s** (budget: 90 s / 200 q → 5.6× margin) |

**Recall@k** — via the real `./moulinette evaluate_student_search_results`, `k=10`, `max_context_length=2000`:

| k | Docs | Code |
|---|---|---|
| 1 | 59.0 % | 37.4 % |
| 3 | 79.0 % | 52.5 % |
| **5** | **85.0 %** (req. ≥80%) | **61.6 %** (req. ≥50%) |
| 10 | 88.0 % | 72.7 % |

**Generation** — `answer_dataset`, Qwen3-0.6B, CPU, `max_new_tokens=300`, 100 docs questions: **[GENERATION_TIME_PLACEHOLDER]**. Ungraded (no threshold in the subject for this stage), reported for transparency.

### From slow to fast

The biggest lever wasn't algorithmic — it was a one-character bug. `corpus.py`'s exclusion check originally read `path in EXCLUDED_DIR_NAMES` instead of `part in EXCLUDED_DIR_NAMES`, so the entire exclusion list was silently a no-op: **1,965 files** were being indexed instead of the intended **1,240** — CI configs, build plumbing, and vendored kernel code all diluting the BM25 index for zero benefit. Fixing the loop variable both sped up indexing and removed retrieval noise, before any tuning of BM25 parameters was needed.

## Design decisions

- **Persist offsets, never text.** `ChunkIndex` stores only `file_path` + character offsets. Chunk text is always re-sliced from the source file on demand (fitting BM25, building prompts) — one source of truth, no risk of the index drifting from the corpus, no duplicated storage.
- **BM25 over TF-IDF**, for the saturation + length-normalization reasons above — concretely relevant given vLLM's file-size spread.
- **Sub-word identifier tokenization**, trading a small BM25 length-normalization bias against code chunks for meaningfully better recall on paraphrased code questions.
- **Zero-source robustness in `answer_dataset`**: a question with no retrieved sources skips generation entirely and gets a canned "No relevant sources found" answer, rather than asking a 0.6B model to answer with no context — removes a real hallucination risk for zero cost.
- **Per-question isolation in `answer_dataset`**: one failing generation is caught and recorded as that question's answer rather than aborting a batch that can run 15+ minutes on CPU.
- **Prompt budget is greedy, not truncating.** Sources are added in rank order until the next one would exceed a soft character budget, then dropped whole — never truncated mid-chunk, which risks cutting off exactly the sentence that answers the question. At least one source is always included even if it alone exceeds the budget.

## Challenges faced

| # | Problem | Fix | Verified impact |
|---|---|---|---|
| 1 | Corpus exclusion list was a silent no-op (`path` vs `part` in the membership check) | One-line fix | 1,965 → 1,240 files indexed |
| 2 | Tokenizer had a stray early `return` inside its loop — only the first token of every chunk was ever indexed | Moved `return` outside the loop | Caught before it shipped, via an unexpectedly tiny vocabulary |
| 3 | `Makefile`'s `lint` target used `/` instead of `\` for line continuation, silently splitting one command into three (one of which pointed mypy at `/`, the filesystem root) | One-character fix | `make lint` actually runs the intended command |
| 4 | `transformers`' stubs are incomplete for dynamically-mixed-in methods: `model.generate` types as `Tensor \| Module` under mypy, `tokenizer.decode()` returns an overly broad `str \| list[str]` union | Two narrowly-scoped `# type: ignore[operator]` / `[union-attr]`, not a blanket suppression | `mypy --strict` clean without hiding real errors |
| 5 | `python-fire` ships no type stubs anywhere (no `types-fire` package exists) | Per-module `ignore_missing_imports` override in `pyproject.toml`, scoped to just `fire` | Rest of the codebase stays under full `--strict` |
| 6 | Hugging Face's `hf_xet` transfer backend downloaded model weights at ~60–250 KB/s despite a ~9.5 MB/s connection | `HF_HUB_DISABLE_XET=1` forces the plain HTTP downloader | One-time download went from unbounded to under a minute |
| 7 | `moulinette` rejects a results file if any question has more sources than the `--k` passed to *moulinette itself* — independent of `search_dataset -k` | Keep both flags in sync | Avoids a false "invalid" verdict; Recall@5 is unaffected either way |

## Example usage

```
❯ uv run python -m src search "How to configure the OpenAI server?" -k 3
data/raw/vllm-0.10.1/examples/online_serving/openai_chat_completion_client_with_tools_required.py [107:566]
data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md [1936:3170]
data/raw/vllm-0.10.1/examples/online_serving/openai_transcription_client.py [107:1350]
```

```
❯ uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" -k 5

Answer: The HTTP endpoint used to dynamically load a LoRA adapter in vLLM is `/v1/load_lora_adapter`.

Sources used:
 data/raw/vllm-0.10.1/docs/features/lora.md [3835:5714]
 data/raw/vllm-0.10.1/docs/features/lora.md [5716:7656]
 data/raw/vllm-0.10.1/vllm/plugins/lora_resolvers/README.md [0:830]
 data/raw/vllm-0.10.1/examples/others/tensorize_vllm_model.py [2618:4457]
 data/raw/vllm-0.10.1/docs/features/lora.md [0:1882]
```

## Resources

**Technical references**
- Robertson, S. & Zaragoza, H., *The Probabilistic Relevance Framework: BM25 and Beyond* (2009) — the ranking function behind retrieval
- Salton, G. & Buckley, C., *Term-weighting approaches in automatic text retrieval* — TF-IDF background
- [vLLM documentation](https://docs.vllm.ai/) — the indexed corpus itself
- [Hugging Face Transformers docs](https://huggingface.co/docs/transformers) — `generate`, `apply_chat_template`
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- Python [`ast`](https://docs.python.org/3/library/ast.html) module docs
- [Pydantic](https://docs.pydantic.dev/), [python-fire](https://github.com/google/python-fire), [uv](https://docs.astral.sh/uv/) docs

**AI usage**: Claude Code (Anthropic) was used throughout as a milestone-by-milestone pairing/teaching tool, per the project's own learning workflow — for each milestone it first explained the underlying concept and any design tradeoff (e.g. TF-IDF/BM25 math worked by hand, AST-based chunking boundaries, why offsets are persisted instead of text) before any code was written; I typed and understood every line of `src/` myself rather than having it inserted directly. Claude also: helped debug the seven real issues listed in [Challenges faced](#challenges-faced) by reasoning about root cause rather than guessing; ran the verification commands (`make lint`, `make lint-strict`, the full `index → search_dataset → moulinette → answer_dataset` pipeline) that produced every number in this README; and, with my explicit go-ahead, wrote this README file directly (the only project file it wrote directly — all `src/` code was written by hand, milestone by milestone, in chat first).
