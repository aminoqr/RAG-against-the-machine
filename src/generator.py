"""Loading Qwen3-0.6B and generates grounded answers from retrieved
sources: the "generate" step, wired to the "retrieve" and "augment"
steps via answer_question()."""

import uuid

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import json
from pathlib import Path

from pydantic import ValidationError
from tqdm import tqdm

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.prompt_builder import build_prompt

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 300
NO_SOURCES_ANSWER = "No relevant sources found -- cannot answer."


def load_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the tokenizer and model once, for reuse across many
    generate calls (loading takes a few seconds -- don't reapeat it
    per question in a batch).

    Returns:
        (tokenizer, model), ready to pass into generate_answer().
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto")
    return tokenizer, model


def generate_answer(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Run one prompt through an already-loaded model and decode the
    generated answer text.

    Args:
        prompt: the full promp built by build_promp().
        tokenizer: a loaded tokenizer from load_model().
        model: a loaded model from load_model().
        max_new_tokens: generation length cap.

    Returns:
        The decoded answer text (generated tokens only).
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    n_input_tokens = inputs["input_ids"].shape[1]

    output_ids = model.generate(  # type: ignore[operator]
        **inputs, max_new_tokens=max_new_tokens
    )
    generated_only = output_ids[0][n_input_tokens:]
    decoded = tokenizer.decode(
        generated_only, skip_special_tokens=True
    )
    return decoded.strip()  # type: ignore[union-attr]


def answer_question(
    question: str,
    sources: list[MinimalSource],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    max_new_tokens: int = MAX_NEW_TOKENS,
    question_id: str | None = None,
) -> MinimalAnswer:
    """Build a grounded prompt from sources and generate an answer.

    Args:
        question: the question text.
        sources: retrieved source locations from search(), best first.
        tokenizer: a loaded tokenizer from load_model().
        model: a loaded model from load_model().
        max_new_tokens: generation length cap.
        question_id: an existing question_id to reuse (e.g. when
            answering a dataset question); a fresh UUID is generated
            if not given.

    Returns:
        A MinimalAnswer with the question, retrieved sources, and the
        generated answer text.
    """
    prompt = build_prompt(question, sources)
    answer_text = generate_answer(prompt, tokenizer, model, max_new_tokens)
    return MinimalAnswer(
        question_id=question_id or str(uuid.uuid4()),
        question=question,
        retrieved_sources=sources,
        answer=answer_text,
    )


def answer_dataset(
        student_search_results_path: Path,
        save_directory: Path,
        max_new_tokens: int = MAX_NEW_TOKENS,
) -> Path:
    """Generate an answer for every question in a StudentSearchResults
    file (search_dataset's output) and presist a
    StudentSearchResultsAndAnswer JSON under save_directory.

    Loads the model once for the whole batch. Questions with zero
    retrieved sources skip generation entirely (a canned answer is
    used instead) -- there's nothing to ground an answer in, so
    asking the model anyway would only risk a hallucinated respone
    for no benefit. A per-question generation failure doesn't abort
    the whole batch: it's recorded as that question's answer so
    earlier work in a long CPU run isn't lost.

    Args:
        student_search_results_path: path to a StudentSearchResults
            JSON file (search_dataset's output).
        save_directory: directory to write the output JSON into
            (created if missing); the output file keeps the input
            file's name (e.g. dataset_docs_public.json).
        max_new_tokens: generation length cap, passed through to
            generate_answer for every question.

    Returns:
        The path of the written StudentSearchResultsAndAnswer JSON.

    Raises:
        FileNotFoundError: if student_search_results_path doesn't exist.
        ValueError: if student_search_results_path isn't valid JSON, or
            doesn't match the expected StudentSearchResults schema.
    """
    try:
        with student_search_results_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{student_search_results_path} is not valid JSON: {e}"
        ) from e

    try:
        student_results = StudentSearchResults.model_validate(raw)
    except ValidationError as e:
        raise ValueError(
            f"{student_search_results_path} doesn't match the expected "
            f"StudentSearchResults format: {e}"
        ) from e

    print("Loading model...")
    tokenizer, model = load_model()

    answers: list[MinimalAnswer] = []
    for result in tqdm(student_results.search_results, desc="Answering"):
        if not result.retrieved_sources:
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=[],
                    answer=NO_SOURCES_ANSWER,
                )
            )
            continue

        try:
            answer = answer_question(
                result.question,
                result.retrieved_sources,
                tokenizer,
                model,
                max_new_tokens,
                question_id=result.question_id,
            )
        except Exception as e:
            answer = MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=f"Generation failed: {e}",
            )
        answers.append(answer)

    output = StudentSearchResultsAndAnswer(
        search_results=answers, k=student_results.k
    )

    save_directory.mkdir(parents=True, exist_ok=True)
    out_path = save_directory / student_search_results_path.name
    with out_path.open("w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    return out_path
