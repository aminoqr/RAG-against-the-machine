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

from src.models import MinimalAnswer, MinimalSource
from src.prompt_builder import build_prompt

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 300


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
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_only = output_ids[0][n_input_tokens:]
    return tokenizer.decode(generated_only, skip_special_tokens=True).strip()


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