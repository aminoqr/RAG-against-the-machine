"""Builds the generation prompt (the "augment" step): reads the actual
text for each retrieved source and assembles it with the questions and
a grounding instruction."""

from pathlib import Path

from src.models import MinimalSource

MAX_CONTENT_CHARS = 3000

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant answering questions about the vLLM "
    "source code and documentation. Use ONLY the information in the  "
    "context below to answer the question. If the context doesn't "
    "contain enough information to answer, say so directly instead of"
    "guessing. Keep your answer concise and focused on exactly what "
    "was asked."
)


def _read_source_text(source: MinimalSource) -> str:
    """Read the exact text slice a MinimalSource points to."""
    try:
        text = Path(source.file_path).read_text(
            encoding="utf-8", errors="ignore"
        )
    except OSError:
        return ""
    return text[source.first_character_index:source.last_character_index]
    
    
def build_prompt(
    question: str,
    sources: list[MinimalSource],
    max_context_chars: int = MAX_CONTENT_CHARS,
) -> str:
    """Assemble a grounded generation prompt from retrieved sources/
    
    Sources are expected in rank order (best first, as returned by
    search()). They're included in that order until adding the next
    one would exceed max_context_chars; lower-ranking sources beyond
    that point are dropped entirely rather than truncating a chunk
    mid-way, since a partial chunk risks cutting off exactly the
    sentence that answers the question. At least one source is always
    included if any were passed, even if it alone exceeds the budget.
    
    Args:
        question: the user's question.
        sources: ranked source locations from search(), best first.
        max_context_chars: soft budget on total retrieved-context
            characters included in the prompt.
    
    Returns:
        The full prompt text, ready to use as the user turn's content
        when building the messages list for apply_chat_template.
    """
    context_blocks: list[str] = []
    total_chars = 0
    
    for source in sources:
        text = _read_source_text(source)
        if not text:
            continue
        block = f"[Source: {source.file_path}]\n{text}"
        if total_chars + len(block) > max_context_chars and context_blocks:
            break
        context_blocks.append(block)
        total_chars += len(block)
        
    context = "\n\n".join(context_blocks)
    
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )