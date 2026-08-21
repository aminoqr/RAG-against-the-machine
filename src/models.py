"""Pydantic data models shared across the RAG pipeline stages."""

import uuid
from pydantic import BaseModel, Field

class MinimalSource(BaseModel):
    """A pointer to one chunk of the corpus: which
    file, which character range."""

    file_path: str
    first_character_index: int
    last_character_index: int

class UnansweredQuestion(BaseModel):
    """A question with no known answer yet
    -- what search_dataset/answer_dataset take as input."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    """An UnansweredQuestion plus its good-truth sources and answer."""

    sources: list[MinimalSource]
    answer: str

class RagDataset(BaseModel):
    """A full dataset file: a list of questions, answered or not."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

class MinimalSearchResults(BaseModel):
    """One question ples whatever sources your retriever found for it."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    """MinimalSearchResults plus the generated answer."""

    answer: str

class StudentSearchResults(BaseModel):
    """What search_dataset writes: search results for every question in
    a dataset"""

    search_results: list[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    """What answer_dataset writes: answers for every question in a dataset."""

    search_results: list[MinimalAnswer]
    k: int