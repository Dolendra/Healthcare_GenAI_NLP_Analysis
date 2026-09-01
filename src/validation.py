"""Pydantic schemas and post-LLM validation for structured NLP output."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .config import VALID_SENTIMENTS, VALID_TOPICS


class TopicSentiment(BaseModel):
    """Single topic with sentiment, evidence, and model self-assessed confidence."""

    topic: str
    sentiment: str
    evidence: str
    model_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("topic")
    @classmethod
    def topic_must_be_valid(cls, v: str) -> str:
        if v not in VALID_TOPICS:
            raise ValueError(f"Invalid topic: {v!r}. Must be one of {VALID_TOPICS}")
        return v

    @field_validator("sentiment")
    @classmethod
    def sentiment_must_be_valid(cls, v: str) -> str:
        v_lower = v.lower().strip()
        if v_lower not in VALID_SENTIMENTS:
            raise ValueError(f"Invalid sentiment: {v!r}")
        return v_lower

    @field_validator("evidence")
    @classmethod
    def evidence_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Evidence snippet cannot be empty")
        return v.strip()


class NLPResult(BaseModel):
    """Full structured output from Gemini for one text record."""

    drugs: List[str] = Field(default_factory=list)
    diseases: List[str] = Field(default_factory=list)
    study_names: List[str] = Field(default_factory=list)
    topics: List[TopicSentiment] = Field(default_factory=list)

    @field_validator("drugs", "diseases", "study_names", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return list(v)


class FailedNLPResult(BaseModel):
    """Placeholder returned when LLM processing fails after all retries."""

    drugs: List[str] = Field(default_factory=list)
    diseases: List[str] = Field(default_factory=list)
    study_names: List[str] = Field(default_factory=list)
    topics: List[TopicSentiment] = Field(default_factory=list)
    error: Optional[str] = None


def validate_nlp_result(raw: dict) -> NLPResult:
    """Parse and validate a raw Gemini JSON response."""
    # Accept legacy field name from older responses
    for topic in raw.get("topics", []):
        if "confidence" in topic and "model_confidence" not in topic:
            topic["model_confidence"] = topic.pop("confidence")
    return NLPResult.model_validate(raw)


def result_to_row_dict(result: NLPResult) -> dict:
    """Flatten NLPResult into columns suitable for a DataFrame row."""
    return {
        "Drugs": result.drugs,
        "Diseases": result.diseases,
        "Study_Names": result.study_names,
        "Topics": [t.topic for t in result.topics],
        "Topic_Sentiments": [t.sentiment for t in result.topics],
        "Evidence": [t.evidence for t in result.topics],
        "Model_Confidence_Scores": [t.model_confidence for t in result.topics],
    }
