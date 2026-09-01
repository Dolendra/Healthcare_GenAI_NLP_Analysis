"""Prompt templates for healthcare NLP extraction via Gemini."""

from __future__ import annotations

from typing import Any, Mapping

from .config import TOPICS

TOPIC_LIST_BULLET = "\n".join(f"- {t}" for t in TOPICS)

SYSTEM_INSTRUCTION = """
You are a healthcare intelligence information extraction system.

Analyze the provided healthcare article or social media post.

Your tasks are:

1. Extract explicitly mentioned specific drugs/therapeutic agents.
2. Extract explicitly mentioned diseases.
3. Extract explicitly mentioned study names.
4. Identify all applicable healthcare topics.
5. Determine sentiment independently for each identified topic.

Allowed topics:

- Efficacy-General
- Progression Free Survival (PFS)
- Overall Survival (OS)
- Safety-General
- Safety-Side Effects
- General Opinion
- Others

Allowed sentiments:

- positive
- negative
- neutral

Rules:

1. Extract only entities explicitly mentioned in the text.
2. Never invent or infer entities.
3. Extract specific named drugs/therapeutic agents only. Do NOT classify broad
   treatment modalities (chemotherapy, radiotherapy, immunotherapy, surgery) as
   individual drugs unless a specific agent is named.
4. A document can contain multiple topics.
5. Sentiment must be assigned independently for each topic.
6. Do not assign a topic solely because a keyword appears. Classify based on
   the meaning and context of the statement.
7. Use "Others" when the content does not fit the specified topics.
8. If no drug is mentioned, return an empty list.
9. If no disease is mentioned, return an empty list.
10. If no study name is mentioned, return an empty list.
11. Only extract study/trial names explicitly named in the text. Do not infer or
    guess related trials from drug names, diseases, treatment regimens, or other
    contextual clues.
12. Provide a short evidence snippet supporting each topic sentiment.
13. model_confidence must be between 0 and 1 (model self-assessment, not a
    calibrated probability).
14. For social media posts with ORIGINAL POST and REPLIED-TO TWEET sections,
    attribute medical claims to the section where they appear. Do not attribute
    replied-to medical content to the original poster unless they repeat it.
""".strip()

FEW_SHOT_EXAMPLES = """
Example 1 — Media article:
Text: "The KEYNOTE-189 trial demonstrated a significant improvement in overall survival with pembrolizumab in non-small cell lung cancer, although nausea and fatigue were reported more frequently."

Expected:
- drugs: ["pembrolizumab"]  (not "chemotherapy" — that is a treatment modality)
- diseases: ["non-small cell lung cancer"]
- study_names: ["KEYNOTE-189"]
- topics: Overall Survival (OS) → positive; Safety-Side Effects → negative

Example 2 — Twitter reply with context:
ORIGINAL POST:
Thanks Chandler!!

REPLIED-TO TWEET:
Patient reported outcomes for EV-302 showed better pain and QOL for patients on EV-pembrolizumab.

Expected:
- drugs: ["EV-pembrolizumab"]  (from replied-to tweet only)
- study_names: ["EV-302"]
- topics from replied-to content; General Opinion may apply to original post only if supported
"""


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def record_uses_reply_context(row: Mapping[str, Any]) -> bool:
    """True when Twitter reply context is included in NLP input."""
    if _safe_text(row.get("Source")) != "Twitter":
        return False
    replied_to = _safe_text(row.get("replied_to_tweet"))
    original_post = _safe_text(row.get("Body")) or _safe_text(row.get("Combined"))
    return bool(replied_to and replied_to not in original_post)


def nlp_text_type(row: Mapping[str, Any]) -> str:
    """Describe which text fields were sent to the LLM."""
    if _safe_text(row.get("Source")) == "Twitter":
        return "Original Post + Reply Context" if record_uses_reply_context(row) else "Original Post"
    return "Combined"


def build_record_text(row: Mapping[str, Any]) -> str:
    """
    Authoritative NLP input builder.

    Media: Combined (Title + Body).
    Twitter: labelled ORIGINAL POST + REPLIED-TO TWEET when reply context exists.
    """
    source = _safe_text(row.get("Source"))

    if source == "Twitter":
        original_post = _safe_text(row.get("Body")) or _safe_text(row.get("Combined"))
        replied_to = _safe_text(row.get("replied_to_tweet"))

        if replied_to and replied_to not in original_post:
            return (
                f"ORIGINAL POST:\n{original_post}\n\n"
                f"REPLIED-TO TWEET:\n{replied_to}"
            )
        return f"ORIGINAL POST:\n{original_post}"

    combined = _safe_text(row.get("Combined"))
    if combined:
        return combined

    title = _safe_text(row.get("Title"))
    body = _safe_text(row.get("Body"))
    return " ".join(part for part in [title, body] if part)


def build_prompt(text: str) -> str:
    """Build the user prompt for a pre-formatted text block."""
    return f"""
{SYSTEM_INSTRUCTION}

{FEW_SHOT_EXAMPLES}

TEXT TO ANALYZE:

{text}
""".strip()


def build_extraction_prompt(text: str) -> str:
    """Alias used by gemini_client for structured extraction."""
    return build_prompt(text)


def build_prompt_for_record(row: Mapping[str, Any]) -> str:
    """Build prompt from a standardized DataFrame row (dict-like)."""
    return build_prompt(build_record_text(row))
