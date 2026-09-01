"""Prompt templates for healthcare NLP extraction via Gemini."""

from __future__ import annotations

from typing import Any, Mapping

from .config import TOPICS

TOPIC_LIST_BULLET = "\n".join(f"- {t}" for t in TOPICS)

SYSTEM_INSTRUCTION = """
You are a healthcare intelligence information extraction system.

Analyze the provided healthcare article or social media post.

Your tasks are:

1. Extract explicitly mentioned drugs.
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
3. A document can contain multiple topics.
4. Sentiment must be assigned independently for each topic.
5. Use "Others" when the content does not fit the specified topics.
6. If no drug is mentioned, return an empty list.
7. If no disease is mentioned, return an empty list.
8. If no study name is mentioned, return an empty list.
9. Provide a short evidence snippet supporting each topic sentiment.
10. Confidence must be between 0 and 1.
11. For social media posts with ORIGINAL POST and REPLIED-TO TWEET sections,
    attribute medical claims to the section where they appear. Do not attribute
    replied-to medical content to the original poster unless they repeat it.
""".strip()

FEW_SHOT_EXAMPLES = """
Example 1 — Media article:
Text: "The KEYNOTE-189 trial demonstrated a significant improvement in overall survival with pembrolizumab plus chemotherapy in non-small cell lung cancer, although nausea and fatigue were reported more frequently."

Expected:
- drugs: ["pembrolizumab", "chemotherapy"]
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


def build_record_text(row: Mapping[str, Any]) -> str:
    """
    Build NLP input text from a standardized record.

    Media: Title + Body via Combined.
    Twitter: labeled ORIGINAL POST + REPLIED-TO TWEET to avoid attribution errors.
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
