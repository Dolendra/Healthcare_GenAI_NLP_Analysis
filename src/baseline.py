"""Rule-based keyword baseline for comparison against Gemini outputs."""

from __future__ import annotations

import re
from typing import List

from .config import TOPICS

# Keyword rules → topic mapping (simple baseline, not production-quality)
TOPIC_KEYWORDS = {
    "Efficacy-General": [
        r"\befficacy\b",
        r"\beffective\b",
        r"\bresponse rate\b",
        r"\btumor shrink",
        r"\bclinical benefit\b",
    ],
    "Progression Free Survival (PFS)": [
        r"\bprogression[- ]free survival\b",
        r"\bpfs\b",
        r"\bprogression free\b",
    ],
    "Overall Survival (OS)": [
        r"\boverall survival\b",
        r"\bos\b",
        r"\bsurvival benefit\b",
        r"\bmedian survival\b",
    ],
    "Safety-General": [
        r"\bsafety profile\b",
        r"\btolerability\b",
        r"\btolerated well\b",
        r"\bsafe\b",
    ],
    "Safety-Side Effects": [
        r"\bnausea\b",
        r"\bfatigue\b",
        r"\brash\b",
        r"\bdiarrhea\b",
        r"\badverse event",
        r"\bside effect",
        r"\btoxicity\b",
    ],
    "General Opinion": [
        r"\bexcited\b",
        r"\bpromising\b",
        r"\bdisappoint",
        r"\bhopeful\b",
        r"\bconcern",
        r"\bimpressive\b",
    ],
}


def baseline_classify_topics(text: str) -> List[str]:
    """Return topics matched by keyword rules. Falls back to 'Others'."""
    text_lower = text.lower()
    matched = []

    for topic, patterns in TOPIC_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matched.append(topic)
                break

    return matched if matched else ["Others"]


def baseline_sentiment(text: str, topic: str) -> str:
    """Crude sentiment from positive/negative word lists."""
    text_lower = text.lower()
    positive = ["improve", "benefit", "promising", "significant", "effective", "safe", "excited"]
    negative = ["worse", "failed", "toxic", "severe", "disappoint", "concern", "risk", "adverse"]

    pos = sum(1 for w in positive if w in text_lower)
    neg = sum(1 for w in negative if w in text_lower)

    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"
