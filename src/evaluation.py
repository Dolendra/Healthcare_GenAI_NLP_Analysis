"""Shared evaluation utilities for quality checks and metrics."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

import pandas as pd

from .config import SENTIMENTS, TOPICS, VALID_SENTIMENTS, VALID_TOPICS

SUSPICIOUS_DRUG_TERMS = {
    "chemotherapy",
    "immunotherapy",
    "radiotherapy",
    "radiation therapy",
    "surgery",
    "therapy",
    "treatment",
    "endocrine therapy",
    "hormone therapy",
    "targeted therapy",
}

REQUIRED_PREDICTION_COLUMNS = [
    "Record_ID",
    "Source",
    "Combined",
    "NLP_Text",
    "Drugs",
    "Diseases",
    "Study_Names",
    "Topics",
    "Topic_Sentiments",
    "Evidence",
    "Model_Confidence_Scores",
    "Processing_Status",
]


def parse_list_field(value: Any) -> list:
    """Parse a CSV field that may be a JSON list or Python list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        return [text] if text else []


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def multilabel_set_metrics(y_true: list[set], y_pred: list[set]) -> dict[str, float]:
    """Micro-averaged precision/recall/F1 over multi-label sets."""
    tp = fp = fn = 0
    for true_set, pred_set in zip(y_true, y_pred):
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    return precision_recall_f1(tp, fp, fn)


def sentiment_pair_metrics(
    pairs_true: list[tuple[str, str]],
    pairs_pred: list[tuple[str, str]],
) -> dict[str, float]:
    """Accuracy-style metrics for (topic, sentiment) pairs."""
    true_set = set(pairs_true)
    pred_set = set(pairs_pred)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    accuracy = tp / len(true_set) if true_set else 0.0
    prf = precision_recall_f1(tp, fp, fn)
    prf["accuracy"] = accuracy
    return prf


def audit_suspicious_drugs(drugs: Iterable[str]) -> list[str]:
    return [d for d in drugs if d.strip().lower() in SUSPICIOUS_DRUG_TERMS]


def validate_prediction_row(row: pd.Series) -> list[str]:
    """Return validation issue strings for one prediction row."""
    issues: list[str] = []

    topics = parse_list_field(row.get("Topics"))
    sentiments = parse_list_field(row.get("Topic_Sentiments"))
    evidence = parse_list_field(row.get("Evidence"))
    confidences = parse_list_field(row.get("Model_Confidence_Scores"))

    if len({len(topics), len(sentiments), len(evidence), len(confidences)}) > 1:
        issues.append("topic/sentiment/evidence/confidence length mismatch")

    for topic in topics:
        if topic not in VALID_TOPICS:
            issues.append(f"invalid topic: {topic}")

    for sentiment in sentiments:
        if str(sentiment).lower() not in VALID_SENTIMENTS:
            issues.append(f"invalid sentiment: {sentiment}")

    for item in evidence:
        if not str(item).strip():
            issues.append("empty evidence snippet")

    for conf in confidences:
        try:
            value = float(conf)
            if value < 0 or value > 1:
                issues.append(f"confidence out of range: {value}")
        except (TypeError, ValueError):
            issues.append(f"non-numeric confidence: {conf}")

    suspicious = audit_suspicious_drugs(parse_list_field(row.get("Drugs")))
    for drug in suspicious:
        issues.append(f"suspicious drug term: {drug}")

    return issues
