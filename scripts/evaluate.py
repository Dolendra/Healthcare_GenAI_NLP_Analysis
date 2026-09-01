#!/usr/bin/env python3
"""
Evaluate Gemini predictions against manual_validation.csv ground truth.

Fill Expected_* columns in outputs/manual_validation.csv before running.
Do NOT edit predictions_NDST.csv — compare raw Gemini output to human labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import baseline_classify_topics, baseline_sentiment
from src.config import PREDICTIONS_FILE, VALIDATION_SAMPLE_SIZE
from src.evaluation import (
    multilabel_set_metrics,
    parse_list_field,
    precision_recall_f1,
    sentiment_pair_metrics,
)

VALIDATION_FILE = PROJECT_ROOT / "outputs" / "manual_validation.csv"
REPORT_FILE = PROJECT_ROOT / "outputs" / "evaluation_report.txt"


def _parse_expected_list(value) -> list[str]:
    items = parse_list_field(value)
    return [str(x).strip() for x in items if str(x).strip()]


def _parse_expected_sentiments(value) -> dict[str, str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    if text.startswith("{"):
        data = json.loads(text)
        return {k: str(v).lower() for k, v in data.items()}
    # topic:sentiment;topic:sentiment
    result = {}
    for part in text.split(";"):
        if ":" in part:
            topic, sentiment = part.split(":", 1)
            result[topic.strip()] = sentiment.strip().lower()
    return result


def entity_metrics(expected_lists: list[list[str]], predicted_lists: list[list[str]]) -> dict:
    tp = fp = fn = 0
    for exp, pred in zip(expected_lists, predicted_lists):
        exp_set = {x.lower() for x in exp}
        pred_set = {x.lower() for x in pred}
        tp += len(exp_set & pred_set)
        fp += len(pred_set - exp_set)
        fn += len(exp_set - pred_set)
    return precision_recall_f1(tp, fp, fn)


def main() -> int:
    if not VALIDATION_FILE.exists():
        print(f"ERROR: {VALIDATION_FILE} not found. Run create_validation_sample.py first.")
        return 1

    manual = pd.read_csv(VALIDATION_FILE)
    pred = pd.read_csv(PREDICTIONS_FILE)

    labeled = manual[
        manual["Expected_Topics"].fillna("").astype(str).str.strip().ne("")
        | manual["Expected_Drugs"].fillna("").astype(str).str.strip().ne("")
    ]

    if labeled.empty:
        print("No labeled records found in manual_validation.csv.")
        print("Fill Expected_Drugs, Expected_Diseases, Expected_Study_Names,")
        print("Expected_Topics, and Expected_Sentiments, then re-run.")
        print(f"Validation sample has {len(manual)} records ready for annotation.")
        return 2

    merged = labeled.merge(pred, on="Record_ID", how="left", suffixes=("_gt", "_pred"))

    report: list[str] = []

    exp_drugs, pred_drugs = [], []
    exp_diseases, pred_diseases = [], []
    exp_studies, pred_studies = [], []
    topic_true_sets, topic_pred_sets = [], []
    baseline_topic_sets: list[set] = []
    sentiment_true_pairs, sentiment_pred_pairs = [], []
    baseline_sentiment_pairs: list[tuple[str, str]] = []

    for _, row in merged.iterrows():
        exp_drugs.append(_parse_expected_list(row.get("Expected_Drugs")))
        pred_drugs.append(parse_list_field(row.get("Drugs")))
        exp_diseases.append(_parse_expected_list(row.get("Expected_Diseases")))
        pred_diseases.append(parse_list_field(row.get("Diseases")))
        exp_studies.append(_parse_expected_list(row.get("Expected_Study_Names")))
        pred_studies.append(parse_list_field(row.get("Study_Names")))

        exp_topics = set(_parse_expected_list(row.get("Expected_Topics")))
        pred_topics = set(parse_list_field(row.get("Topics")))
        nlp_text = str(row.get("NLP_Text", row.get("Combined", "")))
        base_topics = set(baseline_classify_topics(nlp_text))

        topic_true_sets.append(exp_topics)
        topic_pred_sets.append(pred_topics)
        baseline_topic_sets.append(base_topics)

        exp_sent = _parse_expected_sentiments(row.get("Expected_Sentiments"))
        pred_sent_map = dict(
            zip(parse_list_field(row.get("Topics")), parse_list_field(row.get("Topic_Sentiments")))
        )
        for topic, sentiment in exp_sent.items():
            sentiment_true_pairs.append((topic, sentiment))
            sentiment_pred_pairs.append((topic, pred_sent_map.get(topic, "__missing__")))
            baseline_sentiment_pairs.append((topic, baseline_sentiment(nlp_text, topic)))

    def emit(line: str = "") -> None:
        report.append(line)
        print(line)

    emit("=" * 50)
    emit("EVALUATION REPORT")
    emit("=" * 50)
    emit(f"Labeled records: {len(merged)}")

    emit("\nENTITY METRICS (Gemini)")
    emit("-" * 50)
    entity_results: dict[str, dict] = {}
    for name, exp, pr in [
        ("Drugs", exp_drugs, pred_drugs),
        ("Diseases", exp_diseases, pred_diseases),
        ("Study Names", exp_studies, pred_studies),
    ]:
        m = entity_metrics(exp, pr)
        entity_results[name] = m
        emit(f"{name:12} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    emit("\nTOPIC METRICS")
    emit("-" * 50)
    gemini_topics = multilabel_set_metrics(
        [set(x) for x in topic_true_sets], [set(x) for x in topic_pred_sets]
    )
    baseline_topics = multilabel_set_metrics(
        [set(x) for x in topic_true_sets], [set(x) for x in baseline_topic_sets]
    )
    emit(
        f"Gemini   P={gemini_topics['precision']:.3f} "
        f"R={gemini_topics['recall']:.3f} F1={gemini_topics['f1']:.3f}"
    )
    emit(
        f"Baseline P={baseline_topics['precision']:.3f} "
        f"R={baseline_topics['recall']:.3f} F1={baseline_topics['f1']:.3f}"
    )

    emit("\nBASELINE VS GEMINI (F1)")
    emit("-" * 50)
    emit(f"{'Task':<14} {'Baseline F1':>12} {'Gemini F1':>12}")
    emit(f"{'Topics':<14} {baseline_topics['f1']:>12.3f} {gemini_topics['f1']:>12.3f}")
    for name in ("Drugs", "Diseases", "Study Names"):
        emit(f"{name:<14} {'n/a':>12} {entity_results[name]['f1']:>12.3f}")

    emit("\nSENTIMENT METRICS (topic-level pairs)")
    emit("-" * 50)
    if sentiment_true_pairs:
        gemini_sent = sentiment_pair_metrics(sentiment_true_pairs, sentiment_pred_pairs)
        base_sent = sentiment_pair_metrics(sentiment_true_pairs, baseline_sentiment_pairs)
        emit(
            f"Gemini   Acc={gemini_sent['accuracy']:.3f} "
            f"P={gemini_sent['precision']:.3f} R={gemini_sent['recall']:.3f} F1={gemini_sent['f1']:.3f}"
        )
        emit(
            f"Baseline Acc={base_sent['accuracy']:.3f} "
            f"P={base_sent['precision']:.3f} R={base_sent['recall']:.3f} F1={base_sent['f1']:.3f}"
        )
        emit(f"{'Sentiment':<14} {base_sent['f1']:>12.3f} {gemini_sent['f1']:>12.3f}")
    else:
        emit("No Expected_Sentiments labels found.")

    emit("\nERROR SAMPLES (entity/topic mismatches)")
    emit("-" * 50)
    shown = 0
    for _, row in merged.iterrows():
        rid = int(row["Record_ID"])
        exp_topics = set(_parse_expected_list(row.get("Expected_Topics")))
        pred_topics = set(parse_list_field(row.get("Topics")))
        exp_studies = {x.lower() for x in _parse_expected_list(row.get("Expected_Study_Names"))}
        pred_studies = {x.lower() for x in parse_list_field(row.get("Study_Names"))}
        fp_topics = pred_topics - exp_topics
        fn_topics = exp_topics - pred_topics
        fp_studies = pred_studies - exp_studies
        if not (fp_topics or fn_topics or fp_studies):
            continue
        emit(f"Record {rid}: FP topics={sorted(fp_topics)} FN topics={sorted(fn_topics)} FP studies={sorted(fp_studies)}")
        shown += 1
        if shown >= 10:
            break

    emit("=" * 50)
    report.append(f"\nSaved report: {REPORT_FILE}")
    write_report(report, REPORT_FILE)
    print(f"Saved report: {REPORT_FILE}")
    return 0


def write_report(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
