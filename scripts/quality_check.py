#!/usr/bin/env python3
"""Run final quality checks on standardized data and predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EXPECTED_COMBINED_ROWS,
    EXPECTED_MEDIA_ROWS,
    EXPECTED_TWITTER_ROWS,
    PREDICTIONS_FILE,
    STANDARDIZED_FILE,
)
from src.evaluation import (
    REQUIRED_PREDICTION_COLUMNS,
    audit_suspicious_drugs,
    parse_list_field,
    validate_prediction_row,
)


def main() -> int:
    if not PREDICTIONS_FILE.exists():
        print(f"ERROR: Predictions file not found: {PREDICTIONS_FILE}")
        return 1

    std = pd.read_csv(STANDARDIZED_FILE) if STANDARDIZED_FILE.exists() else None
    pred = pd.read_csv(PREDICTIONS_FILE)

    print("=" * 50)
    print("TASK 1 — FINAL QUALITY REPORT")
    print("=" * 50)

    # DATA
    print("\nDATA")
    print("-" * 50)
    if std is not None:
        media_n = int((std["Source"] == "Media").sum())
        twitter_n = int((std["Source"] == "Twitter").sum())
        print(f"Media records:                    {media_n}")
        print(f"Twitter records:                  {twitter_n}")
        print(f"Total records:                    {len(std)}")
        print(f"Duplicate Record_ID:              {std['Record_ID'].duplicated().sum()}")
        print(f"Duplicate unique_id:              {std['unique_id'].duplicated().sum()}")
    else:
        media_n = int((pred["Source"] == "Media").sum())
        twitter_n = int((pred["Source"] == "Twitter").sum())
        print(f"Media records:                    {media_n}")
        print(f"Twitter records:                  {twitter_n}")
        print(f"Total records:                    {len(pred)}")

    # LLM PROCESSING
    print("\nLLM PROCESSING")
    print("-" * 50)
    status_counts = pred["Processing_Status"].value_counts()
    success = int(status_counts.get("success", 0))
    retry_success = int(status_counts.get("retry_success", 0))
    failed = int(pred["Processing_Status"].astype(str).str.startswith("failed").sum())
    print(f"Success:                         {success}")
    print(f"Retry success:                    {retry_success}")
    print(f"Failed:                           {failed}")
    print(f"Prediction rows:                  {len(pred)}")
    print(f"Expected columns present:         {sum(c in pred.columns for c in REQUIRED_PREDICTION_COLUMNS)}/{len(REQUIRED_PREDICTION_COLUMNS)}")

    # NLP OUTPUT validation
    print("\nNLP OUTPUT")
    print("-" * 50)
    invalid_topic_rows = 0
    invalid_sentiment_rows = 0
    missing_evidence_rows = 0
    bad_confidence_rows = 0
    suspicious_drug_hits: list[str] = []

    for _, row in pred.iterrows():
        issues = validate_prediction_row(row)
        if any("invalid topic" in i for i in issues):
            invalid_topic_rows += 1
        if any("invalid sentiment" in i for i in issues):
            invalid_sentiment_rows += 1
        if any("empty evidence" in i for i in issues):
            missing_evidence_rows += 1
        if any("confidence" in i for i in issues):
            bad_confidence_rows += 1
        for issue in issues:
            if issue.startswith("suspicious drug term:"):
                suspicious_drug_hits.append(f"Record {row['Record_ID']}: {issue.split(': ', 1)[1]}")

    print(f"Invalid topics:                   {invalid_topic_rows}")
    print(f"Invalid sentiments:               {invalid_sentiment_rows}")
    print(f"Missing evidence:                 {missing_evidence_rows}")
    print(f"Confidence outside [0,1]:         {bad_confidence_rows}")
    print(f"Suspicious drug terms:            {len(suspicious_drug_hits)}")
    if suspicious_drug_hits[:10]:
        for hit in suspicious_drug_hits[:10]:
            print(f"  - {hit}")
        if len(suspicious_drug_hits) > 10:
            print(f"  ... and {len(suspicious_drug_hits) - 10} more")

    # CONTEXT
    print("\nCONTEXT")
    print("-" * 50)
    if "Context_Used" in pred.columns:
        twitter = pred[pred["Source"] == "Twitter"]
        ctx_true = int(twitter["Context_Used"].astype(str).str.lower().isin({"true", "1"}).sum())
        ctx_false = len(twitter) - ctx_true
        print(f"Twitter records using reply context: {ctx_true}")
        print(f"Twitter records without context:     {ctx_false}")
    else:
        print("Context_Used column not found.")

    # ENTITY AUDIT summary
    print("\nENTITY AUDIT")
    print("-" * 50)
    all_drugs = []
    all_studies = []
    for _, row in pred.iterrows():
        all_drugs.extend(parse_list_field(row.get("Drugs")))
        all_studies.extend(parse_list_field(row.get("Study_Names")))
    print(f"Unique drugs extracted:           {len(set(all_drugs))}")
    print(f"Unique study names extracted:     {len(set(all_studies))}")
    print(f"Records with empty study names:   {(pred['Study_Names'].apply(lambda x: len(parse_list_field(x)) == 0)).sum()}")

    # PASS/FAIL
    print("\n" + "=" * 50)
    checks = [
        len(pred) == EXPECTED_COMBINED_ROWS,
        media_n == EXPECTED_MEDIA_ROWS,
        twitter_n == EXPECTED_TWITTER_ROWS,
        pred["Record_ID"].duplicated().sum() == 0,
        failed == 0,
        invalid_topic_rows == 0,
        invalid_sentiment_rows == 0,
        missing_evidence_rows == 0,
        bad_confidence_rows == 0,
    ]
    status = "PASS" if all(checks) else "REVIEW REQUIRED"
    print(f"STATUS: {status}")
    print("=" * 50)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
