#!/usr/bin/env python3
"""Retry only failed records from predictions CSV (same frozen prompt)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GEMINI_MODEL, PREDICTIONS_FILE, STANDARDIZED_FILE
from src.gemini_client import GeminiClient, _build_result_record

RETRY_DELAY = 5.0  # respect free-tier 15 req/min limit


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS_FILE)
    combined = pd.read_csv(STANDARDIZED_FILE)

    list_cols = [
        "Drugs", "Diseases", "Study_Names", "Topics",
        "Topic_Sentiments", "Evidence", "Model_Confidence_Scores",
    ]
    for col in list_cols:
        if col in predictions.columns:
            predictions[col] = predictions[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
            )

    failed_mask = predictions["Processing_Status"].astype(str).str.startswith("failed")
    failed_ids = predictions.loc[failed_mask, "Record_ID"].tolist()

    if not failed_ids:
        print("No failed records to retry.")
        return

    print(f"Retrying {len(failed_ids)} failed records: {failed_ids}")
    print(f"Waiting 60s for rate-limit window to reset...")
    time.sleep(60)

    client = GeminiClient(model=GEMINI_MODEL)

    for rid in failed_ids:
        row_dict = combined[combined["Record_ID"] == rid].iloc[0].to_dict()
        result, status, attempts, error = client.analyze_record_with_retry(row_dict)
        record = _build_result_record(row_dict, result, status, attempts, error, GEMINI_MODEL)

        idx = predictions.index[predictions["Record_ID"] == rid][0]
        for key, value in record.items():
            predictions.at[idx, key] = value

        print(f"Record {rid}: {status} (attempts={attempts})")
        time.sleep(RETRY_DELAY)

    list_cols = [
        "Drugs", "Diseases", "Study_Names", "Topics",
        "Topic_Sentiments", "Evidence", "Model_Confidence_Scores",
    ]
    export_df = predictions.copy()
    for col in list_cols:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )

    export_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nUpdated: {PREDICTIONS_FILE}")
    print(predictions["Processing_Status"].value_counts().to_string())


if __name__ == "__main__":
    main()
