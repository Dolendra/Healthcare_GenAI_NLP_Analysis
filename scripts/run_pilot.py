#!/usr/bin/env python3
"""Run a 10-record deliberate pilot and save results for manual review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GEMINI_MODEL, OUTPUT_DIR, PILOT_SAMPLE_SIZE, STANDARDIZED_FILE
from src.gemini_client import GeminiClient, _build_result_record
from src.preprocessing import load_and_consolidate, select_pilot_records
from src.prompts import build_record_text

PILOT_OUTPUT = OUTPUT_DIR / "pilot_results.csv"


def print_pilot_record(row: dict) -> None:
    print("=" * 72)
    print(f"Record ID: {row['Record_ID']}  |  Source: {row['Source']}  |  Status: {row['Processing_Status']}")
    print(f"NLP Text Type: {row['NLP_Text_Type']}  |  Context Used: {row['Context_Used']}")
    print("-" * 72)
    print("TEXT (preview):")
    print(str(row["NLP_Text"])[:500])
    if len(str(row["NLP_Text"])) > 500:
        print("...")
    print("\nENTITIES:")
    print(f"  Drugs:   {row['Drugs']}")
    print(f"  Diseases: {row['Diseases']}")
    print(f"  Studies: {row['Study_Names']}")
    print("\nTOPICS:")
    topics = row.get("Topics", [])
    sentiments = row.get("Topic_Sentiments", [])
    evidence = row.get("Evidence", [])
    confidences = row.get("Model_Confidence_Scores", [])
    if not topics:
        print("  (none)")
    for i, topic in enumerate(topics):
        sent = sentiments[i] if i < len(sentiments) else "?"
        ev = evidence[i] if i < len(evidence) else ""
        conf = confidences[i] if i < len(confidences) else "?"
        print(f"  - {topic}")
        print(f"    Sentiment: {sent}  |  Model confidence: {conf}")
        print(f"    Evidence: {ev}")
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if STANDARDIZED_FILE.exists():
        combined_df = pd.read_csv(STANDARDIZED_FILE)
        print(f"Loaded standardized dataset: {STANDARDIZED_FILE}")
    else:
        _, _, combined_df = load_and_consolidate()
        combined_df.to_csv(STANDARDIZED_FILE, index=False)
        print(f"Built and saved standardized dataset: {STANDARDIZED_FILE}")

    pilot_df = select_pilot_records(combined_df, n=PILOT_SAMPLE_SIZE)
    print(f"Pilot records selected: {pilot_df['Record_ID'].tolist()}")
    print(f"Model: {GEMINI_MODEL}\n")

    client = GeminiClient(model=GEMINI_MODEL)
    results: list[dict] = []

    for _, row in pilot_df.iterrows():
        row_dict = row.to_dict()
        result, status, attempt_count, last_error = client.analyze_record_with_retry(row_dict)
        record = _build_result_record(
            row_dict, result, status, attempt_count, last_error, GEMINI_MODEL
        )
        results.append(record)
        print_pilot_record(record)

    pilot_results = pd.DataFrame(results)

    export_df = pilot_results.copy()
    list_cols = [
        "Drugs", "Diseases", "Study_Names", "Topics",
        "Topic_Sentiments", "Evidence", "Model_Confidence_Scores",
    ]
    for col in list_cols:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )

    export_df.to_csv(PILOT_OUTPUT, index=False)
    print(f"Saved pilot results: {PILOT_OUTPUT}")
    print(pilot_results["Processing_Status"].value_counts().to_string())


if __name__ == "__main__":
    main()
