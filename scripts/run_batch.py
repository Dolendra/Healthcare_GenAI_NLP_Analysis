#!/usr/bin/env python3
"""Run full 100-record Gemini batch with frozen prompt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GEMINI_MODEL, PREDICTIONS_FILE, STANDARDIZED_FILE
from src.gemini_client import GeminiClient, process_dataframe
from src.preprocessing import load_and_consolidate


def main() -> None:
    if STANDARDIZED_FILE.exists():
        combined_df = pd.read_csv(STANDARDIZED_FILE)
        print(f"Loaded: {STANDARDIZED_FILE} ({len(combined_df)} rows)")
    else:
        _, _, combined_df = load_and_consolidate()
        combined_df.to_csv(STANDARDIZED_FILE, index=False)

    client = GeminiClient(model=GEMINI_MODEL)
    print(f"Model: {GEMINI_MODEL}")
    print("Starting full batch — prompt is frozen; do not edit prompts during this run.\n")

    results_df = process_dataframe(combined_df, client, model=GEMINI_MODEL)

    print("\nProcessing status:")
    print(results_df["Processing_Status"].value_counts().to_string())

    export_df = results_df.copy()
    list_cols = [
        "Drugs", "Diseases", "Study_Names", "Topics",
        "Topic_Sentiments", "Evidence", "Model_Confidence_Scores",
    ]
    for col in list_cols:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nSaved: {PREDICTIONS_FILE}")
    print(f"Shape: {export_df.shape}")


if __name__ == "__main__":
    main()
