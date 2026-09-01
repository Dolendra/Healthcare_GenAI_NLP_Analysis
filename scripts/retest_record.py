#!/usr/bin/env python3
"""Re-test a single record after prompt changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GEMINI_MODEL, STANDARDIZED_FILE
from src.gemini_client import GeminiClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-test one record with Gemini")
    parser.add_argument("record_id", type=int, help="Record_ID to analyze")
    args = parser.parse_args()

    df = pd.read_csv(STANDARDIZED_FILE)
    row = df[df["Record_ID"] == args.record_id]
    if row.empty:
        raise SystemExit(f"Record_ID {args.record_id} not found")

    row_dict = row.iloc[0].to_dict()
    client = GeminiClient(model=GEMINI_MODEL)
    result, status, attempts, error = client.analyze_record_with_retry(row_dict)

    print(f"Record_ID: {args.record_id}")
    print(f"Status: {status} | Attempts: {attempts} | Error: {error or 'none'}")
    if result is None:
        return

    print(f"Drugs: {result.drugs}")
    print(f"Diseases: {result.diseases}")
    print(f"Study_Names: {result.study_names}")
    print("Topics:")
    for t in result.topics:
        print(f"  - {t.topic} | {t.sentiment} | conf={t.model_confidence} | {t.evidence[:80]}")


if __name__ == "__main__":
    main()
