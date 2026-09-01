#!/usr/bin/env python3
"""Create a 50-record human validation sample (25 Media + 25 Twitter)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, STANDARDIZED_FILE, VALIDATION_SAMPLE_SIZE
from src.preprocessing import select_validation_records

VALIDATION_FILE = OUTPUT_DIR / "manual_validation.csv"


def main() -> None:
    std = pd.read_csv(STANDARDIZED_FILE)
    sample = select_validation_records(std, n=VALIDATION_SAMPLE_SIZE)

    rows = []
    for _, row in sample.iterrows():
        rows.append(
            {
                "Record_ID": row["Record_ID"],
                "Source": row["Source"],
                "Text_Type": row.get("Text_Type", ""),
                "Context_Used": row.get("Context_Used", False),
                "Combined_Preview": str(row.get("Combined", ""))[:200],
                "Expected_Drugs": "",
                "Expected_Diseases": "",
                "Expected_Study_Names": "",
                "Expected_Topics": "",
                "Expected_Sentiments": "",
                "Notes": "",
            }
        )

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(VALIDATION_FILE, index=False)

    print(f"Created validation sample: {VALIDATION_FILE}")
    print(f"Records: {len(out)}")
    print(f"Media: {(out['Source'] == 'Media').sum()}")
    print(f"Twitter: {(out['Source'] == 'Twitter').sum()}")
    print(f"Record_IDs: {out['Record_ID'].tolist()}")


if __name__ == "__main__":
    main()
