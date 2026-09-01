#!/usr/bin/env python3
"""Apply validation_ground_truth.json labels to manual_validation.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "validation_ground_truth.json"
VALIDATION_FILE = PROJECT_ROOT / "outputs" / "manual_validation.csv"


def _serialize_list(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def _serialize_sentiments(sentiments: dict[str, str]) -> str:
    return json.dumps(sentiments, ensure_ascii=False)


def main() -> int:
    if not GROUND_TRUTH_FILE.exists():
        print(f"ERROR: {GROUND_TRUTH_FILE} not found.")
        return 1
    if not VALIDATION_FILE.exists():
        print(f"ERROR: {VALIDATION_FILE} not found. Run create_validation_sample.py first.")
        return 1

    labels = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    manual = pd.read_csv(VALIDATION_FILE)

    for col in [
        "Expected_Drugs",
        "Expected_Diseases",
        "Expected_Study_Names",
        "Expected_Topics",
        "Expected_Sentiments",
    ]:
        manual[col] = manual[col].astype("object")

    updated = 0
    for idx, row in manual.iterrows():
        record_id = str(int(row["Record_ID"]))
        if record_id not in labels:
            continue
        gt = labels[record_id]
        manual.at[idx, "Expected_Drugs"] = _serialize_list(gt.get("Expected_Drugs", []))
        manual.at[idx, "Expected_Diseases"] = _serialize_list(gt.get("Expected_Diseases", []))
        manual.at[idx, "Expected_Study_Names"] = _serialize_list(gt.get("Expected_Study_Names", []))
        manual.at[idx, "Expected_Topics"] = _serialize_list(gt.get("Expected_Topics", []))
        manual.at[idx, "Expected_Sentiments"] = _serialize_sentiments(gt.get("Expected_Sentiments", {}))
        updated += 1

    manual.to_csv(VALIDATION_FILE, index=False)
    print(f"Applied ground truth to {updated}/{len(manual)} records -> {VALIDATION_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
