#!/usr/bin/env python3
"""Export compact validation review text for annotation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_FILE

VALIDATION_FILE = PROJECT_ROOT / "outputs" / "manual_validation.csv"
STD_FILE = PROJECT_ROOT / "outputs" / "standardized_dataset.csv"
PRED_FILE = PREDICTIONS_FILE
OUT_FILE = PROJECT_ROOT / "outputs" / "validation_compact.txt"


def main() -> None:
    manual = pd.read_csv(VALIDATION_FILE)
    std = pd.read_csv(STD_FILE)
    pred = pd.read_csv(PRED_FILE)
    ids = manual["Record_ID"].tolist()

    lines: list[str] = []
    for rid in ids:
        r = std[std.Record_ID == rid].iloc[0]
        p = pred[pred.Record_ID == rid].iloc[0]
        title = str(r.get("Title", ""))
        combined = str(r.get("Combined", ""))[:1200]
        lines.append(f"### Record {rid} ({r['Source']})")
        lines.append(f"Title: {title}")
        lines.append(f"Combined[:1200]: {combined}")
        lines.append(
            f"Gemini: drugs={p.get('Drugs')} diseases={p.get('Diseases')} "
            f"studies={p.get('Study_Names')}"
        )
        lines.append(
            f"Gemini: topics={p.get('Topics')} sentiments={p.get('Topic_Sentiments')}"
        )
        lines.append("")

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_FILE} ({len(ids)} records)")


if __name__ == "__main__":
    main()
