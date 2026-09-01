#!/usr/bin/env python3
"""Smoke-test the notebook analysis path (no Gemini API calls)."""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analytics import build_insights, format_insights_table
from src.config import MEDIA_FILE, PREDICTIONS_FILE, STANDARDIZED_FILE, TWITTER_FILE
from src.preprocessing import consolidate, load_media, load_twitter
from src.visualization import create_final_charts, explode_topics

sns.set_theme(style="whitegrid")


def main() -> int:
    errors: list[str] = []

    if not PREDICTIONS_FILE.exists():
        errors.append(f"Missing predictions: {PREDICTIONS_FILE}")
    if not STANDARDIZED_FILE.exists():
        errors.append(f"Missing standardized data: {STANDARDIZED_FILE}")

    if errors:
        for e in errors:
            print(f"FAIL  {e}")
        return 1

    # Data path (notebook sections 6-11)
    media_df = load_media(MEDIA_FILE)
    twitter_df = load_twitter(TWITTER_FILE)
    combined_df = consolidate(media_df, twitter_df)
    assert len(combined_df) == 100, f"Expected 100 records, got {len(combined_df)}"

    # Analysis path (notebook sections 18-24)
    results_df = pd.read_csv(PREDICTIONS_FILE)
    assert len(results_df) == 100
    assert len(results_df.columns) == 21

    topic_df = explode_topics(results_df)
    insights = build_insights(results_df, topic_df)
    assert len(insights) >= 5

    plt.close("all")
    create_final_charts(combined_df, results_df, save_dir=PROJECT_ROOT / "outputs" / "charts")

    print("PASS  Data loading + consolidation (100 records)")
    print("PASS  Predictions CSV (100 rows × 21 columns)")
    print("PASS  Topic explosion + insights generation")
    print("PASS  Chart dashboard generation")
    print("PASS  Notebook analysis path smoke test complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
