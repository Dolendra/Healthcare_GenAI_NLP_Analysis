#!/usr/bin/env python3
"""Generate Media vs Twitter analytics, charts, and business insights report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import analytics_report, format_insights_table
from src.config import OUTPUT_DIR, PREDICTIONS_FILE, STANDARDIZED_FILE
from src.visualization import create_final_charts, explode_topics

REPORT_FILE = OUTPUT_DIR / "analytics_report.txt"
CHARTS_DIR = OUTPUT_DIR / "charts"


def main() -> int:
    if not PREDICTIONS_FILE.exists():
        print(f"ERROR: {PREDICTIONS_FILE} not found. Run run_batch.py first.")
        return 1

    results_df = pd.read_csv(PREDICTIONS_FILE)
    combined_df = (
        pd.read_csv(STANDARDIZED_FILE)
        if STANDARDIZED_FILE.exists()
        else results_df[["Record_ID", "Source"]].copy()
    )

    report = analytics_report(results_df)
    topic_df = explode_topics(results_df)

    lines = [
        "=" * 50,
        "MEDIA VS TWITTER ANALYTICS REPORT",
        "=" * 50,
        "",
        "PROCESSING",
        "-" * 50,
        f"Total records:     {report['processing']['total']}",
        f"Success:           {report['processing']['success']}",
        f"Retry success:     {report['processing']['retry_success']}",
        f"Failed:            {report['processing']['failed']}",
        "",
        "CONTEXT",
        "-" * 50,
        f"Twitter with reply context: {report['context']['with_context']}",
        f"Twitter without context:    {report['context']['without_context']}",
        "",
        "TOPIC COUNTS (all records)",
        "-" * 50,
    ]
    for topic, count in sorted(report["topic_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"  {topic}: {count}")

    lines.extend(["", "TOPIC SHARE BY SOURCE (%)", "-" * 50])
    share_df = pd.DataFrame(report["topic_share_by_source"]).round(1)
    lines.append(share_df.to_string())

    lines.extend(["", "SENTIMENT BY SOURCE", "-" * 50])
    sent_df = pd.crosstab(topic_df["Source"], topic_df["Sentiment"])
    lines.append(sent_df.to_string())

    for label, key in [
        ("TOP DRUGS", "top_drugs"),
        ("TOP DISEASES", "top_diseases"),
        ("TOP STUDIES", "top_studies"),
    ]:
        lines.extend(["", label, "-" * 50])
        for name, count in report[key]:
            lines.append(f"  {name}: {count}")

    lines.extend(["", "KEY FINDINGS", "-" * 50, format_insights_table(report["insights"]), ""])
    lines.append("=" * 50)

    text = "\n".join(lines)
    print(text)
    REPORT_FILE.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved report: {REPORT_FILE}")

    plt.close("all")
    create_final_charts(combined_df, results_df, save_dir=CHARTS_DIR)
    print(f"Saved charts: {CHARTS_DIR / 'final_charts_dashboard.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
