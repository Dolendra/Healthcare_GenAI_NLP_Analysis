#!/usr/bin/env python3
"""Run standardization on the real assignment datasets. No Gemini calls."""

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, STANDARDIZED_FILE
from src.preprocessing import load_and_consolidate, quality_report


def main() -> None:
    media_df, twitter_df, combined_df = load_and_consolidate()

    print("=" * 70)
    print("PREPROCESSING OUTPUT")
    print("=" * 70)
    print("Media:", len(media_df))
    print("Twitter:", len(twitter_df))
    print("Combined:", len(combined_df))
    print()
    print("combined_df.columns:")
    print(combined_df.columns.tolist())
    print()
    print("groupby Source:")
    print(combined_df.groupby("Source").size())
    print()
    print("groupby Source, Text_Type:")
    print(combined_df.groupby(["Source", "Text_Type"]).size())
    print()
    print("Original_Source value counts:")
    print(combined_df["Original_Source"].value_counts())
    print()
    print("--- Combined quality ---")
    print(quality_report(combined_df).to_string(index=False))
    print()
    print("Text length by Source:")
    print(combined_df.groupby("Source")[["Word_Count", "Contextual_Word_Count"]].describe())
    print()
    print("--- head(10): Record_ID / Source / Title / Body preview / Combined preview ---")
    preview = combined_df[["Record_ID", "Source", "Title"]].head(10).copy()
    preview["Body"] = combined_df["Body"].head(10).str.slice(0, 160)
    preview["Combined"] = combined_df["Combined"].head(10).str.slice(0, 180)
    print(preview.to_string())
    print()
    print("--- Twitter sample: Body / replied_to_tweet / Combined / Contextual_Text ---")
    twitter_preview = combined_df[combined_df["Source"] == "Twitter"][
        ["Record_ID", "Body", "replied_to_tweet", "Combined", "Contextual_Text"]
    ].head(10).copy()
    for col in ["Body", "replied_to_tweet", "Combined", "Contextual_Text"]:
        twitter_preview[col] = twitter_preview[col].str.slice(0, 140)
    pd_option_width = 80
    import pandas as pd

    with pd.option_context("display.max_colwidth", pd_option_width, "display.width", 160):
        print(twitter_preview.to_string())

    replies = combined_df[
        (combined_df["Source"] == "Twitter")
        & (combined_df["replied_to_tweet"].fillna("").str.len() > 0)
    ]
    print()
    print(f"Twitter rows with replied_to_tweet: {len(replies)}")
    longer_context = (combined_df["Contextual_Word_Count"] > combined_df["Word_Count"]).sum()
    print(f"Rows where Contextual_Text is longer than Combined: {longer_context}")
    shortened = (combined_df["Body"].str.len() < combined_df["Body_Raw"].str.len()).sum()
    print(f"Rows where Body was shortened by duplicate cleanup: {shortened}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(STANDARDIZED_FILE, index=False)
    print()
    print(f"Saved standardized dataset: {STANDARDIZED_FILE}")


if __name__ == "__main__":
    main()
