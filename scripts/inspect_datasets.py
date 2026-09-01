#!/usr/bin/env python3
"""
Inspect assignment Excel datasets — run this FIRST when you receive the real files.

Usage:
  python scripts/inspect_datasets.py
  python scripts/inspect_datasets.py --media path/to/media.xlsx --twitter path/to/twitter.xlsx

Prints columns, row counts, missing values, duplicates, and sample rows.
Use the output to update column mappings in src/config.py if needed.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEDIA = PROJECT_ROOT / "data" / "Media & Research Articles data.xlsx"
DEFAULT_TWITTER = PROJECT_ROOT / "data" / "Twitter Posts Data.xlsx"


def inspect(path: Path, name: str) -> None:
    print("=" * 70)
    print(f"DATASET: {name}")
    print(f"PATH:    {path}")
    print("=" * 70)

    if not path.exists():
        print(f"  FILE NOT FOUND — place your file at: {path}\n")
        return

    df = pd.read_excel(path)

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isna().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    print(f"\nText length stats (first object column):")
    for col in df.select_dtypes(include="object").columns:
        lengths = df[col].fillna("").astype(str).str.len()
        print(f"  {col}: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.0f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect assignment Excel datasets")
    parser.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument("--twitter", type=Path, default=DEFAULT_TWITTER)
    args = parser.parse_args()

    inspect(args.media, "Media & Research Articles")
    inspect(args.twitter, "Twitter Posts")


if __name__ == "__main__":
    main()
