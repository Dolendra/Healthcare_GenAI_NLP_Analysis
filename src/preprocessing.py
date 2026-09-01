"""Data loading, profiling, cleaning, and consolidation for media + Twitter datasets."""

from __future__ import annotations

import re
from typing import Dict, Optional

import pandas as pd

from .config import (
    EXPECTED_COMBINED_ROWS,
    EXPECTED_MEDIA_ROWS,
    EXPECTED_TWITTER_ROWS,
    MEDIA_COLUMN_MAP,
    MEDIA_FILE,
    TARGET_COLUMNS,
    TWITTER_COLUMN_MAP,
    TWITTER_FILE,
)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a per-column data quality summary."""
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
            "Missing %": (df.isna().mean() * 100).round(2).values,
            "Unique": df.nunique().values,
        }
    )


def _rename_columns(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    """Rename columns using a mapping; only renames columns that exist."""
    rename = {col: column_map[col] for col in df.columns if col in column_map}
    return df.rename(columns=rename)


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Add missing columns as empty strings so downstream code always has the schema."""
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = ""
    return out


def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def combine_text(title, body) -> str:
    """Merge title and body into Combined — the assignment-required field."""
    title = _safe_text(title)
    body = _safe_text(body)
    return " ".join(x for x in [title, body] if x)


def normalize_whitespace(text: str) -> str:
    """Normalize line endings and collapse excessive blank lines / spaces."""
    text = _safe_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_consecutive_duplicate_paragraphs(text: str) -> str:
    """Drop only consecutive exact-duplicate paragraphs (common scrape artifact)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return text.strip()

    deduped: list[str] = []
    for paragraph in paragraphs:
        if not deduped or paragraph != deduped[-1]:
            deduped.append(paragraph)
    return "\n\n".join(deduped)


def remove_duplicated_document_half(text: str) -> str:
    """
    If the document is the same block concatenated twice, keep one copy.

    Conservative: only collapses an exact 50/50 duplicate of the full text.
    Legitimate repeated phrases elsewhere are left untouched.
    """
    stripped = text.strip()
    if len(stripped) < 200:
        return stripped

    midpoint = len(stripped) // 2
    first = stripped[:midpoint].strip()
    second = stripped[midpoint:].strip()
    if first and first == second:
        return first
    return stripped


def clean_body_text(text: str) -> str:
    """Conservative NLP-oriented cleanup: whitespace + scrape duplication only."""
    cleaned = normalize_whitespace(text)
    cleaned = remove_duplicated_document_half(cleaned)
    cleaned = remove_consecutive_duplicate_paragraphs(cleaned)
    return cleaned


def build_contextual_text(combined: str, replied_to_tweet) -> str:
    """
    Combined is Title + Body (assignment field).

    Contextual_Text adds replied_to_tweet when present so short reply tweets
    keep the medical thread they are responding to.
    """
    combined = _safe_text(combined)
    reply = clean_body_text(_safe_text(replied_to_tweet))
    if not reply:
        return combined
    if reply in combined:
        return combined
    return f"{combined}\n\n[In reply to]\n{reply}".strip()


def load_media(path: Optional[str] = None) -> pd.DataFrame:
    """Load and standardize the media/research articles dataset."""
    path = path or str(MEDIA_FILE)
    df = pd.read_excel(path)
    df = _rename_columns(df, MEDIA_COLUMN_MAP)
    df = _ensure_columns(df, ["unique_id", "Title", "Body", "Original_Source", "Source Type", "Link", "Published date"])

    df["Body_Raw"] = df["Body"].map(_safe_text)
    df["Title"] = df["Title"].map(_safe_text)
    df["Body"] = df["Body_Raw"].map(clean_body_text)
    df["Source"] = "Media"
    df["Text_Type"] = "Article"
    df["HCP Handle"] = ""
    df["replied_to_tweet"] = ""
    df["Combined"] = df.apply(lambda x: combine_text(x["Title"], x["Body"]), axis=1)
    df["Contextual_Text"] = df["Combined"]
    return df


def load_twitter(path: Optional[str] = None) -> pd.DataFrame:
    """Load and standardize the Twitter posts dataset."""
    path = path or str(TWITTER_FILE)
    df = pd.read_excel(path)
    df = _rename_columns(df, TWITTER_COLUMN_MAP)
    df = _ensure_columns(
        df,
        [
            "unique_id",
            "Body",
            "Original_Source",
            "Source Type",
            "Link",
            "Published date",
            "HCP Handle",
            "replied_to_tweet",
        ],
    )

    df["Title"] = ""
    df["Body_Raw"] = df["Body"].map(_safe_text)
    df["Body"] = df["Body_Raw"].map(clean_body_text)
    df["replied_to_tweet"] = df["replied_to_tweet"].map(_safe_text)
    df["Source"] = "Twitter"
    df["Text_Type"] = "Tweet"
    df["Combined"] = df["Body"]
    df["Contextual_Text"] = df.apply(
        lambda x: build_contextual_text(x["Combined"], x["replied_to_tweet"]),
        axis=1,
    )
    return df


def consolidate(media_df: pd.DataFrame, twitter_df: pd.DataFrame) -> pd.DataFrame:
    """Merge both datasets into a unified schema with Record_ID."""
    keep = [
        "unique_id",
        "Title",
        "Body",
        "Body_Raw",
        "Combined",
        "Contextual_Text",
        "Source",
        "Text_Type",
        "Original_Source",
        "Source Type",
        "Link",
        "Published date",
        "HCP Handle",
        "replied_to_tweet",
    ]

    media_clean = _ensure_columns(media_df, keep)[keep]
    twitter_clean = _ensure_columns(twitter_df, keep)[keep]

    combined = pd.concat([media_clean, twitter_clean], ignore_index=True)
    if len(combined) != len(media_df) + len(twitter_df):
        raise AssertionError("Row count mismatch after merge")

    combined.insert(0, "Record_ID", range(1, len(combined) + 1))

    combined["Text_Length"] = combined["Combined"].fillna("").str.len()
    combined["Word_Count"] = combined["Combined"].fillna("").str.split().str.len()
    combined["Contextual_Length"] = combined["Contextual_Text"].fillna("").str.len()
    combined["Contextual_Word_Count"] = (
        combined["Contextual_Text"].fillna("").str.split().str.len()
    )

    if len(media_df) == EXPECTED_MEDIA_ROWS and len(twitter_df) == EXPECTED_TWITTER_ROWS:
        if len(combined) != EXPECTED_COMBINED_ROWS:
            raise AssertionError(f"Expected {EXPECTED_COMBINED_ROWS} combined rows, got {len(combined)}")
        source_counts = combined["Source"].value_counts()
        if source_counts.get("Media", 0) != EXPECTED_MEDIA_ROWS:
            raise AssertionError("Media row count after merge is incorrect")
        if source_counts.get("Twitter", 0) != EXPECTED_TWITTER_ROWS:
            raise AssertionError("Twitter row count after merge is incorrect")

    ordered = [col for col in TARGET_COLUMNS if col in combined.columns]
    extras = [col for col in combined.columns if col not in ordered]
    return combined[ordered + extras]


def load_and_consolidate(
    media_path: Optional[str] = None,
    twitter_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end: load both datasets and return (media, twitter, combined)."""
    media_df = load_media(media_path)
    twitter_df = load_twitter(twitter_path)
    combined_df = consolidate(media_df, twitter_df)
    return media_df, twitter_df, combined_df


def profile_dataset(df: pd.DataFrame, name: str) -> dict:
    """Return a quick profiling dict for documentation / logging."""
    return {
        "name": name,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "duplicates": int(df.duplicated().sum()),
        "missing_by_column": df.isna().sum().to_dict(),
    }


def select_pilot_records(combined_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Build a 10-record pilot that is not a blind random sample.

    Preference order:
    1. CheckMate-901 / OS+PFS clinical trial article
    2. USPSTF / screening general-healthcare article
    3. Longest media article
    4. Short tweet
    5. Reply tweet with replied_to_tweet context
    Then fill remaining slots with 5 Media + 5 Twitter.
    """
    selected_ids: list[int] = []

    def add_ids(record_ids, limit: int = 1) -> None:
        added = 0
        for record_id in record_ids:
            if record_id in selected_ids:
                continue
            selected_ids.append(int(record_id))
            added += 1
            if added >= limit or len(selected_ids) >= n:
                break

    text = (
        combined_df["Title"].fillna("")
        + " "
        + combined_df["Body"].fillna("")
        + " "
        + combined_df["unique_id"].fillna("").astype(str)
    ).str.lower()

    add_ids(combined_df.loc[text.str.contains("checkmate-901|checkmate 901", regex=True), "Record_ID"])
    add_ids(combined_df.loc[text.str.contains("uspstf"), "Record_ID"])
    add_ids(
        combined_df[combined_df["Source"] == "Media"].nlargest(1, "Word_Count")["Record_ID"]
    )
    add_ids(
        combined_df[
            (combined_df["Source"] == "Twitter") & (combined_df["Word_Count"] <= 12)
        ]["Record_ID"]
    )
    add_ids(
        combined_df[
            (combined_df["Source"] == "Twitter")
            & (combined_df["replied_to_tweet"].fillna("").str.len() > 40)
        ]["Record_ID"]
    )

    for source in ["Media", "Twitter"]:
        needed = 5 - sum(
            1
            for rid in selected_ids
            if combined_df.loc[combined_df["Record_ID"] == rid, "Source"].iloc[0] == source
        )
        if needed <= 0:
            continue
        pool = combined_df[
            (combined_df["Source"] == source)
            & (~combined_df["Record_ID"].isin(selected_ids))
        ]
        for record_id in pool["Record_ID"].head(needed).tolist():
            selected_ids.append(record_id)

    selected_ids = selected_ids[:n]
    return (
        combined_df[combined_df["Record_ID"].isin(selected_ids)]
        .sort_values("Record_ID")
        .reset_index(drop=True)
    )
