"""Exploratory visualizations for NLP results."""

from __future__ import annotations

import json
from collections import Counter
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parse_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def explode_topics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Expand multi-label topics into one row per topic for plotting."""
    rows = []
    for _, row in results_df.iterrows():
        topics = _parse_list(row.get("Topics", []))
        sentiments = _parse_list(row.get("Topic_Sentiments", []))
        for i, topic in enumerate(topics):
            rows.append(
                {
                    "Record_ID": row["Record_ID"],
                    "Source": row.get("Source", ""),
                    "Topic": topic,
                    "Sentiment": sentiments[i] if i < len(sentiments) else "neutral",
                }
            )
    return pd.DataFrame(rows)


def plot_source_distribution(combined_df: pd.DataFrame, ax=None):
    """Bar chart: Media vs Twitter record counts."""
    ax = ax or plt.gca()
    counts = combined_df["Source"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Set2")
    ax.set_title("Source Distribution")
    ax.set_xlabel("Source")
    ax.set_ylabel("Record Count")
    return ax


def plot_topic_distribution(topic_df: pd.DataFrame, ax=None):
    """Bar chart: topic frequency across all records."""
    ax = ax or plt.gca()
    counts = topic_df["Topic"].value_counts()
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_title("Topic Distribution")
    ax.set_xlabel("Count")
    return ax


def plot_sentiment_by_topic(topic_df: pd.DataFrame, ax=None):
    """Stacked bar: sentiment breakdown per topic."""
    ax = ax or plt.gca()
    pivot = (
        topic_df.groupby(["Topic", "Sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.plot(kind="barh", stacked=True, ax=ax, colormap="RdYlGn")
    ax.set_title("Sentiment by Topic")
    ax.set_xlabel("Count")
    return ax


def plot_media_vs_twitter_topics(topic_df: pd.DataFrame, ax=None):
    """Grouped comparison of topic proportions by source."""
    ax = ax or plt.gca()
    cross = pd.crosstab(topic_df["Source"], topic_df["Topic"], normalize="index") * 100
    cross.T.plot(kind="bar", ax=ax, rot=45)
    ax.set_title("Topic Distribution: Media vs Twitter (%)")
    ax.set_ylabel("Percentage")
    ax.legend(title="Source")
    return ax


def plot_entity_frequency(results_df: pd.DataFrame, column: str, top_n: int = 10, ax=None):
    """Bar chart of top-N most frequent entities in a column."""
    ax = ax or plt.gca()
    counter: Counter = Counter()
    for val in results_df[column]:
        for entity in _parse_list(val):
            counter[entity] += 1

    top = counter.most_common(top_n)
    if not top:
        ax.set_title(f"Top {top_n} {column} (no data)")
        return ax

    labels, counts = zip(*top)
    sns.barplot(x=list(counts), y=list(labels), ax=ax, palette="mako")
    ax.set_title(f"Top {top_n} {column.replace('_', ' ')}")
    ax.set_xlabel("Frequency")
    return ax


def create_dashboard(combined_df: pd.DataFrame, results_df: pd.DataFrame, save_path=None):
    """Generate a 2×2 visualization dashboard."""
    topic_df = explode_topics(results_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plot_source_distribution(combined_df, ax=axes[0, 0])
    plot_topic_distribution(topic_df, ax=axes[0, 1])
    plot_sentiment_by_topic(topic_df, ax=axes[1, 0])
    plot_media_vs_twitter_topics(topic_df, ax=axes[1, 1])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
