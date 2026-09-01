#!/usr/bin/env python3
"""Generate the submission Jupyter notebook with all 28 sections."""

import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "Healthcare_GenAI_NLP_Analysis_NDST.ipynb"
)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.split("\n"),
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md("""# Healthcare Media & Social Intelligence using Generative AI

## Entity Extraction, Topic Classification & Topic-Level Sentiment Analysis

---

## 01. Executive Summary

This project analyzes healthcare-related media articles and Twitter posts using Generative AI to identify:

- **Drugs**
- **Diseases**
- **Study Names**
- **Healthcare topics** (7 categories)
- **Topic-level sentiment** with evidence snippets

| Setting | Value |
|---------|-------|
| Model | Gemini 2.5 Flash-Lite |
| Framework | Google GenAI SDK |
| Input Sources | Media + Twitter |
| Output | Structured CSV |

> 📖 Full project docs: see `docs/PROJECT_GUIDE.md` in the project folder."""),

    md("""## 02. Business Objective

Pharmaceutical and healthcare organizations need to monitor how their therapies, trials, and safety profiles are discussed across **formal media** and **social channels**. Manual review does not scale.

This pipeline automates:
1. Unified ingestion of articles and tweets
2. Entity extraction (drugs, diseases, trials)
3. Topic tagging against standard oncology/clinical categories
4. Sentiment analysis **per topic** (not just per document)

**Business value:** Faster signal detection, auditable predictions, cross-source comparison (Media vs Twitter)."""),

    md("""## 03. Dataset Description

| Dataset | Description | Role |
|---------|-------------|------|
| Media & Research Articles data.xlsx | Healthcare media/research content | Media source |
| Twitter Posts Data.xlsx | Individual social media posts | Twitter source |

Both are merged into a unified schema with `Source`, `Combined`, and `Record_ID` columns."""),

    md("""## 04. Technology Stack

```
TECHNOLOGY STACK

Python
│
├── Pandas          → Data manipulation
├── NumPy           → Numerical operations
├── OpenPyXL        → Excel ingestion
├── Matplotlib      → Visualization
├── Seaborn         → Statistical visualization
│
├── Google GenAI    → LLM API
├── Pydantic        → Structured output validation
└── python-dotenv   → API key management
```

**Architecture:** Two layers — deterministic data processing + probabilistic LLM extraction."""),

    code("""# 05. Import Libraries
import sys, json, os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Project paths — works in local Jupyter and Colab
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent  # running from notebooks/
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path("/content/Healthcare_GenAI_NLP")  # Colab fallback

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    STUDENT_ID, GEMINI_MODEL, TOPICS, MEDIA_FILE, TWITTER_FILE,
    PREDICTIONS_FILE, OUTPUT_DIR, PILOT_SAMPLE_SIZE,
)
from src.preprocessing import (
    quality_report, load_media, load_twitter, consolidate, profile_dataset,
)
from src.validation import NLPResult, TopicSentiment, validate_nlp_result
from src.prompts import build_extraction_prompt, SYSTEM_INSTRUCTION
from src.gemini_client import get_client, analyze_text_with_retry, process_dataframe
from src.baseline import baseline_classify_topics, baseline_sentiment
from src.visualization import (
    explode_topics, create_dashboard,
    plot_entity_frequency, plot_source_distribution,
)

load_dotenv(PROJECT_ROOT / ".env")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_colwidth", 120)
sns.set_theme(style="whitegrid")
print(f"Project root: {PROJECT_ROOT}")
print(f"Student ID:   {STUDENT_ID}")
print(f"Model:        {GEMINI_MODEL}")"""),

    md("## 06. Load Data"),
    code("""media_df = pd.read_excel(MEDIA_FILE)
twitter_df = pd.read_excel(TWITTER_FILE)

print("Media:  ", media_df.shape)
print("Twitter:", twitter_df.shape)

display(media_df.head())
display(twitter_df.head())"""),

    md("## 07. Data Exploration"),
    code("""media_df.info()
print()
twitter_df.info()"""),

    code("""print("Media columns:  ", media_df.columns.tolist())
print("Twitter columns:", twitter_df.columns.tolist())

display(media_df.describe(include="all"))
display(twitter_df.describe(include="all"))"""),

    md("## 08. Data Quality Assessment"),
    code("""display(quality_report(media_df))
display(quality_report(twitter_df))

print("Media duplicate rows:  ", media_df.duplicated().sum())
print("Twitter duplicate rows:", twitter_df.duplicated().sum())

print("\\nMedia profile:  ", profile_dataset(media_df, "Media"))
print("Twitter profile:", profile_dataset(twitter_df, "Twitter"))"""),

    md("""## 09. Data Cleaning

Minimal cleaning — we preserve semantic context for the LLM.
Only normalize whitespace and drop fully empty rows."""),
    code("""# Drop rows where all values are missing
media_df = media_df.dropna(how="all").reset_index(drop=True)
twitter_df = twitter_df.dropna(how="all").reset_index(drop=True)

# Strip whitespace from string columns
for df in [media_df, twitter_df]:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

print(f"After cleaning — Media: {media_df.shape}, Twitter: {twitter_df.shape}")"""),

    md("""## 10. Schema Standardization

Column renaming handled by `src/preprocessing.py` using maps in `src/config.py`.
If your Excel columns differ, run `python scripts/inspect_datasets.py` and update the maps."""),
    code("""media_std = load_media(str(MEDIA_FILE))
twitter_std = load_twitter(str(TWITTER_FILE))

print("Standardized media columns:  ", media_std.columns.tolist())
print("Standardized twitter columns:", twitter_std.columns.tolist())
display(media_std.head(2))
display(twitter_std.head(2))"""),

    md("## 11. Dataset Consolidation"),
    code("""combined_df = consolidate(media_std, twitter_std)

assert len(combined_df) == len(media_std) + len(twitter_std)
print(combined_df["Source"].value_counts())
print(f"\\nCombined shape: {combined_df.shape}")
display(combined_df.head())"""),

    md("""## 12. Combined Text Creation

Already applied during standardization. Below: text quality checks."""),
    code("""combined_df["Word_Count"].describe()

short_text = combined_df[combined_df["Word_Count"] < 5]
print(f"Records with < 5 words: {len(short_text)}")
if len(short_text):
    display(short_text[["Record_ID", "Source", "Combined"]])"""),

    md("""## 13. NLP Task Definition

### Task 1 — Entity Extraction
Extract **drugs**, **diseases**, and **study names** explicitly mentioned.

### Task 2 — Topic Classification (multi-label)
A document may have **multiple topics**. Allowed categories:"""),
    code("""for i, topic in enumerate(TOPICS, 1):
    print(f"  {i}. {topic}")"""),
    md("""### Task 3 — Topic-Level Sentiment
Each topic gets its own sentiment: `positive`, `negative`, or `neutral`.
Includes an **evidence snippet** and **model confidence score** for auditability."""),

    md("""## 14. Gemini API Configuration

Set `GEMINI_API_KEY` in `.env` (local) or Colab Secrets.
**Never hardcode your API key in this notebook.**"""),
    code("""client = get_client()
print("Gemini client initialized successfully.")
print(f"Using model: {GEMINI_MODEL}")"""),

    md("## 15. Structured Output Schema"),
    code("""print("NLPResult schema fields:")
for name, field in NLPResult.model_fields.items():
    print(f"  {name}: {field.annotation}")

print("\\nTopicSentiment schema fields:")
for name, field in TopicSentiment.model_fields.items():
    print(f"  {name}: {field.annotation}")"""),

    md("## 16. Prompt Engineering"),
    code("""sample_text = combined_df.iloc[0]["Combined"]
print("SYSTEM INSTRUCTION:")
print(SYSTEM_INSTRUCTION)
print("\\n" + "="*60)
print("USER PROMPT (sample):")
print(build_extraction_prompt(sample_text[:500]))"""),

    md("""## 17. Pilot Test

Process 10 records first. Review output quality before the full batch."""),
    code("""pilot = combined_df.sample(n=min(PILOT_SAMPLE_SIZE, len(combined_df)), random_state=42)
pilot_results = []

for _, row in pilot.iterrows():
    result, status = analyze_text_with_retry(client, row["Combined"])
    pilot_results.append({
        "Record_ID": row["Record_ID"],
        "Source": row["Source"],
        "Text_Preview": row["Combined"][:120] + "...",
        "Status": status,
        "Drugs": result.drugs if result else [],
        "Diseases": result.diseases if result else [],
        "Study_Names": result.study_names if result else [],
        "Topics": [t.topic for t in result.topics] if result else [],
        "Sentiments": [t.sentiment for t in result.topics] if result else [],
        "Evidence": [t.evidence for t in result.topics] if result else [],
    })

pilot_df = pd.DataFrame(pilot_results)
display(pilot_df)"""),

    md("""## 18. Batch NLP Processing

Full dataset with retry, rate-limit delay, and checkpointing.
If interrupted, re-run this cell — it resumes from the last checkpoint."""),
    code("""results_df = process_dataframe(
    combined_df,
    client,
    model=GEMINI_MODEL,
)

print(f"Processed: {len(results_df)} records")
print(results_df["Processing_Status"].value_counts())"""),

    md("## 19. Entity Extraction Results"),
    code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, ["Drugs", "Diseases", "Study_Names"]):
    plot_entity_frequency(results_df, col, top_n=10, ax=ax)
plt.tight_layout()
plt.show()"""),

    md("## 20. Topic Classification Results"),
    code("""topic_df = explode_topics(results_df)
print(topic_df["Topic"].value_counts())

fig, ax = plt.subplots(figsize=(10, 6))
topic_df["Topic"].value_counts().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Topic Classification Distribution")
ax.set_xlabel("Count")
plt.tight_layout()
plt.show()"""),

    md("## 21. Topic-Level Sentiment"),
    code("""sentiment_pivot = pd.crosstab(topic_df["Topic"], topic_df["Sentiment"])
display(sentiment_pivot)

fig, ax = plt.subplots(figsize=(10, 6))
sentiment_pivot.plot(kind="barh", stacked=True, ax=ax, colormap="RdYlGn")
ax.set_title("Sentiment by Topic")
ax.set_xlabel("Count")
plt.tight_layout()
plt.show()"""),

    md("## 22. Validation & Error Handling"),
    code("""# Processing status summary
print("Processing Status:")
print(results_df["Processing_Status"].value_counts())

failed = results_df[results_df["Processing_Status"] != "success"]
if len(failed):
    print(f"\\nFailed records ({len(failed)}):")
    display(failed[["Record_ID", "Source", "Processing_Status"]])

# Baseline vs Gemini comparison on pilot sample
print("\\n--- Baseline vs Gemini (pilot sample) ---")
for _, row in pilot.iterrows():
    text = row["Combined"]
    baseline_topics = baseline_classify_topics(text)
    gemini_row = pilot_df[pilot_df["Record_ID"] == row["Record_ID"]].iloc[0]
    print(f"Record {row['Record_ID']}:")
    print(f"  Baseline: {baseline_topics}")
    print(f"  Gemini:   {gemini_row['Topics']}")"""),

    md("## 23. Exploratory Analysis"),
    code("""dashboard_path = OUTPUT_DIR / "dashboard.png"
fig = create_dashboard(combined_df, results_df, save_path=dashboard_path)
plt.show()
print(f"Dashboard saved to: {dashboard_path}")

# Media vs Twitter sentiment
print("\\nSentiment by Source:")
print(pd.crosstab(topic_df["Source"], topic_df["Sentiment"]))"""),

    md("""## 24. Key Insights

> Fill these after reviewing your actual results. Examples of insight types:

- Which topics dominate Media vs Twitter?
- Do social posts show more negative Safety sentiment than formal articles?
- Which drugs/studies appear most frequently?
- How many records required retry/failed?

**Only state insights supported by your data.**"""),
    code("""# Auto-generated summary statistics
print(f"Total records:     {len(results_df)}")
print(f"Success rate:      {(results_df['Processing_Status']=='success').mean():.1%}")
print(f"Unique drugs:      {topic_df['Topic'].nunique()} topics detected")
print(f"\\nTop 5 topics:")
print(topic_df["Topic"].value_counts().head())"""),

    md("## 25. Export Results"),
    code("""export_df = results_df.copy()

list_columns = ["Drugs", "Diseases", "Study_Names", "Topics",
                "Topic_Sentiments", "Evidence", "Confidence_Scores"]

for col in list_columns:
    if col in export_df.columns:
        export_df[col] = export_df[col].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else x
        )

export_df.to_csv(PREDICTIONS_FILE, index=False)
print(f"Saved: {PREDICTIONS_FILE}")
print(f"Shape: {export_df.shape}")
display(export_df.head(3))"""),

    md("""## 26. Assumptions

1. **Multi-label topics** — one document can discuss multiple healthcare topics simultaneously.
2. **Topic-level sentiment** — sentiment is assigned per topic, not per document.
3. **Explicit entities only** — the LLM is instructed not to infer unstated drugs/diseases.
4. **Minimal preprocessing** — text is sent to Gemini with original casing and punctuation to preserve meaning (e.g., "did NOT improve OS").
5. **Model confidence** — reported as a model self-assessment, not a calibrated probability.
6. **English content** — prompts and examples assume English-language input."""),

    md("""## 27. Limitations

1. **Hallucination risk** — LLMs may occasionally extract entities not in the text; mitigated by prompt rules and manual validation.
2. **API dependency** — processing requires network access and is subject to rate limits.
3. **Cost at scale** — very large datasets may exceed free tier limits.
4. **Baseline simplicity** — keyword baseline is intentionally basic; not a fair comparison to modern LLMs but useful for demonstration.
5. **No temporal analysis** — this pipeline does not track sentiment trends over time (future enhancement)."""),

    md("""## 28. Conclusion

This pipeline demonstrates a production-minded approach to healthcare NLP:

- **Deterministic** data layer (Pandas, schema validation)
- **Probabilistic** LLM layer (Gemini with structured JSON)
- **Auditable** outputs (evidence snippets, confidence scores)
- **Resilient** processing (retry, checkpointing)

**Deliverables:**
- `outputs/predictions_{STUDENT_ID}.csv`
- This notebook with full documentation in `docs/`

**Recommended next steps:** Manual validation on 50 records, compare Flash-Lite vs Flash quality, optional Streamlit dashboard."""),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": [
        {**c, "source": [line + "\n" for line in c["source"]]} for c in cells
    ],
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {NOTEBOOK_PATH}")
