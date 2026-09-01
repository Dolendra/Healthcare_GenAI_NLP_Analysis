# Notebook Walkthrough

File: `notebooks/Healthcare_GenAI_NLP_Analysis_NDST.ipynb`

Each section maps to a notebook heading. Run cells **in order**.

---

## Section 01 — Executive Summary

**What:** Markdown intro — project title, objective, model choice.  
**Why:** Reviewer sees purpose immediately.

---

## Section 02 — Business Objective

**What:** Explains the business value of healthcare media intelligence.  
**Why:** Shows you understand *why*, not just *how*.

---

## Section 03 — Dataset Description

**What:** Table describing both Excel files and their roles.  
**Why:** Documents input sources per assignment requirements.

---

## Section 04 — Technology Stack

**What:** ASCII tree of Python libraries and their roles.  
**Why:** Quick reference for reviewers and interview prep.

---

## Section 05 — Import Libraries

**What:** Installs packages + imports from `src/` modules.  
**Action:** Sets project paths so notebook finds `data/` and `src/`.

---

## Section 06 — Load Data

**What:** `pd.read_excel()` for both files.  
**Output:** Shape and `head()` preview.

---

## Section 07 — Data Exploration

**What:** `.info()`, `.describe()`, column lists.  
**Why:** Understand data before cleaning (assignment best practice).

---

## Section 08 — Data Quality Assessment

**What:** `quality_report()` function — missing %, duplicates.  
**Output:** Quality tables for both datasets.

---

## Section 09 — Data Cleaning

**What:** Handle obvious issues (empty rows, whitespace trim).  
**Note:** Minimal cleaning — preserve text for LLM.

---

## Section 10 — Schema Standardization

**What:** Column renaming via `preprocessing.load_media()` / `load_twitter()`.  
**Action:** If inspect script shows different columns, update `config.py` first.

---

## Section 11 — Dataset Consolidation

**What:** `pd.concat()` + `Record_ID` assignment.  
**Validation:** Assert row count equals sum of both sources.

---

## Section 12 — Combined Text Creation

**What:** `combine_text(Title, Body)` → `Combined` column.  
**Also:** Text_Length and Word_Count stats; flag short texts (< 5 words).

---

## Section 13 — NLP Task Definition

**What:** Documents 3 NLP tasks + 7 topic categories + multi-label assumption.

---

## Section 14 — Gemini API Configuration

**What:** Load API key from environment, create client.  
**⚠️ Never hardcode your key.**

---

## Section 15 — Structured Output Schema

**What:** Shows `NLPResult` and `TopicSentiment` Pydantic models.  
**Why:** Demonstrates structured output design.

---

## Section 16 — Prompt Engineering

**What:** Displays the full prompt template + few-shot examples.  
**Tip:** Edit `src/prompts.py` if pilot results need improvement.

---

## Section 17 — Pilot Test

**What:** Process a deliberately selected 10-record pilot (CheckMate-901, USPSTF, long article, short tweet, reply-context tweet) via `select_pilot_records()`.
**⚠️ Review before running full batch.**

---

## Section 18 — Batch NLP Processing

**What:** Full dataset processing with tqdm progress bar.  
**Features:** Retry, checkpointing, rate-limit delay.  
**Time:** Depends on dataset size (~0.5–2 sec/record).

---

## Section 19 — Entity Extraction Results

**What:** Top drugs, diseases, study names frequency tables.

---

## Section 20 — Topic Classification Results

**What:** Topic count distribution, bar chart.

---

## Section 21 — Topic-Level Sentiment

**What:** Sentiment breakdown per topic, stacked bar chart.

---

## Section 22 — Validation & Error Handling

**What:** Processing status summary, failed record review.  
**Also:** Keyword baseline vs Gemini comparison on sample.

---

## Section 23 — Exploratory Analysis

**What:** 2×2 dashboard — source, topics, sentiment, media vs Twitter.

---

## Section 24 — Key Insights

**What:** Business insights derived from actual data (not invented).  
**Template provided — fill after seeing your results.**

---

## Section 25 — Export Results

**What:** JSON-serialize list columns → save `predictions_[ID].csv`.

---

## Section 26 — Assumptions

**What:** Documents design choices (multi-label, topic-level sentiment, etc.).

---

## Section 27 — Limitations

**What:** LLM hallucination risk, API cost, English-only, etc.

---

## Section 28 — Conclusion

**What:** Summary of findings and recommended next steps.

---

## Tips for Running

| Situation | Action |
|-----------|--------|
| First time | Run sections 05→17, review pilot, then 18→25 |
| Crashed mid-batch | Re-run section 18 only (auto-resumes) |
| Wrong columns | Run `inspect_datasets.py`, fix `config.py`, restart from section 06 |
| Colab | Upload `data/` + `src/`, set secret for API key |
