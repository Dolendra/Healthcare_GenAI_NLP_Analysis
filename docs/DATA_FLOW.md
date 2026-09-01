# Data Flow — Step by Step

This document traces **one record** from raw Excel to final CSV row.

---

## Stage 1: Raw Input

### Media file (`Media & Research Articles data.xlsx`)

Confirmed columns (50 rows):

| unique_id | Article title | Content | Source | Source Type | Article link | Published date |
|-----------|---------------|---------|--------|-------------|--------------|----------------|

### Twitter file (`Twitter Posts Data.xlsx`)

Confirmed columns (50 rows):

| unique_id | HCP Handle | Posts | replied_to_tweet | Source | Source Type | Post link | Published date |
|-----------|------------|-------|------------------|--------|-------------|-----------|----------------|

---

## Stage 2: Column Standardization

`preprocessing.py` renames columns using maps in `config.py`:

```
"Article title"  →  Title
"Content"        →  Body
"Source"         →  Original_Source   (publisher preserved)
"Posts"          →  Body
"replied_to_tweet" → replied_to_tweet  (Twitter reply context)
```

Missing columns are added as empty (NaN).

---

## Stage 3: Source Attribution

```python
media_df["Source"] = "Media"
twitter_df["Source"] = "Twitter"
```

Required by assignment.

---

## Stage 4: Combined Text

```python
# Media: Title + Body joined
Combined = "KEYNOTE-189 Trial Shows OS Benefit... The phase 3 KEYNOTE-189 trial..."

# Twitter: Body only (no title)
Combined = "Excited about the new OS data for pembrolizumab!"
```

**Important:** We do NOT lowercase, stem, or remove stopwords before sending to Gemini.

---

## Stage 5: Consolidation

```python
combined_df = pd.concat([media_df, twitter_df])
combined_df.insert(0, "Record_ID", range(1, N+1))
```

Result schema:

| Record_ID | Title | Body | Source | Combined | Text_Length | Word_Count |
|-----------|-------|------|--------|----------|-------------|------------|

---

## Stage 6: LLM Processing

For each row, `gemini_client.py` sends `Combined` text to Gemini with:

- System instruction (healthcare extraction expert)
- Few-shot examples (3 annotated samples)
- Structured output schema (`NLPResult`)

### Example Gemini response (JSON):

```json
{
  "drugs": ["pembrolizumab"],
  "diseases": ["non-small cell lung cancer"],
  "study_names": ["KEYNOTE-189"],
  "topics": [
    {
      "topic": "Overall Survival (OS)",
      "sentiment": "positive",
      "evidence": "significant improvement in overall survival",
      "confidence": 0.92
    },
    {
      "topic": "Safety-Side Effects",
      "sentiment": "negative",
      "evidence": "nausea, fatigue, and rash were more common",
      "confidence": 0.88
    }
  ]
}
```

---

## Stage 7: Validation

`validation.py` checks:

- All topics ∈ allowed 7-topic list
- All sentiments ∈ {positive, negative, neutral}
- Confidence ∈ [0.0, 1.0]
- Evidence not empty
- Entity fields are lists

Invalid responses trigger retry (up to 3 attempts).

---

## Stage 8: DataFrame Flattening

Multi-label topics are stored as parallel lists:

| Column | Value |
|--------|-------|
| Drugs | `["pembrolizumab"]` |
| Diseases | `["non-small cell lung cancer"]` |
| Study_Names | `["KEYNOTE-189"]` |
| Topics | `["Overall Survival (OS)", "Safety-Side Effects"]` |
| Topic_Sentiments | `["positive", "negative"]` |
| Evidence | `["significant improvement...", "nausea, fatigue..."]` |
| Confidence_Scores | `[0.92, 0.88]` |
| Model_Confidence_Scores | `[0.92, 0.88]` |
| Model | `gemini-3.5-flash-lite` |
| Processing_Status | `success` |

---

## Stage 9: CSV Export

Lists converted to JSON strings for clean CSV:

```python
results_df["Drugs"] = results_df["Drugs"].apply(json.dumps)
```

Output: `outputs/predictions_NDST.csv`

---

## Stage 10: Analytics

`visualization.py` explodes multi-label topics for charts:

```
1 record with 2 topics → 2 rows in topic_df
```

Enables:
- Topic frequency bar chart
- Sentiment by topic stacked bar
- Media vs Twitter topic comparison
- Top 10 drugs/diseases/studies

---

## Checkpoint Resume Flow

If processing stops at record 1200/2000:

1. `outputs/checkpoints/latest_checkpoint.json` has records 1–1200
2. Re-run batch processing cell
3. `gemini_client.py` skips completed Record_IDs
4. Continues from 1201

No duplicate API calls, no wasted cost.
