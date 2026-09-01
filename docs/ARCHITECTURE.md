# Architecture

## High-Level Pipeline

```
                    ┌───────────────────────────────┐
                    │        INPUT DATASETS         │
                    └───────────────┬───────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
       Media & Research Articles              Twitter Posts
          .xlsx dataset                         .xlsx dataset
                   │                                 │
                   └────────────────┬────────────────┘
                                    │
                                    ▼
                         DATA PROFILING & QUALITY
                                    │
                                    ▼
                       COLUMN STANDARDIZATION
                                    │
                                    ▼
                         ADD SOURCE COLUMN
                                    │
                                    ▼
                       DATASET CONSOLIDATION
                                    │
                                    ▼
                       COMBINED TEXT CREATION
                                    │
                                    ▼
                    ┌─────────────────────────┐
                    │       GEMINI API        │
                    │  Entity Extraction      │
                    │  Topic Classification   │
                    │  Sentiment Analysis     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                         JSON VALIDATION (Pydantic)
                                 │
                                 ▼
                       RETRY / CHECKPOINTING
                                 │
                                 ▼
                       NLP RESULTS DATAFRAME
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                 Entities      Topics     Sentiment
                    │            │            │
                    └────────────┼────────────┘
                                 ▼
                  predictions_[ID].csv + Charts
```

## Two-Layer Design

### Layer 1 — Deterministic (no LLM)

| Module | Responsibility |
|--------|---------------|
| `preprocessing.py` | Load Excel, rename columns, merge, combine text |
| `validation.py` | Pydantic schema enforcement on LLM output |
| `config.py` | Constants, topic enums, file paths |

Everything in Layer 1 is **reproducible** — same input always gives same output.

### Layer 2 — Probabilistic (LLM)

| Module | Responsibility |
|--------|---------------|
| `prompts.py` | Domain-specific instructions + few-shot examples |
| `gemini_client.py` | API calls, structured JSON, retry, checkpoints |

Layer 2 output is **validated** by Layer 1 before entering the final DataFrame.

## Why Gemini?

| Requirement | Gemini Solution |
|-------------|----------------|
| Assignment allows Gemini | ✅ Explicitly permitted |
| Entity extraction | Structured JSON output |
| Topic classification | Enum-constrained schema |
| High volume | Flash-Lite optimized for classification |
| Cost | Free tier available |
| Validation | Native Pydantic integration |

## Structured Output Flow

```
Text Record
    ↓
Gemini API (response_schema=NLPResult)
    ↓
JSON response
    ↓
Pydantic validate (topics, sentiments, confidence 0-1)
    ↓
Flatten to DataFrame columns
    ↓
CSV export (lists as JSON strings)
```

## Error Handling

```
API Request
    │
    ▼
 Success? ──Yes──→ Validate JSON → Save to checkpoint
    │
    No (429, timeout, invalid JSON)
    │
    ▼
 Exponential backoff (2s, 4s, 8s)
    │
    ▼
 Max retries reached? → Log failure, empty result, continue
```

Checkpoints saved every 10 records to `outputs/checkpoints/`.

## Module Dependencies

```
config.py
    ↑
    ├── preprocessing.py
    ├── prompts.py
    ├── validation.py
    ├── baseline.py
    └── gemini_client.py → prompts, validation, config
            ↑
    visualization.py
```

## What We Deliberately Did NOT Use

| Technology | Reason |
|-----------|--------|
| RAG / Vector DB | Task is extract-from-text, not search-and-answer |
| LangChain | Unnecessary abstraction for single LLM call |
| FastAPI / React | Analytics assignment, not web app |
| Aggressive NLP preprocessing | Would destroy semantic context for LLM |
