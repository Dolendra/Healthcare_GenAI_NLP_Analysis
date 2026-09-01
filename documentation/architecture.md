# Architecture Diagram

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the full write-up.

## Pipeline Overview

```
                 HEALTHCARE GENAI NLP PIPELINE
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
   Media Articles                            Twitter Posts
          │                                       │
          └───────────────────┬───────────────────┘
                              ▼
                     Data Profiling
                              │
                              ▼
                    Quality Assessment
                              │
                              ▼
                  Schema Standardization
                              │
                              ▼
                     Source Attribution
                              │
                              ▼
                    Text Consolidation
                              │
                              ▼
                       Gemini API
                              │
            ┌─────────────────┼──────────────────┐
            │                 │                  │
            ▼                 ▼                  ▼
       Entity NER        Topic Classification   Sentiment
            │                 │                  │
            └─────────────────┼──────────────────┘
                              ▼
                    Structured JSON
                              │
                              ▼
                     Pydantic Validation
                              │
                              ▼
                    Retry / Error Handling
                              │
                              ▼
                      Results DataFrame
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
           CSV Export    Visual Analytics   Evaluation
```

## Module Map

| File | Layer | Purpose |
|------|-------|---------|
| `src/config.py` | Config | Paths, topics, model settings |
| `src/preprocessing.py` | Deterministic | Load, clean, merge |
| `src/prompts.py` | LLM | Prompt templates |
| `src/gemini_client.py` | LLM | API + retry + checkpoints |
| `src/validation.py` | Deterministic | Pydantic schemas |
| `src/baseline.py` | Deterministic | Keyword comparison |
| `src/visualization.py` | Analytics | Charts |
