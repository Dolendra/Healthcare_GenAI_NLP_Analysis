# Healthcare Media & Social Intelligence using GenAI

> Entity Extraction, Topic Classification & Topic-Level Sentiment Analysis  
> Assignment A · Task 1

## Objective

Analyze healthcare-related **media articles** and **Twitter posts** using Google Gemini to extract drugs, diseases, study names, healthcare topics, and topic-level sentiment. Output a structured CSV for downstream intelligence.

## Quick Start

```bash
pip install -r requirements.txt
copy .env.example .env          # add GEMINI_API_KEY
python scripts/generate_sample_data.py   # if you don't have real Excel files yet
jupyter notebook notebooks/Healthcare_GenAI_NLP_Analysis_B210212.ipynb
```

**Full documentation:** [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)

## Dataset

| File | Role |
|------|------|
| `data/Media & Research Articles data.xlsx` | Healthcare media/research content |
| `data/Twitter Posts Data.xlsx` | Social media posts |

## Architecture

Two-layer pipeline: deterministic data processing (Pandas) + probabilistic LLM extraction (Gemini with Pydantic validation).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full diagram.

## Technology Stack

```
Python
├── Pandas          → Data manipulation
├── NumPy           → Numerical operations
├── OpenPyXL        → Excel ingestion
├── Matplotlib      → Visualization
├── Seaborn         → Statistical visualization
├── Google GenAI    → LLM API
├── Pydantic        → Structured output validation
└── python-dotenv   → API key management
```

## NLP Tasks

### Entity Extraction
Drugs, diseases, and clinical study names explicitly mentioned in text.

### Topic Classification (multi-label)
- Efficacy-General
- Progression Free Survival (PFS)
- Overall Survival (OS)
- Safety-General
- Safety-Side Effects
- General Opinion
- Others

### Sentiment Analysis
Topic-level sentiment (positive / negative / neutral) with evidence snippets and model confidence scores.

## Model Selection

**Primary:** `gemini-3.5-flash-lite` — used because newer API keys cannot access `gemini-2.5-flash-lite`; supports structured JSON output.

## Validation Strategy

- Pydantic schema validation on every LLM response
- Retry with exponential backoff (429, timeouts)
- Checkpointing every 10 records (resume-safe)
- Keyword baseline comparison vs Gemini on a 50-record validation set
- **Manual validation:** 50 stratified records (25 Media + 25 Twitter) were manually annotated with expected entities, topics, and topic-level sentiments in `data/validation_ground_truth.json` and used as ground truth for P/R/F1 evaluation

## Output

```
outputs/predictions_B210212.csv
```

Change `STUDENT_ID` in `src/config.py` to match your assignment ID.

## Assumptions

- Multi-label topic classification (one document can have multiple topics)
- Sentiment assigned independently per topic
- Evidence snippets retained for auditability
- Minimal text preprocessing to preserve semantic context for LLM
- English-language content

## Limitations

- LLM may hallucinate entities if prompt is violated — mitigated by "explicit only" rules
- Model confidence scores are not calibrated probabilities
- API rate limits may slow large batches
- Keyword baseline is intentionally simplistic (for comparison only)

## Security

API keys are stored in `.env` (local) or Colab Secrets — never committed to git.

## Project Structure

```
Healthcare_GenAI_NLP/
├── data/           ← Excel input files
├── docs/           ← Full documentation (start here)
├── notebooks/      ← Submission notebook
├── outputs/        ← predictions CSV + checkpoints
├── scripts/        ← Dataset inspection & sample data
└── src/            ← Pipeline modules
```

## How to Run

1. Read [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)
2. Place Excel files in `data/`
3. Run `python scripts/inspect_datasets.py`
4. Set `GEMINI_API_KEY` in `.env`
5. Run preprocessing → pilot → batch:
   ```bash
   python scripts/run_preprocessing.py
   python scripts/run_pilot.py
   python scripts/run_batch.py
   ```
6. Validate and evaluate:
   ```bash
   python scripts/quality_check.py
   python scripts/create_validation_sample.py
   python scripts/apply_ground_truth.py
   python scripts/evaluate.py
   python scripts/analyze_results.py
   ```
7. Open and run the notebook for analytics and final presentation

## Submission Deliverables

| Deliverable | Location |
|-------------|----------|
| Jupyter notebook | `notebooks/Healthcare_GenAI_NLP_Analysis_B210212.ipynb` |
| Predictions CSV | `outputs/predictions_B210212.csv` (submit separately if required; gitignored) |
| Reproducible scripts | `scripts/` |
| Documentation | `docs/` + `README.md` |

**Pre-submission audit:**

```bash
python scripts/submission_audit.py
```

See [docs/SUBMISSION.md](docs/SUBMISSION.md) for the full checklist.
