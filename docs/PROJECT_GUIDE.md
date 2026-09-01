# Healthcare GenAI NLP — Start Here

> **GenAI-Powered Healthcare Media & Social Intelligence Pipeline**  
> Assignment A · Task 1

This document is your **single entry point**. Read it first, then follow the links below.

---

## What This Project Does (30-second version)

1. Loads two Excel files: **media articles** and **Twitter posts**
2. Cleans and merges them into one dataset with a `Combined` text column
3. Sends each record to **Google Gemini** for:
   - Entity extraction (drugs, diseases, study names)
   - Multi-label topic classification (7 healthcare topics)
   - Topic-level sentiment with evidence snippets
4. Validates every response with **Pydantic**
5. Exports `predictions_[YOUR_ID].csv` + visualizations + evaluation metrics

---

## Project Folder Map

```
Healthcare_GenAI_NLP/
│
├── 📖 docs/                          ← YOU ARE HERE
│   ├── PROJECT_GUIDE.md              ← This file (start here)
│   ├── SETUP.md                      ← Install & API key setup
│   ├── ARCHITECTURE.md               ← System design & two-layer pipeline
│   ├── DATA_FLOW.md                  ← Step-by-step data transformations
│   └── NOTEBOOK_WALKTHROUGH.md       ← What each notebook section does
│
├── 📊 data/                          ← Put your Excel files here
│   ├── Media & Research Articles data.xlsx
│   └── Twitter Posts Data.xlsx
│
├── 📓 notebooks/
│   └── Healthcare_GenAI_NLP_Analysis_B210212.ipynb   ← Main submission notebook
│
├── 📤 outputs/
│   ├── predictions_B210212.csv          ← Final deliverable
│   └── checkpoints/                  ← Resume-safe API progress
│
├── 🔧 src/                           ← Reusable Python modules
│   ├── config.py                     ← Paths, model name, topic list
│   ├── preprocessing.py              ← Load, clean, merge datasets
│   ├── prompts.py                    ← LLM prompt + few-shot examples
│   ├── gemini_client.py              ← API calls, retry, checkpointing
│   ├── validation.py                 ← Pydantic schemas
│   ├── baseline.py                   ← Keyword baseline for comparison
│   └── visualization.py              ← Charts & dashboard
│
├── 🛠 scripts/
│   ├── inspect_datasets.py           ← Run FIRST with real Excel files
│   └── generate_sample_data.py       ← Creates demo data for testing
│
├── .env.example                      ← Copy to .env, add API key
├── requirements.txt
└── README.md
```

---

## Quick Start (5 steps)

### Step 1 — Install dependencies

```bash
cd D:\Assignment1\Healthcare_GenAI_NLP
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Add your data

Copy the assignment Excel files into `data/`:

```
data/Media & Research Articles data.xlsx
data/Twitter Posts Data.xlsx
```

**Don't have the files yet?** Run the sample data generator:

```bash
python scripts/generate_sample_data.py
```

### Step 3 — Inspect your data (important!)

```bash
python scripts/inspect_datasets.py
```

If column names differ from `Title` / `Body`, update the mappings in `src/config.py`.

### Step 4 — Set your Gemini API key

```bash
copy .env.example .env
# Edit .env → GEMINI_API_KEY=your_key_from_google_ai_studio
```

Get a free key at: https://aistudio.google.com/apikey

### Step 5 — Run the notebook

```bash
jupyter notebook notebooks/Healthcare_GenAI_NLP_Analysis_B210212.ipynb
```

Run all cells top-to-bottom. The notebook will:
- Profile and clean data
- Run a 10-record pilot test
- Process the full dataset (with checkpoints)
- Generate charts and export CSV

---

## Change Your Student ID

Edit **one line** in `src/config.py`:

```python
STUDENT_ID = "B210212"   # ← change to your ID
```

This updates the output filename to `predictions_YOUR_ID.csv`.

---

## Documentation Index

| Document | When to read |
|----------|-------------|
| [SETUP.md](SETUP.md) | First-time install, Colab setup, API key |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Interview prep — system design |
| [DATA_FLOW.md](DATA_FLOW.md) | Understand every transformation step |
| [NOTEBOOK_WALKTHROUGH.md](NOTEBOOK_WALKTHROUGH.md) | Section-by-section notebook guide |

---

## What Makes This Project Stand Out

| Feature | Why it matters |
|---------|---------------|
| Structured JSON output (Pydantic) | Reliable parsing, no regex on LLM text |
| Multi-label topics | One article can discuss OS + Safety |
| Topic-level sentiment | Not just "document is positive" |
| Evidence snippets | Auditable predictions — not a black box |
| Retry + checkpointing | Production-grade error handling |
| Keyword baseline comparison | Shows *why* GenAI beats rules |
| Manual validation framework | Precision/recall/F1 on 50-record sample |
| Media vs Twitter analysis | Cross-source business insights |

---

## Submission Checklist

- [ ] `notebooks/Healthcare_GenAI_NLP_Analysis_[ID].ipynb` — well-commented
- [ ] `outputs/predictions_[ID].csv` — final predictions
- [ ] API key **not** in notebook or CSV
- [ ] Assumptions & limitations sections filled in
- [ ] Pilot test reviewed before full batch run

---

## Need Help?

1. **Column names don't match?** → Run `scripts/inspect_datasets.py`, update `src/config.py`
2. **API rate limits?** → Increase `REQUEST_DELAY` in `src/config.py`
3. **Notebook crashed mid-run?** → Re-run batch cell; checkpoints auto-resume
4. **Poor extraction quality?** → Review pilot test, edit prompts in `src/prompts.py`
