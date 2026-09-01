# Setup Guide

## Local Setup (Windows)

```powershell
cd D:\Assignment1\Healthcare_GenAI_NLP
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=AIza...your_key_here
```

## Google Colab Setup

Upload the project folder or clone from GitHub, then in the first notebook cell:

```python
!pip install -q pandas numpy openpyxl matplotlib seaborn google-genai pydantic python-dotenv tqdm scikit-learn

from google.colab import userdata
import os
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

Add your API key in Colab: **Secrets** (key icon) → `GEMINI_API_KEY`.

Upload Excel files to `data/` or mount Google Drive.

## Get a Gemini API Key

1. Go to https://aistudio.google.com/apikey
2. Create API key (free tier available)
3. Never commit the key to git

## Verify Installation

```bash
python -c "from google import genai; print('google-genai OK')"
python scripts/inspect_datasets.py
```

## Model Selection

Default: `gemini-3.5-flash-lite` (available for new API keys; supports structured output)

Model is set in `src/config.py`:

```python
GEMINI_MODEL = "gemini-3.5-flash-lite"
```

## Processing Large Datasets

| Setting | Location | Default | Purpose |
|---------|----------|---------|---------|
| `CHECKPOINT_INTERVAL` | `src/config.py` | 10 | Save progress every N records |
| `REQUEST_DELAY` | `src/config.py` | 0.5s | Avoid rate limits |
| `MAX_RETRIES` | `src/config.py` | 3 | Retry failed API calls |

For 2000+ records, expect ~20–40 minutes depending on rate limits.
