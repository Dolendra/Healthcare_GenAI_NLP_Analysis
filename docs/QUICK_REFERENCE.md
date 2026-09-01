# Quick Reference Card

## One-Page Cheat Sheet

### Start Here
```
docs/PROJECT_GUIDE.md
```

### Run Pipeline
```powershell
cd D:\Assignment1\Healthcare_GenAI_NLP
.venv\Scripts\Activate.ps1
jupyter notebook notebooks/Healthcare_GenAI_NLP_Analysis_B210212.ipynb
```

### Before First Run
1. Put Excel files in `data/`
2. `python scripts/inspect_datasets.py`
3. Set `GEMINI_API_KEY` in `.env`
4. Change `STUDENT_ID` in `src/config.py`

### Key Files
| What | Where |
|------|-------|
| Main notebook | `notebooks/Healthcare_GenAI_NLP_Analysis_B210212.ipynb` |
| Output CSV | `outputs/predictions_B210212.csv` |
| API client | `src/gemini_client.py` |
| Prompts | `src/prompts.py` |
| Topics list | `src/config.py` → `TOPICS` |

### 7 Topic Categories
1. Efficacy-General
2. Progression Free Survival (PFS)
3. Overall Survival (OS)
4. Safety-General
5. Safety-Side Effects
6. General Opinion
7. Others

### Processing Flow
```
Excel → Clean → Merge → Combined text → Gemini → Validate → CSV
```

### If Something Breaks
| Problem | Fix |
|---------|-----|
| Wrong columns | `inspect_datasets.py` → edit `src/config.py` |
| Rate limit 429 | Increase `REQUEST_DELAY` in config |
| Crash mid-run | Re-run notebook section 18 (auto-resumes) |
| Bad extractions | Edit `src/prompts.py`, re-run pilot (section 17) |

### Interview Answer (Model Choice)
> "I selected Gemini because the assignment permits it, structured outputs suit entity extraction and constrained classification, and Flash-Lite is optimized for high-volume workloads. I separated deterministic processing from probabilistic LLM calls and added Pydantic validation, evidence snippets, and checkpointing."
