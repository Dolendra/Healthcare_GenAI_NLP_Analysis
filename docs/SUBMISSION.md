# Submission Checklist — Task 1

Use this checklist before submitting Assignment A / Task 1.

## 1. Run automated audit

```powershell
python scripts/submission_audit.py
python scripts/smoke_test_notebook.py   # verifies analysis path without API
```

Expected: all checks pass.

## 2. Core pipeline scripts (inference — run once)

```powershell
python scripts/run_preprocessing.py
python scripts/run_pilot.py
python scripts/run_batch.py
```

**Do not re-run the full batch** unless you intentionally want to regenerate predictions.

## 3. Validation & analytics (no Gemini API required)

```powershell
python scripts/quality_check.py      # expect STATUS: PASS
python scripts/apply_ground_truth.py
python scripts/evaluate.py             # writes evaluation_report.txt
python scripts/analyze_results.py    # writes analytics_report.txt + charts
```

## 4. Notebook

Open `notebooks/Healthcare_GenAI_NLP_Analysis_NDST.ipynb` and **Restart Kernel → Run All**.

When `outputs/predictions_NDST.csv` exists, the notebook runs in **analysis mode** (no API calls for pilot/batch).

## 5. Files to submit

| File | Notes |
|------|-------|
| `notebooks/Healthcare_GenAI_NLP_Analysis_NDST.ipynb` | Main deliverable |
| `outputs/predictions_NDST.csv` | Required output CSV (100 rows × 21 columns) |
| GitHub repo link | Optional but recommended |

## 6. Files that must NOT be committed

```
.env
outputs/predictions_NDST.csv      # gitignored — submit separately
outputs/pilot_results.csv
outputs/checkpoints/
outputs/*.txt
outputs/charts/
```

## 7. Measured evaluation metrics (verify against evaluation_report.txt)

| Task | Baseline F1 | Gemini F1 |
|------|------------:|----------:|
| Topics | 0.531 | 0.883 |
| Drugs | n/a | 0.738 |
| Diseases | n/a | 0.612 |
| Study Names | n/a | 0.741 |
| Sentiment | 0.950 | 0.905 |

## 8. Key business narrative (verify against analytics_report.txt)

- **Media** is more clinically oriented (OS 18.2%, safety topics 25.8%)
- **Twitter** is more opinion-oriented (General Opinion 38.6%)
- All **50/50** Twitter records used reply context
- **100/100** records processed (95 success + 5 retry_success)

## 9. Error analysis — do NOT edit predictions

Keep `predictions_NDST.csv` as raw Gemini output. Use Record 27 (`EV-302`, `TAR-200` hallucination) as an error-analysis example in the notebook.

## 10. Final sign-off

- [ ] Quality check: PASS
- [ ] Evaluation report generated
- [ ] Analytics report + charts generated
- [ ] Notebook runs top-to-bottom in analysis mode
- [ ] README wording matches validation approach
- [ ] `.env` not in git
- [ ] `predictions_NDST.csv` ready for separate upload
