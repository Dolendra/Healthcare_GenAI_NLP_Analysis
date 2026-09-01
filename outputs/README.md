# Generated Outputs

This folder holds files produced when you run the pipeline locally.

| File | How to generate |
|------|-----------------|
| `standardized_dataset.csv` | `python scripts/run_preprocessing.py` |
| `pilot_results.csv` | `python scripts/run_pilot.py` |
| `predictions_NDST.csv` | Run notebook Section 18 or `process_dataframe()` |
| `checkpoints/` | Created automatically during batch processing |
| `*.png` | Created by visualization cells in the notebook |

Generated outputs are intentionally **gitignored** (except this README).

Do not commit `.env` or API keys.
