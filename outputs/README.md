# Generated Outputs

This folder holds files produced when you run the pipeline locally.

| File | How to generate |
|------|-----------------|
| `standardized_dataset.csv` | `python scripts/run_preprocessing.py` |
| `pilot_results.csv` | `python scripts/run_pilot.py` |
| `predictions_B210212.csv` | `python scripts/run_batch.py` |
| `manual_validation.csv` | `python scripts/create_validation_sample.py` then `python scripts/apply_ground_truth.py` |
| Ground truth labels | `data/validation_ground_truth.json` (applied to manual_validation.csv) |
| Quality report | `python scripts/quality_check.py` |
| Evaluation metrics | `python scripts/evaluate.py` |
| Analytics + charts | `python scripts/analyze_results.py` |
| `checkpoints/` | Created automatically during batch processing |
| `*.png` | Created by visualization cells in the notebook |

Generated outputs are intentionally **gitignored** (except this README).

Do not commit `.env` or API keys.
