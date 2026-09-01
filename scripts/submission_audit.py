#!/usr/bin/env python3
"""Final pre-submission audit — runs checks and prints a pass/fail checklist."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EXPECTED_COMBINED_ROWS,
    EXPECTED_MEDIA_ROWS,
    EXPECTED_TWITTER_ROWS,
    PREDICTIONS_FILE,
    STUDENT_ID,
)

REQUIRED_SCRIPTS = [
    "quality_check.py",
    "evaluate.py",
    "analyze_results.py",
    "apply_ground_truth.py",
    "run_batch.py",
]

REQUIRED_SOURCE = [
    "src/config.py",
    "src/gemini_client.py",
    "src/prompts.py",
    "src/validation.py",
    "src/evaluation.py",
    "src/analytics.py",
]

GITIGNORED_SENSITIVE = [".env"]
GITIGNORED_OUTPUTS = [
    "outputs/predictions_NDST.csv",
    "outputs/checkpoints",
]


def check_file(path: Path, label: str) -> tuple[bool, str]:
    ok = path.exists()
    return ok, f"{'PASS' if ok else 'FAIL'}  {label}: {path}"


def main() -> int:
    checks: list[tuple[bool, str]] = []

    # Structure
    for rel in REQUIRED_SCRIPTS:
        checks.append(check_file(PROJECT_ROOT / "scripts" / rel, f"Script {rel}"))
    for rel in REQUIRED_SOURCE:
        checks.append(check_file(PROJECT_ROOT / rel, f"Module {rel}"))

    checks.append(check_file(PROJECT_ROOT / "data" / "validation_ground_truth.json", "Ground truth JSON"))
    checks.append(check_file(PROJECT_ROOT / "notebooks" / f"Healthcare_GenAI_NLP_Analysis_{STUDENT_ID}.ipynb", "Notebook"))
    checks.append(check_file(PROJECT_ROOT / "docs" / "SUBMISSION.md", "Submission guide"))

    # Predictions
    if PREDICTIONS_FILE.exists():
        import pandas as pd

        pred = pd.read_csv(PREDICTIONS_FILE)
        media_n = int((pred["Source"] == "Media").sum())
        twitter_n = int((pred["Source"] == "Twitter").sum())
        failed = int(pred["Processing_Status"].astype(str).str.startswith("failed").sum())
        checks.append((len(pred) == EXPECTED_COMBINED_ROWS, f"{'PASS' if len(pred)==100 else 'FAIL'}  Predictions rows: {len(pred)}/100"))
        checks.append((media_n == EXPECTED_MEDIA_ROWS, f"{'PASS' if media_n==50 else 'FAIL'}  Media rows: {media_n}/50"))
        checks.append((twitter_n == EXPECTED_TWITTER_ROWS, f"{'PASS' if twitter_n==50 else 'FAIL'}  Twitter rows: {twitter_n}/50"))
        checks.append((failed == 0, f"{'PASS' if failed==0 else 'FAIL'}  Failed predictions: {failed}"))
        checks.append((len(pred.columns) == 21, f"{'PASS' if len(pred.columns)==21 else 'FAIL'}  Columns: {len(pred.columns)}/21"))
    else:
        checks.append((False, f"FAIL  Predictions file missing: {PREDICTIONS_FILE}"))

    # Reports (optional but recommended)
    for name in ["evaluation_report.txt", "analytics_report.txt"]:
        p = PROJECT_ROOT / "outputs" / name
        checks.append((p.exists(), f"{'PASS' if p.exists() else 'WARN'}  Report: {name}"))

    chart = PROJECT_ROOT / "outputs" / "charts" / "final_charts_dashboard.png"
    checks.append((chart.exists(), f"{'PASS' if chart.exists() else 'WARN'}  Chart dashboard"))

    # .env must not be tracked
    try:
        tracked = subprocess.run(
            ["git", "ls-files", ".env"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        env_tracked = bool(tracked.stdout.strip())
        checks.append((not env_tracked, f"{'PASS' if not env_tracked else 'FAIL'}  .env not in git"))
    except FileNotFoundError:
        checks.append((True, "SKIP  git not available for .env check"))

    print("=" * 50)
    print("SUBMISSION AUDIT")
    print("=" * 50)
    for ok, msg in checks:
        print(msg)

    passed = sum(1 for ok, _ in checks if ok)
    failed = sum(1 for ok, _ in checks if not ok)
    print("=" * 50)
    print(f"Result: {passed} passed, {failed} failed/warn")
    print("See docs/SUBMISSION.md for the full checklist.")
    print("=" * 50)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
