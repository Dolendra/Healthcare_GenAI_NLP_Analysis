#!/usr/bin/env python3
"""Basic Gemini connectivity test — no batch processing."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gemini_client import GeminiClient


def main() -> None:
    client = GeminiClient()
    response = client.generate(
        "Explain what progression-free survival means in one sentence."
    )
    print(response.text)


if __name__ == "__main__":
    main()
