"""Gemini API client with basic connectivity and structured NLP extraction."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm.auto import tqdm

from .config import (
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    GEMINI_MODEL,
    MAX_RETRIES,
    REQUEST_DELAY,
    RETRY_BASE_DELAY,
)
from .prompts import (
    SYSTEM_INSTRUCTION,
    build_extraction_prompt,
    build_prompt_for_record,
    build_record_text,
    nlp_text_type,
    record_uses_reply_context,
)
from .validation import NLPResult, result_to_row_dict, validate_nlp_result

load_dotenv()


class GeminiClient:
    """Thin wrapper around the Google GenAI SDK."""

    def __init__(self, model: str = GEMINI_MODEL, api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found in environment.")

        self.client = genai.Client(api_key=key)
        self.model = model

    def generate(self, prompt: str):
        """Simple text generation — used for connectivity tests."""
        return self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

    def analyze_structured(
        self,
        text: str,
        temperature: float = 0.1,
    ) -> NLPResult:
        """Call Gemini with structured JSON output validated by Pydantic."""
        prompt = build_extraction_prompt(text)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=NLPResult,
                temperature=temperature,
            ),
        )

        raw = json.loads(response.text)
        return validate_nlp_result(raw)

    def analyze_record(self, row: Mapping[str, Any]) -> NLPResult:
        """Analyze one standardized record using labelled Twitter context when needed."""
        prompt = build_prompt_for_record(row)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=NLPResult,
                temperature=0.1,
            ),
        )
        raw = json.loads(response.text)
        return validate_nlp_result(raw)

    def analyze_record_with_retry(
        self,
        row: Mapping[str, Any],
        max_retries: int = MAX_RETRIES,
    ) -> tuple[NLPResult | None, str, int, str]:
        """
        Analyze one record with retry.

        Returns: (result, status, attempt_count, last_error)
        status: success | retry_success | failed
        """
        last_error = ""

        for attempt in range(max_retries):
            try:
                result = self.analyze_record(row)
                status = "success" if attempt == 0 else "retry_success"
                return result, status, attempt + 1, ""
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))

        return None, "failed", max_retries, last_error


def get_client(api_key: Optional[str] = None) -> GeminiClient:
    """Factory for notebooks and scripts."""
    return GeminiClient(api_key=api_key)


def analyze_text(
    client: GeminiClient | genai.Client,
    text: str,
    model: str = GEMINI_MODEL,
) -> NLPResult:
    """Backward-compatible structured analysis from plain text."""
    if isinstance(client, GeminiClient):
        return client.analyze_structured(text)

    prompt = build_extraction_prompt(text)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=NLPResult,
            temperature=0.1,
        ),
    )
    return validate_nlp_result(json.loads(response.text))


def analyze_text_with_retry(
    client: GeminiClient | genai.Client,
    text: str,
    model: str = GEMINI_MODEL,
    max_retries: int = MAX_RETRIES,
) -> tuple[NLPResult | None, str, int, str]:
    """Backward-compatible retry wrapper for plain text."""
    last_error = ""
    for attempt in range(max_retries):
        try:
            result = analyze_text(client, text, model=model)
            status = "success" if attempt == 0 else "retry_success"
            return result, status, attempt + 1, ""
        except Exception as exc:
            last_error = str(exc)
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    return None, "failed", max_retries, last_error


def _save_checkpoint(results: list[dict], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _load_checkpoint(checkpoint_path: Path) -> list[dict]:
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _build_result_record(
    row: Mapping[str, Any],
    result: NLPResult | None,
    status: str,
    attempt_count: int,
    last_error: str,
    model: str,
) -> dict:
    """Build one output row with processing metadata."""
    row_dict = dict(row)
    nlp_text = build_record_text(row_dict)

    record = {
        "Record_ID": row_dict.get("Record_ID"),
        "unique_id": row_dict.get("unique_id", ""),
        "Source": row_dict.get("Source", ""),
        "Text_Type": row_dict.get("Text_Type", ""),
        "Title": row_dict.get("Title", ""),
        "Body": row_dict.get("Body", ""),
        "Combined": row_dict.get("Combined", ""),
        "NLP_Text": nlp_text,
        "NLP_Text_Type": nlp_text_type(row_dict),
        "Context_Used": record_uses_reply_context(row_dict),
        "Model": model,
        "Processing_Status": status,
        "Attempt_Count": attempt_count,
        "Last_Error": last_error,
    }

    if result is not None:
        record.update(result_to_row_dict(result))
    else:
        record.update(
            {
                "Drugs": [],
                "Diseases": [],
                "Study_Names": [],
                "Topics": [],
                "Topic_Sentiments": [],
                "Evidence": [],
                "Model_Confidence_Scores": [],
            }
        )
    return record


def process_dataframe(
    df: pd.DataFrame,
    client: GeminiClient,
    model: str = GEMINI_MODEL,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    request_delay: float = REQUEST_DELAY,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Batch-process records with retry, checkpointing, and labelled Twitter context.

    NLP input is always built via build_record_text() — not a single column name.
    """
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    master_checkpoint = checkpoint_dir / "latest_checkpoint.json"

    completed: list[dict] = _load_checkpoint(master_checkpoint)
    done_ids = {r["Record_ID"] for r in completed}

    rows_to_process = df[~df["Record_ID"].isin(done_ids)]
    total = len(df)
    already_done = len(done_ids)

    if already_done:
        print(f"Resuming: {already_done}/{total} records already processed.")

    client.model = model

    for idx, (_, row) in enumerate(
        tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Gemini NLP")
    ):
        row_dict = row.to_dict()
        result, status, attempt_count, last_error = client.analyze_record_with_retry(row_dict)

        record = _build_result_record(
            row_dict, result, status, attempt_count, last_error, model
        )
        completed.append(record)

        if request_delay > 0:
            time.sleep(request_delay)

        current_count = already_done + idx + 1
        if progress_callback:
            progress_callback(current_count, total)

        if (idx + 1) % checkpoint_interval == 0:
            _save_checkpoint(completed, master_checkpoint)
            batch_path = checkpoint_dir / f"checkpoint_{current_count}.json"
            _save_checkpoint(completed, batch_path)

    _save_checkpoint(completed, master_checkpoint)
    return pd.DataFrame(completed).sort_values("Record_ID").reset_index(drop=True)
