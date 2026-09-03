"""
Thin wrapper around the Google Gen AI SDK (google-genai) so the rest of the
codebase never has to think about SDK details, JSON-mode quirks, or retries.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from google import genai
from google.genai import types

from config import settings


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set — see .env.example. "
                "The LLM-powered agents cannot run without it."
            )
        self._client = genai.Client(api_key=self.api_key)

    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> str:
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                return (resp.text or "").strip()
            except Exception as e:  # noqa: BLE001 - want to retry on any transient error
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Gemini text generation failed after {max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------
    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> Any:
        """Ask Gemini for JSON and parse it robustly (JSON mode + fallback cleanup)."""
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
        )
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                raw = (resp.text or "").strip()
                return _parse_json_loose(raw)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Gemini JSON generation failed after {max_retries} attempts: {last_err}")


def _parse_json_loose(raw: str) -> Any:
    """Gemini's JSON mode is usually clean, but strip code fences / stray text defensively."""
    text = raw.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort: grab the largest {...} or [...] block
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise
