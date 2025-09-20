# pre_tune_app/llm/utils/tokens.py
from __future__ import annotations
from typing import Any
from google import genai

def count_tokens_safe(client: genai.Client, model: str, contents: Any) -> int:
    """
    Safe wrapper for models.count_tokens(). Returns -1 on failure.
    """
    try:
        resp = client.models.count_tokens(model=model, contents=contents)
        return int(getattr(resp, "total_tokens", -1))
    except Exception:
        return -1

def trim_by_chars_approx(text: str, max_tokens: int) -> str:
    """
    Roughly ~4 chars/token. Use greedy char trim as a safe fallback.
    """
    approx_chars = max_tokens * 4
    if not isinstance(text, str):
        text = str(text)
    return text if len(text) <= approx_chars else text[:approx_chars]
