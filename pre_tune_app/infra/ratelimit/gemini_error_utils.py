from __future__ import annotations

import random
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

_LOG = logging.getLogger(__name__)

try:
    from google.genai import errors as genai_errors
except Exception:  # pragma: no cover
    genai_errors = None  # type: ignore

__all__ = [
    "GeminiErrorInfo",
    "GeminiErrorType",
    "GeminiRetryDecision",
    "GeminiErrorHandler",
]


@dataclass(frozen=True)
class GeminiErrorInfo:
    code: str = ""
    status: str = ""
    message: str = ""


class GeminiErrorType:
    RATE_LIMIT = "rate_limit"  # 429 / RESOURCE_EXHAUSTED
    BAD_REQUEST = "bad_request"  # 400 / INVALID_ARGUMENT
    NOT_FOUND = "not_found"  # 404 / NOT_FOUND
    UNAVAILABLE = "unavailable"  # 503 / UNAVAILABLE
    SERVER_ERROR = "server_error"  # other 5xx
    CLIENT_ERROR = "client_error"  # other 4xx
    UNKNOWN = "unknown"  # unclassified


@dataclass
class GeminiRetryDecision:
    error_type: str
    sleep: Optional[float] = None
    penalize: bool = False
    switch_model: bool = False
    action: str = "retry"


class GeminiErrorHandler:
    def __init__(self, base_backoff_sec: float = 1.0, jitter_sec: float = 0.0) -> None:
        self._base_backoff = float(base_backoff_sec)
        self._jitter = float(jitter_sec)

    def extract_info(self, exc: Exception) -> GeminiErrorInfo:
        if genai_errors and isinstance(exc, (genai_errors.ClientError, genai_errors.ServerError)):
            ej = getattr(exc, "response_json", None) or {}
            code = str(ej.get("error", {}).get("code", "")) or str(getattr(exc, "status_code", "") or "")
            status = (ej.get("error", {}).get("status", "") or "").upper()
            message = ej.get("error", {}).get("message", "") or str(exc)
            return GeminiErrorInfo(code=code, status=status, message=message)
        # Generic fallback
        return GeminiErrorInfo(code="", status="", message=str(exc))

    @staticmethod
    def _parse_retry_delay_seconds(exc: Exception) -> Optional[float]:
        try:
            data = getattr(exc, "response_json", None) or {}
            details = data.get("error", {}).get("details", []) or []
            for d in details:
                if d.get("@type", "").endswith("google.rpc.RetryInfo"):
                    rd = d.get("retryDelay")
                    if isinstance(rd, str):
                        if rd.endswith("s"):
                            return float(rd[:-1])
                        if rd.endswith("ms"):
                            return float(rd[:-2]) / 1000.0
        except Exception:
            pass
        return None

    def classify(self, info: GeminiErrorInfo) -> str:
        """Map error code/status to a high level GeminiErrorType."""
        code = info.code
        status = info.status
        if not code and not status:
            return GeminiErrorType.UNKNOWN
        # Rate limit detection: 429 or resource exhausted
        if (code == "429" or status in {"RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED"}):
            return GeminiErrorType.RATE_LIMIT
        # 400 invalid argument / bad request
        if code == "400" or status == "INVALID_ARGUMENT":
            return GeminiErrorType.BAD_REQUEST
        # Not found
        if code == "404" or status == "NOT_FOUND":
            return GeminiErrorType.NOT_FOUND
        # Service unavailable
        if code == "503" or status == "UNAVAILABLE":
            return GeminiErrorType.UNAVAILABLE
        # Other server errors (5xx)
        if code and code.startswith("5"):
            return GeminiErrorType.SERVER_ERROR
        # Other client errors (4xx)
        if code and code.startswith("4"):
            return GeminiErrorType.CLIENT_ERROR
        return GeminiErrorType.UNKNOWN

    def make_decision(
        self,
        error_info: GeminiErrorInfo,
        attempt: int,
        max_retries: int,
        models: int = 1,
        exc: Optional[Exception] = None,
    ) -> GeminiRetryDecision:
        err_type = self.classify(error_info)
        # Default decision: retry with no backoff and no model switch
        dec = GeminiRetryDecision(error_type=err_type, action="retry")

        # Determine if we should suggest switching models
        should_switch = (attempt >= 1 and models > 1)

        # Rate limit: penalize and backoff using RetryInfo or exponential backoff
        if err_type == GeminiErrorType.RATE_LIMIT:
            dec.penalize = True
            # Attempt to read RetryInfo from the original exception, if provided
            retry_s = None
            if exc is not None:
                retry_s = self._parse_retry_delay_seconds(exc)
            # If the server suggests a delay, honour it and add jitter
            if retry_s is not None:
                dec.sleep = retry_s + random.uniform(0.0, max(0.5, self._jitter))
            else:
                # Exponential backoff base (minimum 0.5s)
                exp = self._base_backoff * (1.5 ** attempt)
                base = max(0.5, exp)
                dec.sleep = base + random.uniform(0.0, max(0.5, self._jitter))
            dec.switch_model = should_switch
            return dec

        # Service unavailable: penalize and use stronger exponential backoff
        if err_type == GeminiErrorType.UNAVAILABLE:
            dec.penalize = True
            exp = self._base_backoff * (2.0 ** attempt)
            base = max(1.0, exp)
            dec.sleep = base + random.uniform(0.0, self._jitter)
            dec.switch_model = should_switch
            return dec

        # Bad request: usually indicates payload too large or invalid
        if err_type == GeminiErrorType.BAD_REQUEST:
            # Let caller decide how to split/trim; here we simply indicate retry
            dec.action = "bad_request"
            dec.sleep = 0.0
            dec.switch_model = False
            return dec

        # Not found: model not available or missing resource
        if err_type == GeminiErrorType.NOT_FOUND:
            # Suggest immediate switch if multiple models; fallback otherwise
            dec.action = "fallback"
            dec.switch_model = models > 1
            dec.sleep = 0.0
            return dec

        # Other server errors: exponential backoff and optional model switch
        if err_type == GeminiErrorType.SERVER_ERROR:
            exp = self._base_backoff * (1.5 ** attempt)
            dec.sleep = max(0.2, exp) + random.uniform(0.0, self._jitter)
            dec.switch_model = should_switch
            return dec

        # Other client errors: soft backoff; fallback after max retries
        if err_type == GeminiErrorType.CLIENT_ERROR:
            if attempt < max_retries - 1:
                dec.sleep = self._base_backoff * (1.5 ** attempt) + random.uniform(0.0, self._jitter)
                dec.switch_model = should_switch
                dec.action = "retry"
            else:
                dec.action = "fallback"
                dec.sleep = 0.0
                dec.switch_model = False
            return dec

        # Unknown errors: generic exponential backoff
        dec.sleep = self._base_backoff * (1.5 ** attempt) + random.uniform(0.0, self._jitter)
        dec.switch_model = should_switch
        return dec