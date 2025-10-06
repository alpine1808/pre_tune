# pre_tune_app/error_handles/errors.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Optional, Any, Union

try:
    from google.genai import errors as genai_errors  # type: ignore
except Exception:
    genai_errors = None  # optional


class GeminiErrorType(str, Enum):
    RATE_LIMIT = "RATE_LIMIT"
    UNAVAILABLE = "UNAVAILABLE"
    BAD_REQUEST = "BAD_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    SERVER_ERROR = "SERVER_ERROR"
    CLIENT_ERROR = "CLIENT_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class GeminiErrorInfo:
    # Giữ code dạng int (nếu ép được); classify sẽ vẫn chấp nhận dạng str khi so sánh
    code: Optional[int] = None
    status: Optional[str] = None
    message: Optional[str] = None


@dataclass
class GeminiDecision:
    sleep: float = 0.0
    switch_model: bool = False


class ErrorHandler(Protocol):
    def extract_info(self, exc: BaseException) -> GeminiErrorInfo: ...
    def classify(self, info: GeminiErrorInfo) -> GeminiErrorType: ...
    def make_decision(
        self, info: GeminiErrorInfo, attempt: int, max_retries: int, models: int, exc: BaseException
    ) -> GeminiDecision: ...


class GeminiErrorHandler:
    """
    Handler lỗi cho Gemini:
      - Trích xuất code/status/message từ exception SDK
      - Phân loại lỗi rộng (bao cả RATE_LIMIT_EXCEEDED, 4xx/5xx tổng quát)
      - Ra quyết định backoff/switch, tôn trọng RetryInfo nếu có
    """
    def __init__(self, base_backoff_sec: float, jitter_sec: float):
        self._base = float(base_backoff_sec)
        self._jitter = float(jitter_sec)

    # ---------- Static helpers (API tương thích + tiện dụng) ----------
    @staticmethod
    def _parse_retry_delay_seconds(exc: BaseException) -> Optional[float]:
        """
        Đọc google.rpc.RetryInfo từ exc.response_json.error.details nếu có.
        Hỗ trợ chuỗi '3s' / '250ms'.
        """
        try:
            data = getattr(exc, "response_json", None) or {}
            details = (data.get("error", {}) or {}).get("details", []) or []
            for d in details:
                if isinstance(d, dict) and str(d.get("@type", "")).endswith("google.rpc.RetryInfo"):
                    rd = d.get("retryDelay")
                    if isinstance(rd, str):
                        s = rd.strip().lower()
                        if s.endswith("ms"):
                            return float(s[:-2]) / 1000.0
                        if s.endswith("s"):
                            return float(s[:-1])
                        # fallback: số thuần tính là giây
                        try:
                            return float(s)
                        except Exception:
                            pass
                    # { "seconds": 3, "nanos": 200000000 } dạng protobuf-json
                    if isinstance(rd, dict):
                        sec = float(rd.get("seconds", 0.0) or 0.0)
                        nanos = float(rd.get("nanos", 0.0) or 0.0)
                        return sec + nanos / 1e9
        except Exception:
            pass
        return None

    # alias công khai cho code cũ nếu có gọi thẳng
    parse_retry_delay_seconds = _parse_retry_delay_seconds

    @staticmethod
    def _str_code(info: GeminiErrorInfo) -> str:
        return "" if info.code is None else str(info.code)

    @staticmethod
    def is_rate_limit(obj: Union[GeminiErrorInfo, BaseException]) -> bool:
        if isinstance(obj, BaseException):
            h = GeminiErrorHandler(1.0, 0.0)
            info = h.extract_info(obj)
        else:
            info = obj
        code_s = GeminiErrorHandler._str_code(info)
        status = (info.status or "").upper()
        return code_s == "429" or status in {"RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED"}

    @staticmethod
    def is_unavailable(obj: Union[GeminiErrorInfo, BaseException]) -> bool:
        if isinstance(obj, BaseException):
            h = GeminiErrorHandler(1.0, 0.0)
            info = h.extract_info(obj)
        else:
            info = obj
        code_s = GeminiErrorHandler._str_code(info)
        status = (info.status or "").upper()
        return code_s == "503" or status in {"UNAVAILABLE", "TRY_AGAIN_LATER"}

    @staticmethod
    def is_non_retryable(err_type: GeminiErrorType) -> bool:
        return err_type in (GeminiErrorType.BAD_REQUEST, GeminiErrorType.NOT_FOUND)

    # ---------- Core API ----------
    def extract_info(self, exc: BaseException) -> GeminiErrorInfo:
        code = None
        status = None
        message = str(exc)

        if genai_errors is not None and isinstance(
            exc, (getattr(genai_errors, "ClientError", Exception), getattr(genai_errors, "ServerError", Exception))
        ):
            try:
                payload = getattr(exc, "response_json", None) or {}
                err = payload.get("error", {}) or {}
                raw_code = err.get("code", None)
                # ép code → int nếu có thể
                try:
                    code = int(raw_code)
                except Exception:
                    code = None
                st = err.get("status", None)
                status = st.upper() if isinstance(st, str) else None
                message = err.get("message") or message
            except Exception:
                pass

        return GeminiErrorInfo(code=code, status=status, message=message)

    def classify(self, info: GeminiErrorInfo) -> GeminiErrorType:
        code_s = self._str_code(info)
        status = (info.status or "").upper()

        # Rate limit
        if code_s == "429" or status in {"RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED"}:
            return GeminiErrorType.RATE_LIMIT
        # Unavailable
        if code_s == "503" or status in {"UNAVAILABLE", "TRY_AGAIN_LATER"}:
            return GeminiErrorType.UNAVAILABLE
        # Bad request
        if code_s == "400" or status == "INVALID_ARGUMENT":
            return GeminiErrorType.BAD_REQUEST
        # Not found
        if code_s == "404" or status == "NOT_FOUND":
            return GeminiErrorType.NOT_FOUND
        # Server side
        if code_s.startswith("5"):
            return GeminiErrorType.SERVER_ERROR
        # Other client 4xx
        if code_s.startswith("4"):
            return GeminiErrorType.CLIENT_ERROR
        return GeminiErrorType.UNKNOWN

    def make_decision(
        self,
        info: GeminiErrorInfo,
        attempt: int,
        max_retries: int,
        models: int,
        exc: BaseException,
    ) -> GeminiDecision:
        et = self.classify(info)

        # ========== chính sách mặc định ==========
        # backoff mũ; UNAVAILABLE mạnh hơn
        base = self._base * ((2.0 if et == GeminiErrorType.UNAVAILABLE else 1.5) ** max(0, attempt))
        sleep = float(base)
        switch = False

        # honour RetryInfo nếu là rate-limit và server có đề xuất delay
        if et == GeminiErrorType.RATE_LIMIT:
            retry_s = self._parse_retry_delay_seconds(exc)
            if retry_s is not None:
                sleep = float(retry_s)

        # khuyến nghị switch model luân phiên khi có nhiều model
        if models > 1:
            if et in (GeminiErrorType.RATE_LIMIT, GeminiErrorType.UNAVAILABLE):
                switch = (attempt % 2 == 1)  # xen kẽ
            elif et in (GeminiErrorType.SERVER_ERROR, GeminiErrorType.UNKNOWN, GeminiErrorType.CLIENT_ERROR):
                switch = (attempt % 3 == 2)

        # BAD_REQUEST/NOT_FOUND → để runner quyết định không retry (early return)
        if et in (GeminiErrorType.BAD_REQUEST, GeminiErrorType.NOT_FOUND):
            sleep = min(sleep, 0.8)

        return GeminiDecision(sleep=sleep, switch_model=switch)


# --------- Factory: đọc tham số từ AppConfig ---------
def make_error_handler_from_cfg(cfg) -> GeminiErrorHandler:
    return GeminiErrorHandler(
        base_backoff_sec=float(getattr(cfg, "base_backoff_sec", 1.0) or 1.0),
        jitter_sec=float(getattr(cfg, "jitter_sec", 0.35) or 0.35),
    )
