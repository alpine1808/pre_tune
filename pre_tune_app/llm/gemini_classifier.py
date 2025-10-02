from __future__ import annotations
import logging, time, random, json, os, unicodedata
from typing import Sequence, List

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.llm.interfaces import ICompletenessClassifier
from pre_tune_app.infra.cache.redis_cache import RedisCache
from pre_tune_app.infra.ratelimit.global_limiter import build_global_limiter
# Import centralized error handler
try:
    from pre_tune_app.infra.ratelimit.gemini_error_utils import GeminiErrorHandler
except Exception:
    GeminiErrorHandler = None  # type: ignore

_LOG = logging.getLogger(__name__)

# ---- SDK guards ----
try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    genai_errors = None  # type: ignore
    _HAS_GENAI = False

# Prompt: yêu cầu trả JSON rất ngắn
_PROMPT = (
    "Bạn là bộ phân loại câu pháp lý tiếng Việt. "
    "Hãy trả về JSON đúng định dạng {\"complete\": true|false} "
    "(không giải thích, không văn bản thừa)."
)

# Schema JSON: chỉ 1 trường boolean
if _HAS_GENAI:
    RESP_SCHEMA = types.Schema(  # type: ignore[attr-defined]
        type=types.Type.OBJECT,
        properties={"complete": types.Schema(type=types.Type.BOOLEAN)},
        required=["complete"],
    )
else:
    RESP_SCHEMA = None  # type: ignore

# ---------------- helpers ----------------
def _trim_soft(s: str, max_chars: int) -> str:
    if not isinstance(s, str):
        s = str(s)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars]

def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return " ".join(s.split())

def _heuristic_complete(s: str) -> bool:
    s = (s or "").strip()
    if not s or len(s) < 10:
        return False
    if s[-1:] in ".;:?!)]”\"'»":
        return True
    # tiêu đề pháp lý thường đủ ý
    import re
    if re.search(r"\b(Điều|Khoản|Mục)\s+\d+(\.|:)", s):
        return True
    if "..." in s or "… " in s:
        return False
    return len(s) > 30

def _build_contents(text: str, max_chars: int) -> list[dict]:
    t = _trim_soft(text, max_chars)
    prompt = f"{_PROMPT}\n---\n{t}"
    return [{"role": "user", "parts": [{"text": prompt}]}]

def _build_contents_multi(texts: List[str], max_chars: int) -> list[dict]:
    """
    Gói nhiều câu trong 1 prompt, yêu cầu trả JSON array [true/false,...] theo đúng thứ tự.
    """
    lines = []
    for i, s in enumerate(texts, 1):
        lines.append(f"{i}. {_trim_soft(s, max_chars)}")
    prompt = (
        f"{_PROMPT} Trả về JSON mảng booleans cho các câu sau theo đúng thứ tự.\n"
        "Ví dụ: [true,false,true]\n---\n" + "\n".join(lines)
    )
    return [{"role": "user", "parts": [{"text": prompt}]}]

# --------------- classifier ----------------
class GeminiCompletenessClassifier(ICompletenessClassifier):
    def __init__(self, cfg: AppConfig, model_id: str | None = None, api_key_env: str | None = None, rpm: int | None = None) -> None:
        if not _HAS_GENAI:
            raise RuntimeError("google-genai SDK không có. Vui lòng cài đặt.")
        self._cfg = cfg

        # Ưu tiên model riêng; lọc bỏ 1.5 để tránh 404 với v1beta
        primary = model_id or getattr(cfg, "classifier_model_id", None) or cfg.model_text
        m_fallback = getattr(cfg, "model_text_fallback", None)
        models = [m for m in [primary, m_fallback] if m]
        models = [m for m in models if not str(m).startswith("gemini-1.5")]
        if not models:
            models = ["gemini-2.0-flash-lite"]
        self._models_cycle: List[str] = models
        self._model_idx = 0
        self._model = self._models_cycle[self._model_idx]

        env = api_key_env or cfg.gemini_api_key_classifier_env or cfg.gemini_api_key_env
        self._api_key = os.getenv(env) or os.getenv(cfg.gemini_api_key_env)
        if not self._api_key:
            raise RuntimeError("Thiếu GEMINI API KEY cho classifier.")
        self._client = genai.Client(api_key=self._api_key)

        self._cache = RedisCache(
            cfg.redis_url,
            prefix="pretune:clf:v2:",
            ttl_sec=getattr(cfg, "cache_ttl_sec", 86400),
            enabled=getattr(cfg, "use_redis_cache", False),
        )
        self._limiter = build_global_limiter(cfg)

        # Initialize global error handler if available
        if GeminiErrorHandler is not None:
            try:
                self._error_handler = GeminiErrorHandler(
                    base_backoff_sec=float(getattr(cfg, "base_backoff_sec", 1.0) or 1.0),
                    jitter_sec=float(getattr(cfg, "jitter_sec", 0.0) or 0.0),
                )
            except Exception:
                self._error_handler = None
        else:
            self._error_handler = None

        # Giới hạn an toàn để tránh 400/MAX_TOKENS
        self._max_chars = int(getattr(cfg, "classifier_max_chars", 400))
        self._max_batch = int(getattr(cfg, "classifier_max_batch", 8))

    def _call_once(self, text: str, model: str, max_tokens: int = 8) -> int | None:
        """
        Gọi 1 lần với JSON schema. Trả 1/0 hoặc None nếu empty/blocked.
        """
        config = types.GenerateContentConfig(  # type: ignore[attr-defined]
            temperature=0.0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=RESP_SCHEMA,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = _build_contents(text, self._max_chars)
        resp = self._client.models.generate_content(model=model, contents=contents, config=config)

        # Ưu tiên parse JSON
        data = getattr(resp, "parsed", None)
        # Nếu SDK chưa map parsed, thử từ resp.text
        if not data:
            raw = (getattr(resp, "text", "") or "").strip()
            if raw:
                try:
                    data = json.loads(raw)
                except Exception:
                    # chấp nhận "1"/"0" như fallback hiếm
                    tok = raw.split()[0] if raw else ""
                    if tok in {"1", "0"}:
                        return 1 if tok == "1" else 0

        if not data or "complete" not in data:
            # Có thể do block hoặc cắt ngắn
            return None

        return 1 if bool(data.get("complete")) else 0

    def _call_once_multi(self, texts: List[str], model: str) -> List[int] | None:
        """
        Gọi 1 lần cho nhiều câu, trả list[int] cùng thứ tự. None nếu empty/blocked.
        """
        # Schema JSON: ARRAY of BOOLEAN
        if RESP_SCHEMA is not None and hasattr(types, "Schema"):
            resp_schema = types.Schema(  # type: ignore[attr-defined]
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.BOOLEAN),
            )
        else:
            resp_schema = None

        # Ước lượng output tokens: ~3 tokens/câu + 8 overhead, cap 128
        max_out = min(128, 3 * max(1, len(texts)) + 8)
        config = types.GenerateContentConfig(  # type: ignore[attr-defined]
            temperature=0.0,
            max_output_tokens=max_out,
            response_mime_type="application/json",
            response_schema=resp_schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = _build_contents_multi(texts, self._max_chars)
        resp = self._client.models.generate_content(model=model, contents=contents, config=config)

        # Ưu tiên parsed (list[bool])
        data = getattr(resp, "parsed", None)
        if isinstance(data, list) and all(isinstance(x, bool) for x in data):
            return [1 if x else 0 for x in data]

        # Fallback: parse text
        raw = (getattr(resp, "text", "") or "").strip()
        if raw:
            try:
                arr = json.loads(raw)
                if isinstance(arr, list):
                    return [1 if bool(x) else 0 for x in arr]
            except Exception:
                # Fallback cuối: bắt 0/1 rời rạc
                toks = [t for t in raw.replace(",", " ").split() if t in {"0", "1"}]
                if toks:
                    return [1 if t == "1" else 0 for t in toks]
        return None

    def classify_batch(self, sentences: Sequence[str]) -> List[int]:
        out: List[int] = []
        to_query: list[tuple[int, str, str]] = []

        # 1) cache trước
        for idx, s in enumerate(sentences):
            text = (s or "").strip()
            if not text:
                out.append(0); continue
            key = self._cache.sha1("clf", _norm_text(text))  # model-agnostic key
            cached = self._cache.get(key)
            if cached is not None:
                out.append(1 if cached.startswith("1") else 0); continue
            out.append(-1); to_query.append((idx, text, key))

        # 2) gom theo mini-batch (giảm request/ngày)
        B = max(1, int(getattr(self, "_max_batch", 8)))
        j = 0
        while j < len(to_query):
            sub = to_query[j : j + B]
            idxs = [it[0] for it in sub]
            texts = [it[1] for it in sub]
            keys  = [it[2] for it in sub]

            attempts = 0
            model_idx = 0
            model = self._models_cycle[model_idx]
            sub_B = len(texts)

            while True:
                self._limiter.wait(getattr(self._cfg, "jitter_sec", 0.0))
                try:
                    vals = self._call_once_multi(texts, model)
                    if vals is None or len(vals) < sub_B:
                        raise RuntimeError("empty_or_blocked_or_truncated")
                    # Map kết quả
                    for k in range(sub_B):
                        v = 1 if vals[k] == 1 else 0
                        out[idxs[k]] = v
                        self._cache.set(keys[k], "1" if v else "0")
                    j += sub_B
                    break

                except Exception as e:
                    attempts += 1
                    # Use centralized error handler if available
                    decision = None
                    if getattr(self, "_error_handler", None) is not None:
                        try:
                            info = self._error_handler.extract_info(e)
                            decision = self._error_handler.make_decision(
                                error_info=info,
                                attempt=attempts - 1,
                                max_retries=self._cfg.max_retries,
                                models=len(self._models_cycle),
                                exc=e,
                            )
                        except Exception:
                            decision = None
                    # Log the error details
                    code = status = ""
                    msg = str(e)
                    if genai_errors and isinstance(e, (genai_errors.ClientError, genai_errors.ServerError)):
                        ej = getattr(e, "response_json", {}) or {}
                        code = str(ej.get("error", {}).get("code", "")) or str(getattr(e, "status_code", "") or "")
                        status = (ej.get("error", {}).get("status", "") or "").upper()
                        msg = ej.get("error", {}).get("message", "") or msg
                    _LOG.warning(
                        "Classifier(batch) err attempt=%d model=%s code=%s status=%s msg=%s size=%d",
                        attempts, model, code, status, msg, sub_B
                    )
                    # If no decision, fall back to simple backoff and eventual heuristic fallback
                    if decision is None:
                        self._limiter.penalize()
                        sleep_s = max(0.2, self._cfg.base_backoff_sec * (1.5 ** (attempts - 1))
                                      + random.uniform(0, getattr(self._cfg, "jitter_sec", 0.0)))
                        time.sleep(sleep_s)
                        if attempts >= self._cfg.max_retries:
                            for k in range(sub_B):
                                v = 1 if _heuristic_complete(texts[k]) else 0
                                out[idxs[k]] = v
                                self._cache.set(keys[k], "1" if v else "0")
                            _LOG.error("Classifier(batch) give-up → heuristic applied for %d items.", sub_B)
                            j += sub_B
                            break
                        continue
                    # Apply penalize, switch model and sleep as per decision
                    if decision.penalize:
                        self._limiter.penalize()
                    if decision.switch_model and len(self._models_cycle) > 1:
                        model_idx = (model_idx + 1) % len(self._models_cycle)
                        model = self._models_cycle[model_idx]
                        _LOG.warning("Classifier(batch) switching model due to %s → %s", decision.error_type, model)
                    if decision.sleep and decision.sleep > 0:
                        time.sleep(decision.sleep)
                    # Handle action-specific logic
                    if decision.action == "bad_request":
                        if sub_B > 1:
                            mid = max(1, sub_B // 2)
                            left = sub[:mid]
                            right = sub[mid:]
                            to_query[j:j+sub_B] = left + right
                            break
                        else:
                            self._max_chars = max(120, int(self._max_chars * 0.6))
                            val = self._call_once(texts[0], model, max_tokens=6)
                            if val is None:
                                val = 1 if _heuristic_complete(texts[0]) else 0
                            out[idxs[0]] = val
                            self._cache.set(keys[0], "1" if val else "0")
                            j += 1
                            break
                    if decision.action in {"fallback", "giveup"}:
                        for k in range(sub_B):
                            v = 1 if _heuristic_complete(texts[k]) else 0
                            out[idxs[k]] = v
                            self._cache.set(keys[k], "1" if v else "0")
                        j += sub_B
                        break
                    # Otherwise, retry
                    continue

        return [0 if v == -1 else v for v in out]
