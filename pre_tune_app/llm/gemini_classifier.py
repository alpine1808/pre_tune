from __future__ import annotations
import logging, time, random, json, os, unicodedata
from typing import Sequence, List

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.llm.interfaces import ICompletenessClassifier
from pre_tune_app.infra.cache.redis_cache import RedisCache

# NEW: error_handles
from pre_tune_app.error_handles import (
    make_limiter,
    make_error_handler_from_cfg,
    make_policy_from_cfg,
    execute_with_retry,
    CallContext,
)

_LOG = logging.getLogger(__name__)

# ---- SDK guards ----
try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GENAI = False

_PROMPT = (
    "Bạn là bộ phân loại câu pháp lý tiếng Việt. "
    "Hãy trả về JSON đúng định dạng {\"complete\": true|false} "
    "(không giải thích, không văn bản thừa)."
)

if _HAS_GENAI:
    RESP_SCHEMA = types.Schema(  # type: ignore[attr-defined]
        type=types.Type.OBJECT,
        properties={"complete": types.Schema(type=types.Type.BOOLEAN)},
        required=["complete"],
    )
else:
    RESP_SCHEMA = None  # type: ignore

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
    lines = []
    for i, s in enumerate(texts, 1):
        lines.append(f"{i}. {_trim_soft(s, max_chars)}")
    prompt = (
        f"{_PROMPT} Trả về JSON mảng booleans cho các câu sau theo đúng thứ tự.\n"
        "Ví dụ: [true,false,true]\n---\n" + "\n".join(lines)
    )
    return [{"role": "user", "parts": [{"text": prompt}]}]

class GeminiCompletenessClassifier(ICompletenessClassifier):
    def __init__(self, cfg: AppConfig, model_id: str | None = None, api_key_env: str | None = None, rpm: int | None = None) -> None:
        if not _HAS_GENAI:
            raise RuntimeError("google-genai SDK không có. Vui lòng cài đặt.")
        self._cfg = cfg

        primary = cfg.classifier_model_id or cfg.model_text
        m_fallback = getattr(cfg, "model_text_fallback", None)
        models = [m for m in [primary, m_fallback] if m]
        models = [m for m in models if not str(m).startswith("gemini-1.5")]
        if not models:
            models = ["gemini-2.0-flash-lite"]
        self._models_cycle: List[str] = models

        self._api_key = cfg.gemini_api_key_classifier_env
        if not self._api_key:
            raise RuntimeError("Thiếu GEMINI API KEY cho classifier.")
        self._client = genai.Client(api_key=self._api_key)

        self._cache = RedisCache(
            cfg.redis_url,
            prefix="pretune:clf:v2:",
            ttl_sec=getattr(cfg, "cache_ttl_sec", 86400),
            enabled=getattr(cfg, "use_redis_cache", False),
        )

        # NEW: limiter + error handler
        self._limiter = make_limiter(cfg, key_prefix="gemini", op="classifier")
        self._error_handler = make_error_handler_from_cfg(cfg)

        self._max_chars = int(getattr(cfg, "classifier_max_chars", 400))
        self._max_batch = int(getattr(cfg, "classifier_max_batch", 8))

    def _call_once(self, text: str, model: str, max_tokens: int = 8) -> int | None:
        config = types.GenerateContentConfig(  # type: ignore[attr-defined]
            temperature=0.0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=RESP_SCHEMA,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = _build_contents(text, self._max_chars)
        resp = self._client.models.generate_content(model=model, contents=contents, config=config)

        data = getattr(resp, "parsed", None)
        if not data:
            raw = (getattr(resp, "text", "") or "").strip()
            if raw:
                try:
                    data = json.loads(raw)
                except Exception:
                    tok = raw.split()[0] if raw else ""
                    if tok in {"1", "0"}:
                        return 1 if tok == "1" else 0
        if not data or "complete" not in data:
            return None
        return 1 if bool(data.get("complete")) else 0

    def _call_once_multi(self, texts: List[str], model: str) -> List[int] | None:
        if RESP_SCHEMA is not None and hasattr(types, "Schema"):
            resp_schema = types.Schema(  # type: ignore[attr-defined]
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.BOOLEAN),
            )
        else:
            resp_schema = None

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

        data = getattr(resp, "parsed", None)
        if isinstance(data, list) and all(isinstance(x, bool) for x in data):
            return [1 if x else 0 for x in data]

        raw = (getattr(resp, "text", "") or "").strip()
        if raw:
            try:
                arr = json.loads(raw)
                if isinstance(arr, list):
                    return [1 if bool(x) else 0 for x in arr]
            except Exception:
                toks = [t for t in raw.replace(",", " ").split() if t in {"0", "1"}]
                if toks:
                    return [1 if t == "1" else 0 for t in toks]
        return None

    def classify_batch(self, sentences: Sequence[str]) -> List[int]:
        out: List[int] = []
        to_query: list[tuple[int, str, str]] = []

        # cache trước
        for idx, s in enumerate(sentences):
            text = (s or "").strip()
            if not text:
                out.append(0); continue
            key = self._cache.sha1("clf", _norm_text(text))
            cached = self._cache.get(key)
            if cached is not None:
                out.append(1 if cached.startswith("1") else 0); continue
            out.append(-1); to_query.append((idx, text, key))

        B = max(1, int(getattr(self, "_max_batch", 8)))
        j = 0
        while j < len(to_query):
            sub = to_query[j : j + B]
            idxs = [it[0] for it in sub]
            texts = [it[1] for it in sub]
            keys  = [it[2] for it in sub]
            sub_B = len(texts)

            policy = make_policy_from_cfg(self._cfg, op="classifier")

            def _do_call(model_id: str):
                return self._call_once_multi(texts, model_id)

            outcome = execute_with_retry(
                models_cycle=[m for m in self._models_cycle if m],
                policy=policy,
                limiter=self._limiter,
                error_handler=self._error_handler,
                call_fn=_do_call,
                ctx=CallContext(op="classifier"),
            )

            if not outcome.ok or outcome.response is None or len(outcome.response) < sub_B:
                # give-up → heuristic như cũ
                for k in range(sub_B):
                    v = 1 if _heuristic_complete(texts[k]) else 0
                    out[idxs[k]] = v
                    self._cache.set(keys[k], "1" if v else "0")
                j += sub_B
                continue

            vals = list(outcome.response)
            for k in range(sub_B):
                v = 1 if vals[k] == 1 else 0
                out[idxs[k]] = v
                self._cache.set(keys[k], "1" if v else "0")
            j += sub_B

        return [0 if v == -1 else v for v in out]
