# pre_tune_app/llm/gemini_group_merge.py
from __future__ import annotations
import time, json, os, logging
from typing import Sequence, List

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.llm.interfaces import IGroupMergeDecider
from pre_tune_app.infra.cache.redis_cache import RedisCache

# NEW: dùng error_handles
from pre_tune_app.error_handles import (
    make_limiter,
    make_error_handler_from_cfg,
    make_policy_from_cfg,
    execute_with_retry,
    CallContext,
)

_LOG = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    _HAS = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS = False

def _trim(s: str, max_chars: int) -> str:
    if not isinstance(s, str): s = str(s)
    if max_chars <= 0 or len(s) <= max_chars: return s
    return s[:max_chars]

_PROMPT = (
    "Bạn là bộ so khớp nội dung pháp lý. Cho câu CHUẨN ở trên và danh sách câu ỨNG VIÊN bên dưới, "
    "hãy IN RA cho MỖI ỨNG VIÊN một ký tự 1 nếu cùng NỘI DUNG cốt lõi (dù câu bị cắt/viết khác), hoặc 0 nếu KHÁC nội dung. "
    "Không giải thích. In đúng số lượng ký tự theo số ứng viên, cách nhau bằng khoảng trắng hoặc xuống dòng."
)

class GeminiGroupMergeDecider(IGroupMergeDecider):
    def __init__(self, cfg: AppConfig, model_id: str | None = None, api_key_env: str | None = None, rpm: int | None = None):
        if not _HAS:
            raise RuntimeError("google-genai SDK không có. Vui lòng cài đặt.")
        self._cfg = cfg
        self._model_id = model_id or cfg.model_text
        self._api_key = cfg.gemini_api_key_group_env or cfg.gemini_api_key_env
        if not self._api_key:
            raise RuntimeError("Thiếu GEMINI API KEY cho group merge.")
        self._client = genai.Client(api_key=self._api_key)

        self._cache = RedisCache(
            cfg.redis_url,
            prefix="pretune:gmerge:",
            ttl_sec=getattr(cfg, "cache_ttl_sec", 86400),
            enabled=getattr(cfg, "use_redis_cache", False),
        )

        # NEW: limiter & error handler & model cycle
        self._limiter = make_limiter(cfg, key_prefix="gemini", op="merge")
        self._error_handler = make_error_handler_from_cfg(cfg)
        self._models_cycle: List[str] = [m for m in [self._model_id, getattr(cfg, "model_text_fallback", None)] if m]

        # Giới hạn an toàn để tránh 400
        self._max_cand_per_call = int(getattr(cfg, "group_llm_max_candidates", 5))
        self._max_chars_canonical = int(getattr(cfg, "group_llm_max_chars_canonical", 800))
        self._max_chars_candidate = int(getattr(cfg, "group_llm_max_chars_candidate", 400))

    def _build_contents(self, can_txt: str, cand_list: Sequence[str]):
        prompt = (
            _PROMPT
            + "\n---\nCHUẨN:\n"
            + can_txt
            + "\n\nỨNG VIÊN:\n- "
            + "\n- ".join(cand_list)
        )
        return [{"role": "user", "parts": [{"text": prompt}]}]

    def _call_once(self, canonical: str, candidates: Sequence[str]) -> List[int]:
        # cache theo canonical + tập candidates (giữ nguyên thứ tự)
        key = self._cache.sha1(self._models_cycle[0], "gmerge", canonical, "||".join(candidates))
        cached = self._cache.get(key)
        if cached is not None:
            try:
                data = json.loads(cached)
                return [1 if x == "1" else 0 for x in data]
            except Exception:
                pass

        can_trim = _trim(canonical, self._max_chars_canonical)
        cands_trim = [_trim(c, self._max_chars_candidate) for c in candidates]
        contents = self._build_contents(can_trim, cands_trim)

        def _do_call(model_id: str):
            if types is not None:
                config = types.GenerateContentConfig(  # type: ignore[attr-defined]
                    temperature=0.0,
                    max_output_tokens=32,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
            else:
                config = {"temperature": 0.0, "max_output_tokens": 32}
            return self._client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )

        policy = make_policy_from_cfg(self._cfg, op="merge")
        outcome = execute_with_retry(
            models_cycle=self._models_cycle,
            policy=policy,
            limiter=self._limiter,
            error_handler=self._error_handler,
            call_fn=_do_call,
            ctx=CallContext(op="merge"),
        )

        if not outcome.ok or outcome.response is None:
            # give-up → trả 0 cho toàn bộ ứng viên
            return [0] * len(cands_trim)

        text = (getattr(outcome.response, "text", "") or "").strip()
        toks = [t for t in text.replace(",", " ").split() if t in {"0", "1"}]
        out = [1 if (i < len(toks) and toks[i] == "1") else 0 for i in range(len(cands_trim))]
        self._cache.set(key, json.dumps(["1" if v else "0" for v in out]))
        return out

    def confirm_membership(self, canonical: str, candidates: Sequence[str]) -> List[int]:
        if not candidates:
            return []
        can = _trim(canonical, self._max_chars_canonical)
        cands_all = [_trim(c, self._max_chars_candidate) for c in candidates]

        out: List[int] = []
        B = max(1, self._max_cand_per_call)
        for i in range(0, len(cands_all), B):
            sub = cands_all[i:i+B]
            dec = self._call_once(can, sub)
            if len(dec) < len(sub):
                dec = dec + [0] * (len(sub) - len(dec))
            elif len(dec) > len(sub):
                dec = dec[:len(sub)]
            out.extend(dec)
        return out
