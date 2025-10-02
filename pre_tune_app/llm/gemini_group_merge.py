# pre_tune_app/llm/gemini_group_merge.py
from __future__ import annotations
import time, random, json, os, logging
from typing import Sequence, List

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.llm.interfaces import IGroupMergeDecider
from pre_tune_app.infra.cache.redis_cache import RedisCache
from pre_tune_app.infra.ratelimit.global_limiter import build_global_limiter
# import centralized error handler for Gemini API
try:
    from pre_tune_app.infra.ratelimit.gemini_error_utils import GeminiErrorHandler
except Exception:
    GeminiErrorHandler = None  # type: ignore

_LOG = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
    _HAS = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    genai_errors = None  # type: ignore
    _HAS = False

def _trim(s: str, max_chars: int) -> str:
    if not isinstance(s, str): s = str(s)
    if max_chars <= 0 or len(s) <= max_chars: return s
    return s[:max_chars]

# Prompt ngắn gọn, yêu cầu in 0/1 theo thứ tự ứng viên
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
        self._api_key = os.getenv(cfg.gemini_api_key_group_env) or os.getenv(cfg.gemini_api_key_env)
        if not self._api_key:
            raise RuntimeError("Thiếu GEMINI API KEY cho group merge.")
        self._client = genai.Client(api_key=self._api_key)
        self._cache = RedisCache(cfg.redis_url, prefix="pretune:gmerge:", ttl_sec=getattr(cfg, "cache_ttl_sec", 86400), enabled=getattr(cfg, "use_redis_cache", False))
        # GLOBAL 10 rpm cho toàn bộ Gemini
        self._limiter = build_global_limiter(cfg)

        # Centralized error handler
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

        # Chuỗi model để failover khi 429/503
        self._models_cycle: List[str] = [m for m in [self._model_id, getattr(cfg, "model_text_fallback", None)] if m]

        # Giới hạn an toàn để tránh 400
        self._max_cand_per_call = int(getattr(cfg, "group_llm_max_candidates", 5))
        self._max_chars_canonical = int(getattr(cfg, "group_llm_max_chars_canonical", 800))
        self._max_chars_candidate = int(getattr(cfg, "group_llm_max_chars_candidate", 400))

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

        # Trim an toàn để giảm rủi ro 400
        can_trim = _trim(canonical, self._max_chars_canonical)
        cands_trim = [_trim(c, self._max_chars_candidate) for c in candidates]

        def _build_contents(can_txt: str, cand_list: Sequence[str]):
            prompt = (
                _PROMPT
                + "\n---\nCHUẨN:\n"
                + can_txt
                + "\n\nỨNG VIÊN:\n- "
                + "\n- ".join(cand_list)
            )
            return [{"role": "user", "parts": [{"text": prompt}]}]

        contents = _build_contents(can_trim, cands_trim)

        attempts = 0
        local_batch_size = len(cands_trim)
        model_idx = 0
        model = self._models_cycle[model_idx]
        while True:
            self._limiter.wait(getattr(self._cfg, "jitter_sec", 0.0))
            try:
                if types is not None:
                    config = types.GenerateContentConfig(  # type: ignore[attr-defined]
                        temperature=0.0,
                        max_output_tokens=32,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    )
                else:
                    config = {"temperature": 0.0, "max_output_tokens": 32}
                resp = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                text = (getattr(resp, "text", "") or "").strip()
                toks = [t for t in text.replace(",", " ").split() if t in {"0", "1"}]
                out = [1 if (i < len(toks) and toks[i] == "1") else 0 for i in range(len(cands_trim))]
                self._cache.set(key, json.dumps(["1" if v else "0" for v in out]))
                return out
            except Exception as e:
                attempts += 1
                # Use error handler if available
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
                # Log error
                code = status = ""
                msg = str(e)
                if genai_errors and isinstance(e, (genai_errors.ClientError, genai_errors.ServerError)):
                    ej = getattr(e, "response_json", {}) or {}
                    code = str(ej.get("error", {}).get("code", "")) or str(getattr(e, "status_code", "") or "")
                    status = (ej.get("error", {}).get("status", "") or "").upper()
                    msg = ej.get("error", {}).get("message", "") or msg
                _LOG.warning(
                    "GroupMerge err code=%s status=%s msg=%s attempt=%d batch=%d",
                    code, status, msg, attempts, local_batch_size
                )
                if decision is None:
                    # basic exponential backoff; penalize to be safe
                    self._limiter.penalize()
                    time.sleep(max(0.2, self._cfg.base_backoff_sec * (1.5 ** (attempts - 1)) + random.uniform(0, getattr(self._cfg, "jitter_sec", 0.0))))
                    if attempts >= self._cfg.max_retries:
                        return [0] * len(cands_trim)
                    continue
                # apply penalize, switch model, sleep
                if decision.penalize:
                    self._limiter.penalize()
                if decision.switch_model and len(self._models_cycle) > 1:
                    model_idx = 1 - model_idx
                    model = self._models_cycle[model_idx]
                    _LOG.warning("GroupMerge switching model due to %s → %s", decision.error_type, model)
                if decision.sleep and decision.sleep > 0:
                    time.sleep(decision.sleep)
                # handle bad_request: split or deep-trim
                if decision.action == "bad_request":
                    if local_batch_size > 1:
                        half = max(1, local_batch_size // 2)
                        left = self._call_once(canonical, candidates[:half])
                        right = self._call_once(canonical, candidates[half:])
                        out = left + right
                        self._cache.set(key, json.dumps(["1" if v else "0" for v in out]))
                        return out
                    else:
                        # deep trim and retry
                        can_deep = _trim(canonical, 200)
                        cand_deep = [_trim(cands_trim[0], 120)]
                        contents = _build_contents(can_deep, cand_deep)
                        continue
                # fallback or giveup: return zero list
                if decision.action in {"fallback", "giveup"}:
                    return [0] * len(cands_trim)
                # else retry
                continue

    def confirm_membership(self, canonical: str, candidates: Sequence[str]) -> List[int]:
        if not candidates:
            return []
        # Trim chuẩn và ứng viên để tránh 400 do prompt quá dài
        can = _trim(canonical, self._max_chars_canonical)
        cands_all = [_trim(c, self._max_chars_candidate) for c in candidates]

        out: List[int] = []
        # Chia nhỏ theo minibatch để giảm rủi ro 400/429
        B = max(1, self._max_cand_per_call)
        for i in range(0, len(cands_all), B):
            sub = cands_all[i:i+B]
            dec = self._call_once(can, sub)
            # pad/truncate phòng sự cố
            if len(dec) < len(sub):
                dec = dec + [0] * (len(sub) - len(dec))
            elif len(dec) > len(sub):
                dec = dec[:len(sub)]
            out.extend(dec)
        return out
