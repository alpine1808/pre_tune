# pre_tune_app/llm/gemini_text.py
from __future__ import annotations
import os, json, time, random, logging, threading, re
from typing import Sequence, Tuple, List, Dict, Any

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.domain.models import Chunk, OpsFlags
from pre_tune_app.llm.interfaces import ITextCleanModel

_LOG = logging.getLogger(__name__)

# --------- Optional: Redis rate limiter ----------
try:
    from pre_tune_app.infra.ratelimit.redis_bucket import (
        DistributedRedisRateLimiter,
        RedisBucketConfig,
    )
    _HAS_REDIS_BUCKET = True
except Exception:
    DistributedRedisRateLimiter = None  # type: ignore
    RedisBucketConfig = None  # type: ignore
    _HAS_REDIS_BUCKET = False


def _build_limiter(cfg: AppConfig, rpm: int, key_suffix: str):
    """
    Ưu tiên limiter phân tán qua Redis nếu có cfg.redis_url, fallback sang limiter nội bộ.
    """
    try:
        redis_url = getattr(cfg, "redis_url", None)
        if _HAS_REDIS_BUCKET and redis_url:
            return DistributedRedisRateLimiter(
                url=redis_url,
                cfg=RedisBucketConfig(
                    key_prefix=f"pre_tune:gemini:{key_suffix}:{getattr(cfg,'project_id','local')}",
                    capacity=float(rpm),
                    refill_per_sec=float(rpm) / 60.0,
                    jitter_ms=120,
                ),
            )
    except Exception:
        _LOG.exception("Init DistributedRedisRateLimiter failed; fallback to local limiter.")
    return _RateLimiter(
        rpm=max(1, int(rpm)),
        floor=max(1, int(getattr(cfg, "min_rpm_floor", 1))),
        slowdown_factor=float(getattr(cfg, "adapt_slowdown_factor", 2.0)),
    )


# --------- SDK guard ----------
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError, ServerError
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    ClientError = Exception  # type: ignore
    ServerError = Exception  # type: ignore
    _HAS_GENAI = False


# --------- Token utils (fallback an toàn) ----------
try:
    from pre_tune_app.llm.utils.tokens import count_tokens_safe, trim_by_chars_approx
except Exception:
    def count_tokens_safe(client, model, contents): return -1
    def trim_by_chars_approx(text: str, max_tokens: int) -> str:
        if not isinstance(text, str):
            text = str(text)
        return text  # non-strict fallback


# --------- JSON schema builder (chuẩn google.genai types.Schema) ----------
def _build_clean_schema():
    if types is None:
        # fallback dạng dict (khi chưa cài SDK)
        return {
            "type": "OBJECT",
            "required": ["clean_text", "ops", "flags"],
            "properties": {
                "clean_text": {"type": "STRING"},
                "ops": {"type": "ARRAY", "items": {"type": "STRING"}},
                "flags": {
                    "type": "OBJECT",
                    "properties": {
                        "continues": {"type": "BOOLEAN"},
                        "error": {"type": "BOOLEAN"},
                    },
                },
            },
        }
    # SDK present → dùng types.Schema/Type
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "clean_text": types.Schema(type=types.Type.STRING),
            "ops": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "flags": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "continues": types.Schema(type=types.Type.BOOLEAN),
                    "error": types.Schema(type=types.Type.BOOLEAN),
                },
            ),
        },
        required=["clean_text", "ops", "flags"],
    )


SYSTEM_INSTRUCTION = (
    "Bạn là công cụ làm sạch chunk văn bản PDF tiếng Việt (sau OCR). "
    "Giữ nguyên nội dung, không suy diễn; sửa lỗi ngắt dòng/khoảng trắng/gạch đầu dòng/ký tự lạ; "
    "không thay đổi số/ký hiệu pháp lý; không thêm bớt câu. "
    "Trả về DUY NHẤT một JSON phù hợp schema."
)


# --------- Local rate limiter ----------
class _RateLimiter:
    def __init__(self, rpm: int, floor: int, slowdown_factor: float):
        self._lock = threading.Lock()
        self._rpm = max(rpm, floor)
        self._floor = floor
        self._factor = slowdown_factor
        self._min_interval = 60.0 / max(self._rpm, 1)
        self._next_t = 0.0

    def wait(self, jitter: float = 0.0) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_t - now)
            if delay > 0:
                time.sleep(delay)
            self._next_t = max(now, self._next_t) + self._min_interval
        if jitter > 0:
            time.sleep(random.uniform(0.0, jitter))

    def penalize(self) -> None:
        with self._lock:
            new_rpm = max(int(self._rpm / self._factor), self._floor)
            if new_rpm != self._rpm:
                _LOG.warning("Text RateLimiter slowdown: RPM %s -> %s", self._rpm, new_rpm)
                self._rpm = new_rpm
                self._min_interval = 60.0 / max(self._rpm, 1)


def _is_preview_model(mid: str) -> bool:
    mid = (mid or "").lower()
    return any(tag in mid for tag in ("-exp", "preview", "experimental"))


# --------- Safety relaxed helper (tùy chọn) ----------
def _get_relaxed_safety():
    if not _HAS_GENAI:
        return []
    cats = [
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_HARASSMENT",
    ]
    out = []
    for c in cats:
        try:
            out.append(types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH"))
        except Exception:
            try:
                enum_cat = getattr(types.HarmCategory, c, None)
                if enum_cat is not None:
                    out.append(types.SafetySetting(category=enum_cat, threshold="BLOCK_ONLY_HIGH"))
            except Exception:
                pass
    return out


# --------- Prompt helpers (context window) ----------
_TERMINAL = tuple(".?!…:;”’\"»)]}」』）>")  # kết câu phổ biến

def _slice_tail(s: str, n: int) -> str:
    s = (s or "")
    return s[-n:] if n > 0 and len(s) > n else s

def _slice_head(s: str, n: int) -> str:
    s = (s or "")
    return s[:n] if n > 0 and len(s) > n else s

def _looks_head_open(text: str, prev_tail: str) -> bool:
    if not text:
        return False
    if prev_tail and not prev_tail.rstrip().endswith(_TERMINAL):
        lead = text.lstrip()[:1]
        return bool(lead and (lead.islower() or lead in {"-", "•", "–"}))
    return False

def _looks_tail_open(text: str, next_head: str) -> bool:
    if not text:
        return False
    if not text.rstrip().endswith(_TERMINAL) and next_head:
        return True
    return False


class GeminiTextModel(ITextCleanModel):
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._client = self._maybe_make_client()
        self._limiter = _build_limiter(cfg, getattr(cfg, "rpm_text", 60), key_suffix="text")
        self._models_cycle: List[str] = [cfg.model_text, cfg.model_text_fallback]
        if self._client is None:
            _LOG.info("GeminiTextModel dry-run or SDK not available; client=None")

    # --- client factory ---
    def _maybe_make_client(self):
        if self._cfg.dry_run or not _HAS_GENAI:
            return None
        if self._cfg.use_vertex:
            return genai.Client(
                vertexai=True,
                project=self._cfg.gcp_project or None,
                location=self._cfg.gcp_location or None,
            )
        api_key = os.getenv(self._cfg.gemini_api_key_env)
        if not api_key:
            _LOG.warning("GEMINI_API_KEY not set; running Text in dry-run.")
            return None
        return genai.Client(api_key=api_key)

    # --- helpers ---
    @staticmethod
    def _parse_retry_delay_seconds(err: Exception) -> float | None:
        try:
            data = getattr(err, "response_json", None) or {}
            details = data.get("error", {}).get("details", [])
            for d in details:
                if d.get("@type", "").endswith("google.rpc.RetryInfo"):
                    retry = d.get("retryDelay")
                    if isinstance(retry, str) and retry.endswith("s"):
                        return float(retry[:-1])
        except Exception:
            pass
        return None

    @staticmethod
    def _sanitize(data: dict, raw: str) -> dict:
        """
        Chỉ giữ các key cho phép; điền mặc định an toàn.
        Lưu ý: JSON Mode + schema đã enforce, nhưng sanitize giúp robust hơn.
        """
        allowed_top = {"clean_text", "ops", "flags"}
        for k in list(data.keys()):
            if k not in allowed_top:
                data.pop(k, None)

        clean_text = data.get("clean_text", raw)
        if not isinstance(clean_text, str):
            clean_text = str(clean_text)

        ops = data.get("ops", [])
        if not isinstance(ops, list):
            ops = [ops]
        ops = [str(x) for x in ops if x is not None]

        flags = data.get("flags", {})
        if not isinstance(flags, dict):
            flags = {"error": True}

        allowed_flags = {"continues", "error"}
        for k in list(flags.keys()):
            if k not in allowed_flags:
                flags.pop(k, None)

        return {"clean_text": clean_text, "ops": ops, "flags": flags}

    # --- Guard: check block/finish/candidates trước khi đọc .text ---
    @staticmethod
    def _assess_response(resp) -> Dict[str, Any]:
        pf = getattr(resp, "prompt_feedback", None)
        block_reason = getattr(pf, "block_reason", None) or getattr(pf, "blockReason", None)
        if block_reason:
            return {"block": True, "reason": str(block_reason)}

        cands = list(getattr(resp, "candidates", []) or [])
        if not cands:
            return {"empty": True}

        c0 = cands[0]
        finish = getattr(c0, "finish_reason", None) or getattr(c0, "finishReason", None)
        if isinstance(finish, str):
            up = finish.upper()
            if up in {"SAFETY", "MAX_TOKENS", "RECITATION"}:
                return {"finish": up}

        txt = getattr(resp, "text", None)
        if not txt:
            content = getattr(c0, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                txt = "".join([getattr(p, "text", "") for p in parts if getattr(p, "text", "")])
        if not txt:
            return {"empty": True}

        return {"text": txt}

    def _mk_prompt(self, base_text: str, prev_tail: str, next_head: str) -> str:
        ctx = []
        if prev_tail:
            ctx.append(f"<<PREV_TAIL>>\n{prev_tail}")
        if next_head:
            ctx.append(f"<<NEXT_HEAD>>\n{next_head}")
        ctx_block = ("\n\n".join(ctx) + "\n\n") if ctx else ""
        return (
            "Ngữ cảnh (chỉ đọc, KHÔNG được sao chép nội dung từ ngữ cảnh vào kết quả):\n"
            f"{ctx_block}"
            "YÊU CẦU: Nếu đoạn hiện tại bắt đầu giữa câu → thêm 'HEAD_OPEN' vào ops; "
            "nếu kết thúc giữa câu → thêm 'TAIL_OPEN' vào ops. Không bịa nội dung.\n\n"
            "Chỉ trả về DUY NHẤT một JSON đúng schema, không giải thích, không Markdown.\n"
            f"Đoạn cần làm sạch (chỉ phần hiện tại):\n---\n{base_text}"
        )

    # --- Core call ---
    def _call_clean_once(self, chunk: Chunk, prev_tail: str = "", next_head: str = "") -> Tuple[str, OpsFlags, float]:
        # Dry-run: trả lại nguyên văn + gợi ý open flags bằng heuristic (không bịa nội dung)
        if self._client is None:
            ct = chunk.raw_text or ""
            ops = []
            if _looks_head_open(ct, prev_tail):
                ops.append("HEAD_OPEN")
            if _looks_tail_open(ct, next_head):
                ops.append("TAIL_OPEN")
            return (ct, OpsFlags(ops=ops, flags={"continues": bool(chunk.continues_flag or False)}), 0.8)

        start = time.monotonic()
        attempts = 0
        model_idx = 0
        model = self._models_cycle[model_idx]

        def _gen(prompt: str, mdl: str):
            config = types.GenerateContentConfig(  # type: ignore[attr-defined]
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=_build_clean_schema(),
                temperature=0,
                max_output_tokens=getattr(self._cfg, "max_output_tokens_text", 1024),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                safety_settings=_get_relaxed_safety(),
            )
            return self._client.models.generate_content(
                model=mdl,
                contents=prompt,
                config=config,
            )

        base_text = chunk.raw_text or ""
        max_prompt_toks = int(getattr(self._cfg, "max_prompt_tokens_text", 0) or 0)
        if max_prompt_toks > 0:
            base_text = trim_by_chars_approx(base_text, max_prompt_toks)

        # context windows (chars)
        win_tail = int(getattr(self._cfg, "window_tail_chars", 280) or 280)
        win_head = int(getattr(self._cfg, "window_head_chars", 280) or 280)
        prev_tail = _slice_tail(prev_tail, win_tail)
        next_head = _slice_head(next_head, win_head)

        prompt = self._mk_prompt(base_text, prev_tail, next_head)

        while True:
            elapsed = time.monotonic() - start
            if attempts >= self._cfg.max_retries or elapsed > self._cfg.max_retry_time_sec:
                _LOG.error("Text give-up (attempts=%s, elapsed=%.2fs). Fallback passthrough.", attempts, elapsed)
                return (chunk.raw_text, OpsFlags(ops=["quota_fallback"], flags={"error": True}), 0.75)

            # respect RPM + jitter
            self._limiter.wait(getattr(self._cfg, "jitter_sec", 0.0))

            try:
                resp = _gen(prompt, model)
                assess = self._assess_response(resp)

                if assess.get("block"):
                    _LOG.warning("Prompt blocked by safety (%s).", assess.get("reason"))
                    return (chunk.raw_text, OpsFlags(ops=["safety_blocked"], flags={"error": True}), 0.72)

                if "finish" in assess:
                    fin = str(assess["finish"]).upper()
                    if fin == "MAX_TOKENS":
                        return (chunk.raw_text, OpsFlags(ops=["max_tokens_truncated"], flags={"error": True}), 0.72)
                    return (chunk.raw_text, OpsFlags(ops=[f"finish_{fin.lower()}"], flags={"error": True}), 0.72)

                if assess.get("empty"):
                    _LOG.warning("Empty candidates/text from model.")
                    return (chunk.raw_text, OpsFlags(ops=["empty_candidate"], flags={"error": True}), 0.70)

                text = assess.get("text", "") or ""

                # Parse JSON lần 1
                try:
                    data = json.loads(text)
                except Exception:
                    # Thử "repair" một nhịp
                    repair = (
                        "JSON ở trên không hợp lệ theo schema. "
                        "Hãy IN LẠI duy nhất một JSON hợp lệ, không giải thích, không Markdown."
                    )
                    self._limiter.wait(getattr(self._cfg, "jitter_sec", 0.0))
                    resp2 = _gen(repair + "\n---\n" + text, model)
                    assess2 = self._assess_response(resp2)
                    if "text" not in assess2:
                        return (chunk.raw_text, OpsFlags(ops=["client_error"], flags={"error": True}), 0.70)
                    data = json.loads(assess2["text"])

                data = self._sanitize(data, chunk.raw_text)
                clean_text, ops, flags = data["clean_text"], data.get("ops", []), data.get("flags", {})

                # Nếu model không gắn HEAD_OPEN/TAIL_OPEN, tự gợi ý nhẹ bằng heuristic (không sửa nội dung)
                if "HEAD_OPEN" not in ops and _looks_head_open(clean_text, prev_tail):
                    ops.append("HEAD_OPEN")
                if "TAIL_OPEN" not in ops and _looks_tail_open(clean_text, next_head):
                    ops.append("TAIL_OPEN")

                q = 0.9 if clean_text else 0.6
                return (clean_text, OpsFlags(ops=ops, flags=flags), q)

            except ServerError as e:  # 5xx
                attempts += 1
                _LOG.warning(
                    "Transient Text error (%s). Retry %d/%d",
                    getattr(e, "response_json", None) or str(e),
                    attempts,
                    self._cfg.max_retries,
                )
                time.sleep(
                    self._cfg.base_backoff_sec * (1.5 ** (attempts - 1))
                    + random.uniform(0, getattr(self._cfg, "jitter_sec", 0.0))
                )
                if attempts >= 2 and len(self._models_cycle) > 1:
                    model_idx = 1 - model_idx
                    model = self._models_cycle[model_idx]
                    _LOG.warning("Text switching model due to server error → %s", model)
                continue

            except ClientError as e:  # 4xx
                attempts += 1
                ej = getattr(e, "response_json", None) or {}
                code = ej.get("error", {}).get("code", None)
                status = (ej.get("error", {}).get("status", "") or "").upper()
                # Một số phiên bản SDK có thuộc tính status_code
                http_code = getattr(e, "status_code", None) or getattr(e, "code", None)

                _LOG.warning(
                    "Client Text error code=%s status=%s http=%s attempt=%d",
                    code, status, http_code, attempts
                )

                # Nhận diện 429 thật chặt:
                is_429 = (
                    str(code) == "429"
                    or str(http_code) == "429"
                    or status in {"RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED"}
                    or "RESOURCE_EXHAUSTED" in str(e).upper()
                    or ("RATE" in str(e).upper() and "LIMIT" in str(e).upper())
                )
                if is_429:
                    # nếu server trả RetryInfo -> dùng; nếu không, backoff mũ tối thiểu 15s + jitter
                    retry_s = self._parse_retry_delay_seconds(e)
                    self._limiter.penalize()
                    if retry_s is None:
                        exp = max(self._cfg.base_backoff_sec * (1.5 ** (attempts - 1)), 15.0)
                        jitter = random.uniform(0, max(0.5, getattr(self._cfg, "jitter_sec", 0.0)))
                        retry_s = exp + jitter
                    _LOG.warning("429 detected. Sleeping for %.2fs then retry.", retry_s)
                    time.sleep(retry_s)

                    # Sau 2 lần thì xoay model (nếu có fallback)
                    if attempts >= 2 and len(self._models_cycle) > 1:
                        model_idx = 1 - model_idx
                        model = self._models_cycle[model_idx]
                        _LOG.warning("Switching model after 429 → %s", model)
                    continue

                # Các 4xx khác: thử backoff mềm nếu còn lượt, hết lượt thì fallback 0.7
                if attempts < self._cfg.max_retries:
                    soft_backoff = self._cfg.base_backoff_sec * (1.5 ** (attempts - 1)) + random.uniform(
                        0, getattr(self._cfg, "jitter_sec", 0.0)
                    )
                    _LOG.warning("Non-429 client error; soft-backoff %.2fs then retry.", soft_backoff)
                    time.sleep(soft_backoff)
                    if attempts >= 2 and len(self._models_cycle) > 1:
                        model_idx = 1 - model_idx
                        model = self._models_cycle[model_idx]
                        _LOG.warning("Switching model after client error → %s", model)
                    continue

                _LOG.error("Non-retryable client error. Fallback passthrough.")
                return (chunk.raw_text, OpsFlags(ops=["client_error"], flags={"error": True}), 0.7)

            except Exception:
                attempts += 1
                _LOG.exception("Unknown Text error; retry %d/%d", attempts, self._cfg.max_retries)
                time.sleep(
                    self._cfg.base_backoff_sec * (1.5 ** (attempts - 1))
                    + random.uniform(0, getattr(self._cfg, "jitter_sec", 0.0))
                )
                continue

    # --- Batch tuần tự (tôn trọng RPM) + bơm context prev/next ---
    def clean_text_batch(self, items: Sequence[Chunk]) -> List[Tuple[str, OpsFlags, float]]:
        out: List[Tuple[str, OpsFlags, float]] = []
        n = len(items)
        for i, c in enumerate(items):
            prev_raw = items[i - 1].raw_text if i > 0 else ""
            next_raw = items[i + 1].raw_text if i + 1 < n else ""
            out.append(self._call_clean_once(c, prev_tail=prev_raw, next_head=next_raw))
        return out
