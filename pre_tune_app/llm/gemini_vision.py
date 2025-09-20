from __future__ import annotations
import os, json, time, random, logging, threading, base64
from typing import Sequence, Tuple, List, Dict, Any
from pre_tune_app.config.settings import AppConfig
from pre_tune_app.domain.models import Chunk, OpsFlags
from pre_tune_app.llm.interfaces import IVisionFixModel

# Redis limiter (tùy chọn)
try:
    from pre_tune_app.infra.ratelimit.redis_bucket import DistributedRedisRateLimiter, RedisBucketConfig
except Exception:
    DistributedRedisRateLimiter = None  # type: ignore
    RedisBucketConfig = None  # type: ignore

# Token utilities (không strict)
try:
    from pre_tune_app.llm.utils.tokens import count_tokens_safe, trim_by_chars_approx
except Exception:
    def count_tokens_safe(client, model, contents):  # type: ignore
        return -1
    def trim_by_chars_approx(text: str, max_tokens: int) -> str:  # type: ignore
        return text

_LOG = logging.getLogger(__name__)

def _is_preview_model(mid: str) -> bool:
    mid = (mid or "").lower()
    return any(tag in mid for tag in ("-exp", "preview", "experimental"))

# --- Google GenAI SDK ---
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

# --- Local RateLimiter (adaptive) ---
class _RateLimiter:
    def __init__(self, rpm: int, floor: int, slowdown_factor: float):
        self._lock = threading.Lock()
        self._rpm = max(int(rpm), int(floor))
        self._floor = int(floor)
        self._factor = float(slowdown_factor if slowdown_factor else 2.0)
        self._target = self._rpm
        self._min_interval = 60.0 / max(self._rpm, 1)
        self._next_t = 0.0
    def wait(self, jitter: float = 0.0) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_t - now)
            if delay > 0: time.sleep(delay)
            self._next_t = max(now, self._next_t) + self._min_interval
        if jitter > 0: time.sleep(random.uniform(0.0, jitter))
    def penalize(self) -> None:
        with self._lock:
            new_rpm = max(int(self._rpm / (self._factor if self._factor > 1 else 2.0)), self._floor)
            if new_rpm != self._rpm:
                _LOG.warning("Vision RateLimiter slowdown: RPM %s -> %s", self._rpm, new_rpm)
                self._rpm = new_rpm
                self._min_interval = 60.0 / max(self._rpm, 1)
    def success(self, warmup: float = 1.1) -> None:
        with self._lock:
            if self._rpm >= self._target: return
            new_rpm = min(int(self._rpm * (warmup if warmup > 1 else 1.1)), self._target)
            if new_rpm != self._rpm:
                self._rpm = max(new_rpm, self._floor)
                self._min_interval = 60.0 / max(self._rpm, 1)

# --- JSON Mode schema (strict) ---
def _build_clean_schema():
    if types is None:
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
                        "vision_checked": {"type": "BOOLEAN"},
                    },
                },
            },
        }
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
                    "vision_checked": types.Schema(type=types.Type.BOOLEAN),
                },
            ),
        },
        required=["clean_text", "ops", "flags"],
    )

SYSTEM_INSTRUCTION = (
    "Bạn là công cụ hiệu chỉnh chunk dựa trên hình ảnh bảng/hình từ PDF. "
    "Chỉ sửa lỗi OCR/định dạng; giữ đúng cột/hàng; không suy diễn. "
    "Trả về DUY NHẤT một JSON đúng schema."
)

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[1].lstrip()
        if s.endswith("```"): s = s[:-3].rstrip()
    return s

def _build_limiter(cfg: AppConfig, rpm: int, key_suffix: str):
    try:
        if DistributedRedisRateLimiter and RedisBucketConfig:
            redis_url = getattr(cfg, "redis_url", None)
            if redis_url:
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
        pass
    return _RateLimiter(
        rpm=rpm,
        floor=getattr(cfg, "min_rpm_floor", 1),
        slowdown_factor=getattr(cfg, "adapt_slowdown_factor", 2.0),
    )

def _is_blocked_or_unfinished(resp) -> bool:
    try:
        cands = getattr(resp, "candidates", None)
        if not cands: return True
        c0 = cands[0]
        fr = getattr(c0, "finish_reason", None) or getattr(c0, "finishReason", None)
        if isinstance(fr, str) and fr.upper() in ("SAFETY", "BLOCKED", "MAX_TOKENS"):
            return True
    except Exception:
        pass
    return False

def _mk_part_from_bytes(raw: bytes, mime: str):
    if types is None: return None
    try:
        img = types.Image.from_bytes(raw, mime_type=mime)  # type: ignore[attr-defined]
        return types.Part.from_image(img)  # type: ignore[attr-defined]
    except Exception:
        try:
            return types.Part.from_image(raw)  # type: ignore[attr-defined]
        except Exception:
            return None

def _image_part_from_meta(meta: dict):
    if types is None or not isinstance(meta, dict): return None
    b = meta.get("image_bytes", None)
    if isinstance(b, (bytes, bytearray)):
        mime = meta.get("image_mime") or "image/png"
        return _mk_part_from_bytes(bytes(b), mime)
    b64 = meta.get("image_b64", None)
    if isinstance(b64, str):
        try:
            raw = base64.b64decode(b64, validate=False)
            mime = meta.get("image_mime") or "image/png"
            return _mk_part_from_bytes(raw, mime)
        except Exception:
            return None
    path = meta.get("image_path", None)
    if isinstance(path, str):
        try:
            img = types.Image.from_file(path)  # type: ignore[attr-defined]
            return types.Part.from_image(img)  # type: ignore[attr-defined]
        except Exception:
            return None
    return None

class GeminiVisionModel(IVisionFixModel):
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._client = self._maybe_make_client()
        self._models_cycle: List[str] = [cfg.model_vision, cfg.model_vision_fallback]
        self._models_cycle = [m for m in self._models_cycle if m] or ["gemini-1.5-pro"]
        rpm = int(getattr(cfg, "rpm_vision", 30))
        if _is_preview_model(self._models_cycle[0]): rpm = max(10, int(rpm * 0.5))
        self._limiter = _build_limiter(cfg, rpm, "vision")
        if self._client is None: _LOG.info("GeminiVisionModel dry-run or SDK missing; client=None")
        self._schema = _build_clean_schema()

    def _maybe_make_client(self):
        if getattr(self._cfg, "dry_run", False) or not _HAS_GENAI: return None
        if getattr(self._cfg, "use_vertex", False):
            return genai.Client(
                vertexai=True,
                project=getattr(self._cfg, "gcp_project", None) or None,
                location=getattr(self._cfg, "gcp_location", None) or None,
            )
        api_key = os.getenv(getattr(self._cfg, "gemini_api_key_env", "GEMINI_API_KEY"))
        if not api_key:
            _LOG.warning("GEMINI_API_KEY not set; running Vision in dry-run.")
            return None
        return genai.Client(api_key=api_key)

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
                    if isinstance(retry, dict):
                        sec = float(retry.get("seconds", 0)); ns = float(retry.get("nanos", 0))
                        return sec + ns / 1e9
        except Exception:
            pass
        return None

    @staticmethod
    def _sanitize(data: dict, raw: str) -> dict:
        allowed_top = {"clean_text", "ops", "flags"}
        for k in list(data.keys()):
            if k not in allowed_top: data.pop(k, None)
        clean_text = data.get("clean_text", raw)
        if not isinstance(clean_text, str): clean_text = str(clean_text)
        ops = data.get("ops", []); 
        if not isinstance(ops, list): ops = [ops]
        ops = [str(x) for x in ops if x is not None]
        flags = data.get("flags", {}); 
        if not isinstance(flags, dict): flags = {"error": True}
        allowed_flags = {"continues", "error", "vision_checked"}
        for k in list(flags.keys()):
            if k not in allowed_flags: flags.pop(k, None)
        return {"clean_text": clean_text, "ops": ops, "flags": flags}

    def _build_contents(self, chunk: Chunk, base_prompt: str):
        meta = chunk.metadata or {}
        p = _image_part_from_meta(meta)
        if p is not None and types is not None:
            parts = [types.Part.from_text(base_prompt), p]  # type: ignore[attr-defined]
            return [types.Content(role="user", parts=parts)]  # type: ignore[attr-defined]
        return base_prompt  # text-only fallback

    def _build_config(self):
        kwargs = dict(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=self._schema,
            temperature=0,
        )
        mot = getattr(self._cfg, "max_output_tokens_vision", None)
        if isinstance(mot, int) and mot > 0: kwargs["max_output_tokens"] = mot
        ccv = getattr(self._cfg, "candidate_count_vision", None)
        if isinstance(ccv, int) and ccv > 0: kwargs["candidate_count"] = ccv
        else: kwargs["candidate_count"] = 1
        return types.GenerateContentConfig(**kwargs)  # type: ignore[attr-defined]

    def _call_fix_once(self, chunk: Chunk) -> Tuple[str, OpsFlags, float]:
        if self._client is None:
            return (chunk.raw_text, OpsFlags(ops=[], flags={"vision_checked": True}), 0.85)

        start = time.monotonic()
        attempts = 0
        model_idx = 0
        model = self._models_cycle[model_idx]

        def _gen(contents, mdl: str):
            config = self._build_config()
            return self._client.models.generate_content(model=mdl, contents=contents, config=config)

        base_prompt = (
            "Phục hồi văn bản bảng/danh sách từ OCR nếu cần; nếu không chắc, giữ nguyên.\n"
            "Chỉ trả về MỘT JSON hợp lệ theo schema.\n"
            f"Đoạn:\n---\n{chunk.raw_text}"
        )

        # Nếu có limit token → cắt TEXT PART trước khi build Parts (không đụng ảnh)
        max_toks = int(getattr(self._cfg, "max_prompt_tokens_vision", 0) or 0)
        trimmed_prompt = base_prompt
        if max_toks > 0:
            # Đếm thử trên text; nếu dùng Parts, SDK sẽ đếm lại tổng quan trong call chính.
            total = count_tokens_safe(self._client, model, base_prompt)
            if total > max_toks > 0:
                trimmed_prompt = trim_by_chars_approx(base_prompt, max_toks)

        contents = self._build_contents(chunk, trimmed_prompt)

        while True:
            elapsed = time.monotonic() - start
            if attempts >= getattr(self._cfg, "max_retries", 3) or elapsed > getattr(self._cfg, "max_retry_time_sec", 20):
                _LOG.error("Vision give-up (attempts=%s, elapsed=%.2fs). Fallback.", attempts, elapsed)
                return (chunk.raw_text, OpsFlags(ops=["vision_quota_fallback"], flags={"error": True}), 0.8)

            self._limiter.wait(getattr(self._cfg, "jitter_sec", 0.05))

            try:
                resp = _gen(contents, model)

                if _is_blocked_or_unfinished(resp):
                    return (chunk.raw_text, OpsFlags(ops=["safety_block"], flags={"error": True}), 0.78)

                text = getattr(resp, "text", "") or ""
                text = _strip_code_fences(text)
                try:
                    data = json.loads(text)
                except Exception:
                    repair = "JSON ở trên không hợp lệ. In lại duy nhất JSON hợp lệ theo schema."
                    self._limiter.wait(getattr(self._cfg, "jitter_sec", 0.05))
                    resp2 = _gen((repair + "\n---\n" + text), model)
                    if _is_blocked_or_unfinished(resp2):
                        return (chunk.raw_text, OpsFlags(ops=["safety_block"], flags={"error": True}), 0.78)
                    text = getattr(resp2, "text", "") or ""
                    text = _strip_code_fences(text)
                    data = json.loads(text)

                data = self._sanitize(data, chunk.raw_text)
                clean_text, ops, flags = data["clean_text"], data["ops"], data["flags"]
                flags["vision_checked"] = True
                q = 0.92 if clean_text else 0.7
                try: self._limiter.success()
                except Exception: pass
                return (clean_text, OpsFlags(ops=ops, flags=flags), q)

            except ServerError as e:
                attempts += 1
                _LOG.warning("Transient Vision error; retry %d/%d",
                             attempts, getattr(self._cfg, "max_retries", 3))
                time.sleep(getattr(self._cfg, "base_backoff_sec", 0.25) * (1.5 ** (attempts - 1)) + random.uniform(0, getattr(self._cfg, "jitter_sec", 0.05)))
                if attempts >= 2 and len(self._models_cycle) > 1:
                    model_idx = 1 - model_idx; model = self._models_cycle[model_idx]
                    _LOG.warning("Vision switching model → %s", model)
                continue

            except ClientError as e:
                attempts += 1
                ej = getattr(e, "response_json", None)
                code = (ej or {}).get("error", {}).get("code", "")
                status = (ej or {}).get("error", {}).get("status", "")
                rid = getattr(e, "request_id", None)
                if rid is None:
                    try: rid = (ej or {}).get("error", {}).get("metadata", {}).get("requestId") or (ej or {}).get("requestId")
                    except Exception: rid = None
                if code == 429 or status == "RESOURCE_EXHAUSTED":
                    self._limiter.penalize()
                    retry_s = self._parse_retry_delay_seconds(e)
                    if retry_s is None:
                        retry_s = getattr(self._cfg, "base_backoff_sec", 0.25) * (1.5 ** (attempts - 1))
                    retry_s += random.uniform(0, getattr(self._cfg, "jitter_sec", 0.05))
                    _LOG.warning("Vision quota hit; sleeping %.2fs (req_id=%s)", retry_s, rid)
                    time.sleep(retry_s)
                    if len(self._models_cycle) > 1 and attempts >= 2:
                        model_idx = 1 - model_idx; model = self._models_cycle[model_idx]
                        _LOG.warning("Vision switching model after 429 → %s", model)
                    continue
                _LOG.error("Vision client error %s %s (req_id=%s); fallback.", code, status, rid)
                return (chunk.raw_text, OpsFlags(ops=["vision_client_error"], flags={"error": True}), 0.78)

            except Exception:
                attempts += 1
                _LOG.exception("Unknown Vision error; retry %d/%d", attempts, getattr(self._cfg, "max_retries", 3))
                time.sleep(getattr(self._cfg, "base_backoff_sec", 0.25) * (1.5 ** (attempts - 1)) + random.uniform(0, getattr(self._cfg, "jitter_sec", 0.05)))
                continue

    def fix_with_image_batch(self, items: Sequence[Chunk]) -> List[Tuple[str, OpsFlags, float]]:
        return [self._call_fix_once(c) for c in items]
