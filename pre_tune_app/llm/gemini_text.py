from __future__ import annotations
import os, json, time, random, logging
from typing import Sequence, Tuple, List, Dict, Any

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.domain.models import Chunk, OpsFlags
from pre_tune_app.llm.interfaces import ITextCleanModel
from pre_tune_app.infra.ratelimit.global_limiter import build_global_limiter
from pre_tune_app.infra.ratelimit.gemini_error_utils import (
    GeminiErrorHandler,
    GeminiErrorType,
)

_LOG = logging.getLogger(__name__)

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
    from pre_tune_app.llm.utils.tokens import trim_by_chars_approx
except Exception:
    def count_tokens_safe(client, model, contents): return -1
    def trim_by_chars_approx(text: str, max_tokens: int) -> str:
        if not isinstance(text, str): text = str(text)
        return text  # giữ nguyên (non-strict)

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
    "Bạn là công cụ làm sạch câu/SCU canonical tiếng Việt (sau OCR). "
    "Giữ nguyên nội dung, không suy diễn; sửa lỗi ngắt dòng/khoảng trắng/gạch đầu dòng/ký tự lạ; "
    "không thay đổi số/ký hiệu pháp lý; không thêm bớt câu. "
    "Trả về DUY NHẤT một JSON phù hợp schema."
)

class GeminiTextModel(ITextCleanModel):
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._client = self._maybe_make_client()
        self._limiter = build_global_limiter(cfg)
        self._models_cycle: List[str] = [cfg.model_text, cfg.model_text_fallback]
        # Shared error handler for Gemini API calls. Centralises logic for
        # interpreting error codes/statuses and computing backoff behaviour. If
        # configuration does not define base_backoff_sec or jitter_sec these
        # default to 1.0 and 0.0 respectively.
        try:
            self._error_handler: GeminiErrorHandler | None = GeminiErrorHandler(
                base_backoff_sec=getattr(cfg, "base_backoff_sec", 1.0),
                jitter_sec=getattr(cfg, "jitter_sec", 0.0),
            )
        except Exception:
            # Should not happen but if instantiation fails, set to None
            self._error_handler = None
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
    # The parsing of retry delays is now handled centrally by GeminiErrorHandler,
    # so this legacy helper has been removed.

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

    # --- Core call ---
    def _call_clean_once(self, chunk: Chunk) -> Tuple[str, OpsFlags, float]:
        if self._client is None:
            return (
                chunk.raw_text,
                OpsFlags(ops=[], flags={"continues": False}),
                0.8,
            )

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
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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

        base_prompt = (
            "Chỉ trả về DUY NHẤT một JSON đúng schema, không giải thích, không Markdown.\n"
            f"Đoạn cần làm sạch:\n---\n{base_text}"
        )

        while True:
            elapsed = time.monotonic() - start
            if attempts >= self._cfg.max_retries or elapsed > self._cfg.max_retry_time_sec:
                _LOG.error("Text give-up (attempts=%s, elapsed=%.2fs). Fallback passthrough.", attempts, elapsed)
                return (chunk.raw_text, OpsFlags(ops=["quota_fallback"], flags={"error": True}), 0.75)

            self._limiter.wait(getattr(self._cfg, "jitter_sec", 0.0))

            try:
                resp = _gen(base_prompt, model)

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
                q = 0.9 if clean_text else 0.6
                return (clean_text, OpsFlags(ops=ops, flags=flags), q)

            except Exception as e:
                # Generic error handling using shared GeminiErrorHandler. This
                # consolidates logic for rate limits, bad requests, server
                # availability and other client/server errors. If handler is not
                # available (e.g. import failure), fallback to simple
                # exponential backoff.
                attempts += 1
                elapsed = time.monotonic() - start
                if attempts >= self._cfg.max_retries or elapsed > self._cfg.max_retry_time_sec:
                    _LOG.error(
                        "Text give-up after error (attempts=%s, elapsed=%.2fs). Fallback passthrough.",
                        attempts,
                        elapsed,
                    )
                    return (
                        chunk.raw_text,
                        OpsFlags(ops=["quota_fallback"], flags={"error": True}),
                        0.75,
                    )

                # Use error handler if available
                if self._error_handler is not None:
                    try:
                        err_info = self._error_handler.extract_info(e)
                        dec = self._error_handler.make_decision(
                            err_info,
                            attempt=attempts - 1,
                            max_retries=self._cfg.max_retries,
                            models=len([m for m in self._models_cycle if m]),
                            exc=e,
                        )
                        # Penalize global limiter when needed
                        if dec.penalize:
                            self._limiter.penalize()
                        # Switch model if suggested and we have fallback models
                        if dec.switch_model and len(self._models_cycle) > 1:
                            model_idx = (model_idx + 1) % len(self._models_cycle)
                            model = self._models_cycle[model_idx]
                            _LOG.warning(
                                "Text switching model due to %s → %s",
                                dec.error_type,
                                model,
                            )
                        # Sleep for the suggested time (if any)
                        if dec.sleep and dec.sleep > 0:
                            time.sleep(dec.sleep)
                        # If action requires fallback or cannot recover, return passthrough
                        if dec.action in {"bad_request", "fallback", "giveup"}:
                            return (
                                chunk.raw_text,
                                OpsFlags(ops=["client_error"], flags={"error": True}),
                                0.7,
                            )
                        # Otherwise continue loop to retry
                        continue
                    except Exception:
                        # If handler fails, fall back to simple exponential backoff
                        _LOG.exception("Error handler failure; falling back to basic backoff.")
                        pass

                # Fallback: simple exponential backoff when error handler is unavailable
                sleep_s = (
                    self._cfg.base_backoff_sec * (1.5 ** (attempts - 1))
                    + random.uniform(0, getattr(self._cfg, "jitter_sec", 0.0))
                )
                time.sleep(sleep_s)
                # Switch model after two attempts if fallback available
                if attempts >= 2 and len(self._models_cycle) > 1:
                    model_idx = (model_idx + 1) % len(self._models_cycle)
                    model = self._models_cycle[model_idx]
                    _LOG.warning("Text switching model due to repeated error → %s", model)
                continue

    # --- Batch tuần tự (tôn trọng RPM) ---
    def clean_text_batch(self, items: Sequence[Chunk]) -> List[Tuple[str, OpsFlags, float]]:
        return [self._call_clean_once(c) for c in items]
