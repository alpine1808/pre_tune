from __future__ import annotations
import os, json, logging
from typing import List, Sequence, Tuple, Optional, Dict, Any

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.domain.models import Chunk
from pre_tune_app.pipeline.step import IPipelineStep

# NEW: error_handles
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
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GENAI = False

HEADER_SCHEMA = {
    "type": "OBJECT",
    "required": ["action", "removed_lines", "header_confidence", "flags"],
    "properties": {
        "action": {"type": "STRING"},
        "removed_lines": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "header_confidence": {"type": "NUMBER"},
        "flags": {
            "type": "OBJECT",
            "properties": {
                "has_national_header": {"type": "BOOLEAN"},
                "has_org_header": {"type": "BOOLEAN"},
                "uncertain": {"type": "BOOLEAN"}
            },
            "required": []
        }
    }
}

SYSTEM_INSTRUCTION = (
    "Bạn là bộ lọc tiêu ngữ/quốc hiệu tiếng Việt. Chỉ được phép xoá các DÒNG Ở ĐẦU đoạn "
    "(quốc hiệu, khẩu hiệu, tên cơ quan, dòng phân cách...). Không xoá tiêu đề nội dung. "
    "Trả về DUY NHẤT MỘT JSON hợp lệ theo schema."
)

def _build_prompt(text: str, max_lines: int = 10, max_chars: int = 1200, include_org_headers: bool = False) -> str:
    head = (text or "")[:max_chars]
    lines = head.splitlines()[:max_lines]
    numbered = "\n".join(f"[{i}] {ln}" for i, ln in enumerate(lines))
    org_hint = (
        "Có thể xoá tên cơ quan nếu rõ ràng là boilerplate." if include_org_headers
        else "Bỏ qua tên cơ quan trừ khi rất rõ là boilerplate; ưu tiên an toàn."
    )
    return (
        "Chỉ IN DUY NHẤT MỘT JSON hợp lệ theo schema. Không giải thích, không markdown.\n"
        "Nhiệm vụ: Nếu các dòng đầu là tiêu ngữ/quốc hiệu/boilerplate thì xoá; nếu không thì giữ nguyên.\n"
        f"{org_hint}\n"
        "removed_lines phải là chuỗi liên tiếp từ 0..k-1 (prefix-only).\n"
        "Các dòng đầu của đoạn là:\n"
        f"{numbered}"
    )

class VNGovHeaderFilterLLMStep(IPipelineStep):
    def __init__(self, cfg: AppConfig, include_org_headers: bool = False) -> None:
        self._cfg = cfg
        self._include_org_headers = include_org_headers
        self._client = self._maybe_make_client()

        # chọn model & fallback (như cũ)
        self._model_id = os.getenv("PRE_TUNE_MODEL_HEADER", self._cfg.model_text)
        self._fallback_model = os.getenv(
            "PRE_TUNE_MODEL_HEADER_FAILOVER",
            os.getenv("PRE_TUNE_MODEL_TEXT_FAILOVER", "gemini-2.5-flash")
        )
        self._models_cycle: List[str] = [m for m in [self._model_id, self._fallback_model] if m and m != self._model_id]  # second optional
        self._models_cycle = [self._model_id] + self._models_cycle  # đảm bảo model chính đứng trước

        # NEW: limiter + error handler
        self._limiter = make_limiter(cfg, key_prefix="gemini", op="header")
        self._error_handler = make_error_handler_from_cfg(cfg)

    def name(self) -> str:
        return "vn_header_filter_llm"

    def _maybe_make_client(self):
        if self._cfg.dry_run or not _HAS_GENAI:
            _LOG.info("VNGovHeaderFilterLLMStep dry-run or SDK not available; client=None")
            return None
        if self._cfg.use_vertex:
            return genai.Client(
                vertexai=True,
                project=self._cfg.gcp_project or None,
                location=self._cfg.gcp_location or None,
            )
        api_key = self._cfg.gemini_api_key_header_env
        if not api_key:
            _LOG.warning("GEMINI_API_KEY not set; header step will run inert.")
            return None
        return genai.Client(api_key=api_key)

    def _call_once(self, prompt: str, model_id: str):
        cfg = types.GenerateContentConfig(  # type: ignore[attr-defined]
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=HEADER_SCHEMA,
            temperature=0,
        )
        return self._client.models.generate_content(model=model_id, contents=prompt, config=cfg)

    def _detect(self, text: str) -> Optional[dict]:
        if self._client is None:
            return None

        prompt = _build_prompt(text, include_org_headers=self._include_org_headers)

        def _do_call(model_id: str):
            return self._call_once(prompt, model_id)

        # policy theo AppConfig (ưu tiên header_* nếu có)
        policy = make_policy_from_cfg(self._cfg, op="header")
        outcome = execute_with_retry(
            models_cycle=[m for m in self._models_cycle if m],
            policy=policy,
            limiter=self._limiter,
            error_handler=self._error_handler,
            call_fn=_do_call,
            ctx=CallContext(op="header"),
        )

        if not outcome.ok or outcome.response is None:
            return None

        payload = getattr(outcome.response, "text", "") or ""
        try:
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("payload is not dict")
            if "action" not in data or "removed_lines" not in data or "flags" not in data:
                raise ValueError("missing required keys")
            return data
        except Exception:
            return None

    @staticmethod
    def _strip_prefix_by_lines(original_text: str, removed_lines: Sequence[int], max_lines: int = 10) -> Optional[Tuple[str, List[str]]]:
        lines = (original_text or "").splitlines()
        consider = lines[:max_lines]
        if not removed_lines:
            return None
        k = len(removed_lines)
        if list(removed_lines) != list(range(k)):
            return None
        removed_preview = consider[:k]
        kept_head = consider[k:]
        tail = lines[max_lines:]
        return "\n".join(kept_head + tail), removed_preview

    def process(self, chunks: List[Chunk], context: Dict[str, Any]) -> List[Chunk]:
        out: List[Chunk] = []
        for c in chunks:
            should_try = ((c.page is None or (c.page is not None and c.page <= 2)) or
                          (c.offset is not None and c.offset <= 200))
            if not should_try or not c.raw_text:
                out.append(c)
                continue

            data = self._detect(c.raw_text)
            if not data or data.get("action") != "strip_prefix":
                out.append(c)
                continue

            removed_lines = data.get("removed_lines") or []
            if (not isinstance(removed_lines, list)) or any((not isinstance(i, int) or i < 0) for i in removed_lines):
                out.append(c)
                continue

            res = self._strip_prefix_by_lines(c.raw_text, removed_lines)
            if not res:
                out.append(c)
                continue

            new_text, removed_preview = res
            meta = dict(c.metadata or {})
            meta["vn_header_filter_llm"] = {
                "removed": len(removed_preview),
                "lines": removed_preview[:5],
                "model": self._model_id,
                "conf": float(data.get("header_confidence") or 0.0),
            }

            out.append(Chunk(
                id=c.id,
                document_id=c.document_id,
                page=c.page,
                offset=c.offset,
                raw_text=new_text,
                type=c.type,
                bbox=c.bbox,
                continues_flag=c.continues_flag,
                metadata=meta,
            ))
        return out
