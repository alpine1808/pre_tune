from __future__ import annotations
import os, json, logging, random, time
from typing import List, Sequence, Tuple, Optional, Dict, Any

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.domain.models import Chunk
from pre_tune_app.pipeline.step import IPipelineStep

_LOG = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GENAI = False

# Schema nghiêm ngặt: flags có properties để tránh INVALID_ARGUMENT
HEADER_SCHEMA = {
    "type": "OBJECT",
    "required": ["action", "removed_lines", "header_confidence", "flags"],
    "properties": {
        "action": {"type": "STRING"},  # "keep" | "strip_prefix"
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
    """
    Bước tiền xử lý dùng Gemini để nhận diện và loại bỏ tiêu ngữ/quốc hiệu ở ĐẦU chunk.
    - Retry/backoff + failover model khi lỗi tạm thời (503/429/timeout...).
    - Circuit breaker: quá nhiều 503 liên tiếp -> đổi primary model ngay.
    - Nếu lỗi/không chắc -> không xoá (fail-safe).
    - Ghi metadata: c.metadata['vn_header_filter_llm'] = {removed, lines, model, conf}
    """

    def __init__(self, cfg: AppConfig, include_org_headers: bool = False) -> None:
        self._cfg = cfg
        self._include_org_headers = include_org_headers
        self._client = self._maybe_make_client()
        self._model_id = os.getenv("PRE_TUNE_MODEL_HEADER", self._cfg.model_text)

        # reliability
        self._max_retries = int(os.getenv("PRE_TUNE_LLM_MAX_RETRIES", "3"))
        self._backoff_base = float(os.getenv("PRE_TUNE_LLM_BACKOFF_BASE", "0.8"))
        self._fallback_model = os.getenv(
            "PRE_TUNE_MODEL_HEADER_FAILOVER",
            os.getenv("PRE_TUNE_MODEL_TEXT_FAILOVER", "gemini-2.5-flash")
        )
        # circuit breaker + throttling
        self._strike = 0
        self._strike_threshold = int(os.getenv("PRE_TUNE_LLM_CIRCUIT_STRIKES", "2"))
        self._qps = float(os.getenv("PRE_TUNE_LLM_QPS", "0"))  # 0 = unlimited
        self._last_call_ts = 0.0

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
        api_key = os.getenv(self._cfg.gemini_api_key_env)
        if not api_key:
            _LOG.warning("GEMINI_API_KEY not set; header step will run inert.")
            return None
        return genai.Client(api_key=api_key)

    def _call_once(self, prompt: str, model_id: str):
        # QPS throttle
        if self._qps > 0:
            now = time.time()
            min_interval = 1.0 / self._qps
            wait = self._last_call_ts + min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call_ts = time.time()

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

        model_id = self._model_id
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            try:
                resp = self._call_once(
                    _build_prompt(text, include_org_headers=self._include_org_headers),
                    model_id,
                )
                payload = getattr(resp, "text", "") or ""
                data = json.loads(payload)
                if not isinstance(data, dict):
                    raise ValueError("payload is not dict")
                if "action" not in data or "removed_lines" not in data or "flags" not in data:
                    raise ValueError("missing required keys")
                self._strike = 0  # reset strike on success
                return data
            except Exception as e:
                msg = str(e)
                transient = any(k in msg for k in (
                    "UNAVAILABLE","503","500","RESOURCE_EXHAUSTED","429",
                    "Timeout","timeout","deadline","DEADLINE","rate limit"
                ))
                is_last = (attempt == attempts - 1)

                if transient:
                    self._strike += 1
                    if self._strike >= self._strike_threshold and self._fallback_model and self._fallback_model != model_id:
                        _LOG.warning("HeaderFilter CIRCUIT OPEN: switch primary %s -> %s (overloads)", model_id, self._fallback_model)
                        self._model_id = self._fallback_model
                        model_id = self._fallback_model
                        self._strike = 0

                if transient and not is_last:
                    if (attempt == attempts - 2) and self._fallback_model and self._fallback_model != model_id:
                        _LOG.warning("HeaderFilter switching model for final retry: %s -> %s (reason: %s)", model_id, self._fallback_model, msg)
                        model_id = self._fallback_model
                    delay = self._backoff_base * (2 ** attempt) + random.uniform(0, 0.25)
                    _LOG.warning("HeaderFilter transient error: %s; retry %d/%d in %.2fs", msg, attempt + 1, attempts - 1, delay)
                    time.sleep(delay)
                    continue

                _LOG.warning("HeaderFilter LLM error (no strip): %s", e)
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
