# pre_tune_app/llm/gemini_splitter.py
from __future__ import annotations
from typing import List, Sequence
import json, logging, unicodedata
from pre_tune_app.llm.interfaces import ISentenceSplitter

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.error_handles import (
    make_limiter,
    make_error_handler_from_cfg,
    make_policy_from_cfg,
    execute_with_retry,
    CallContext,
)

_LOG = logging.getLogger(__name__)

# google-genai
try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GENAI = False


def _build_schema_spans():
    if types is None or not hasattr(types, "Schema"):
        return {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["start", "end"],
                "properties": {
                    "start": {"type": "INTEGER"},
                    "end": {"type": "INTEGER"},
                },
            },
        }
    return types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "start": types.Schema(type=types.Type.INTEGER),
                "end": types.Schema(type=types.Type.INTEGER),
            },
            required=["start", "end"],
        ),
    )

SYSTEM_INSTRUCTION = (
    "Bạn là bộ TÁCH CÂU tiếng Việt. Không thêm/xoá/sửa nội dung. "
    "Chỉ xác định ranh giới câu trên chuỗi gốc. "
    "BẮT BUỘC bao phủ toàn bộ văn bản (trừ khoảng trắng đầu/cuối mỗi câu). "
    "Nếu đoạn cuối/đoạn đầu chưa hoàn chỉnh vẫn phải xuất thành một câu. "
    "Xuất DUY NHẤT JSON: mảng các object {\"start\":int, \"end\":int} theo đúng thứ tự; "
    "không chồng lấn; 0 ≤ start < end ≤ len(text)."
)

def _normalize_sentence(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = " ".join(s.split())
    return s.strip()

class GeminiSentenceSplitter(ISentenceSplitter):
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

        # Model & key riêng cho splitter (qua AppConfig)
        primary = cfg.model_text
        fallback = getattr(cfg, "model_text_fallback", None)
        self._models_cycle: List[str] = [m for m in [primary, fallback] if m]

        api_key = cfg.gemini_api_key_splitter_env 
        self._client = None
        if not cfg.dry_run and _HAS_GENAI and api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            if not _HAS_GENAI:
                _LOG.warning("google-genai SDK not available; splitter will fallback to simple split.")
            if not api_key:
                _LOG.warning("GEMINI_API_KEY_SPLITTER not set; splitter will fallback.")

        # error_handles wiring
        self._limiter = make_limiter(cfg, key_prefix="gemini", op="splitter")
        self._error_handler = make_error_handler_from_cfg(cfg)

        # cấu hình tối thiểu
        self._max_out_tokens = 256  # đủ cho >100 câu ngắn

    # -------- core --------
    def split(self, text: str) -> List[str]:
        text = text or ""
        if not self._client:
            return self._fallback_split(text)

        prompt = (
            "Tách câu theo yêu cầu. Chỉ in JSON spans.\n"
            f"Độ dài văn bản: {len(text)} ký tự.\n---\n{text}"
        )

        def _do_call(model_id: str):
            cfg = types.GenerateContentConfig(  # type: ignore[attr-defined]
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=_build_schema_spans(),
                temperature=0,
                max_output_tokens=self._max_out_tokens,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            return self._client.models.generate_content(
                model=model_id, contents=prompt, config=cfg
            )

        policy = make_policy_from_cfg(self._cfg, op="splitter")
        outcome = execute_with_retry(
            models_cycle=[m for m in self._models_cycle if m],
            policy=policy,
            limiter=self._limiter,
            error_handler=self._error_handler,
            call_fn=_do_call,
            ctx=CallContext(op="splitter"),
        )
        if not outcome.ok or outcome.response is None:
            return self._fallback_split(text)

        spans = None
        parsed = getattr(outcome.response, "parsed", None)
        if isinstance(parsed, list):
            spans = parsed
        else:
            raw = (getattr(outcome.response, "text", "") or "").strip()
            try:
                spans = json.loads(raw)
            except Exception:
                return self._fallback_split(text)

        # validate + fix conservative
        norm_spans = []
        last_end = 0
        n = len(text)
        try:
            for it in spans:
                s = int(it.get("start"))
                e = int(it.get("end"))
                if s < 0 or e < 0 or s >= e:
                    continue
                if s < last_end:
                    # overlap → gộp với câu trước
                    s = last_end
                if e > n:
                    e = n
                if s >= e:
                    continue
                # nếu có gap giữa last_end và s → thêm span lấp khoảng trống
                if s > last_end:
                    norm_spans.append((last_end, s))
                norm_spans.append((s, e))
                last_end = e
            # phần còn lại (nếu thiếu) → thêm nốt
            if last_end < n:
                norm_spans.append((last_end, n))
        except Exception:
            return self._fallback_split(text)

        # cắt từ chuỗi gốc, chỉ strip nhẹ
        out = []
        for s, e in norm_spans:
            seg = text[s:e]
            seg_stripped = seg.strip()
            out.append(seg_stripped if seg_stripped else seg)
        # đảm bảo có ít nhất 1 câu
        return out or [text.strip() or text]

    def split_batch(self, texts: Sequence[str]) -> List[List[str]]:
        return [self.split(t) for t in texts]

    # -------- fallback đơn giản --------
    @staticmethod
    def _fallback_split(text: str) -> List[str]:
        # rule nhẹ: tách theo [.?!…] + xuống dòng
        import re
        txt = (text or "").replace("\r\n", "\n")
        # giữ dòng dài, tách mềm theo dấu câu
        parts = re.split(r"(?<=[\.\?\!\u2026])\s+|\n{2,}", txt)
        sents = [_normalize_sentence(p) for p in parts if _normalize_sentence(p)]
        return sents or [txt.strip()]
