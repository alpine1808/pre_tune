from __future__ import annotations
from typing import List, Dict, Any
import re
from pre_tune_app.domain.models import Chunk
from pre_tune_app.pipeline.step import IPipelineStep

_SENT_END = re.compile(r"[\.!?…]\s*$")
_WS = re.compile(r"\s+")

def _should_merge(prev: Chunk, curr: Chunk) -> bool:
    # Merge nếu prev.flags.continues True hoặc prev kết thúc không có dấu câu kết thúc
    flags_prev = (prev.metadata or {}).get("flags") or {}
    cont_prev = bool(flags_prev.get("continues")) or bool(prev.continues_flag)
    if cont_prev:
        return True
    text_prev = prev.raw_text or ""
    if not _SENT_END.search(text_prev.strip()):
        # nếu prev chưa kết câu và curr bắt đầu bằng chữ thường hoặc dấu nối
        first = (curr.raw_text or "").lstrip()[:1]
        if first and first.islower():
            return True
    return False

def _merge_text(a: str, b: str) -> str:
    if not a:
        return b or ""
    if not b:
        return a or ""
    # Nối bằng khoảng trắng đơn, giữ trật tự
    return _WS.sub(" ", (a.rstrip() + " " + b.lstrip())).strip()

class MergeContinuesStep(IPipelineStep):
    def name(self) -> str:
        return "merge_continues"

    def process(self, chunks: List[Chunk], context: Dict[str, Any]) -> List[Chunk]:
        if not chunks:
            return chunks
        out: List[Chunk] = []
        buf: Chunk | None = None

        def flush():
            nonlocal buf
            if buf is not None:
                out.append(buf)
                buf = None

        for c in chunks:
            if buf is None:
                buf = c
                continue
            if _should_merge(buf, c) and (buf.document_id == c.document_id):
                # gộp vào buf
                merged_text = _merge_text(buf.raw_text, c.raw_text)
                meta = dict(buf.metadata or {})
                # hợp nhất flags/ops nhẹ nếu có
                flags_a = (meta.get("flags") or {}).copy()
                flags_b = (c.metadata or {}).get("flags") or {}
                flags_a["continues"] = bool(flags_b.get("continues", False))
                meta["flags"] = flags_a
                # ghi vết merge
                ops = set((meta.get("ops") or [])) | {"merge_lines"}
                meta["ops"] = sorted(ops)
                buf = Chunk(
                    id=buf.id, document_id=buf.document_id,
                    page=buf.page, offset=buf.offset,
                    raw_text=merged_text, type=buf.type, bbox=buf.bbox,
                    continues_flag=buf.continues_flag, metadata=meta
                )
            else:
                flush()
                buf = c
        flush()
        return out
