from __future__ import annotations
import re
from typing import List, Set, Tuple
from pre_tune_app.domain.models import Chunk
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext

_TERMINAL = tuple(".?!…:;”’\"»)]}」』）>")
_HEADER_PAT = re.compile(r"^\s*(Chương|Điều)\b", flags=re.IGNORECASE)
_WS = re.compile(r"[ \t]+")

def _normalize_soft(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _WS.sub(" ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]}])", r"\1", s)
    return s.strip()

def _looks_cut_sentence(prev: str, nxt: str) -> bool:
    if not prev or not nxt:
        return False
    prev = prev.rstrip()
    if prev.endswith(_TERMINAL):
        return False
    if _HEADER_PAT.match(nxt):
        return False
    lead = nxt.lstrip()[:1]
    return bool(lead and (lead.islower() or lead in {"-", "•", "–"}))

def _fix_double_numbering(s: str) -> Tuple[str, bool]:
    before = s
    s = re.sub(r"\b(\d+\.\s*)\1", r"\1", s)
    return s, (s != before)

def _shingles(s: str, k: int = 5) -> Set[str]:
    s = _normalize_soft(s)
    if len(s) < k:
        return {s}
    return {s[i:i+k] for i in range(0, len(s) - k + 1)}

def _near_duplicate(a: str, b: str, k: int = 5, thr: float = 0.85) -> bool:
    sa, sb = _shingles(a, k), _shingles(b, k)
    if not sa or not sb:
        return False
    inter = len(sa & sb)
    union = len(sa | sb)
    j = inter / union if union else 0.0
    return j >= thr

def _is_strong_boundary(s: str) -> bool:
    return bool(_HEADER_PAT.match(s or ""))

class ReliabilityGuardStep(IPipelineStep):
    """
    Hậu xử lý để đầu ra 'có nghĩa':
      - Ghép ranh giới câu dựa trên ops (HEAD_OPEN/TAIL_OPEN) + heuristics.
      - Khử lặp/đè giữa các chunk liền kề (substring & near-duplicate).
      - Vá lỗi '5. 5.'.
    Không sửa nghĩa, không 'chính tả'.
    """

    def name(self) -> str:
        return "reliability_guard"

    def process(self, chunks: List[Chunk], context: PipelineContext) -> List[Chunk]:
        if not chunks:
            return chunks

        # 1) Chuẩn hoá nhẹ + vá numbering
        normalized: List[Chunk] = []
        for c in chunks:
            txt = _normalize_soft(c.raw_text or "")
            txt, fixed_num = _fix_double_numbering(txt)
            meta = dict(c.metadata or {})
            if fixed_num:
                meta.setdefault("ops", []).append("fix_double_numbering")
            normalized.append(type(c)(
                id=c.id, document_id=c.document_id, page=c.page, offset=c.offset,
                raw_text=txt, type=c.type, bbox=c.bbox,
                continues_flag=c.continues_flag, metadata=meta
            ))

        # 2) Ghép ranh giới câu (ưu tiên theo ops)
        fused: List[Chunk] = []
        buf: Chunk | None = None

        def _append_buf(cb: Chunk | None):
            if cb is None: return
            fused.append(cb)

        for c in normalized:
            if buf is None:
                buf = c
                continue

            prev_txt = buf.raw_text or ""
            curr_txt = c.raw_text or ""
            prev_ops = set((buf.metadata or {}).get("ops", []))
            curr_ops = set((c.metadata or {}).get("ops", []))

            if _is_strong_boundary(curr_txt):
                _append_buf(buf); buf = c; continue

            # join nếu có cặp tín hiệu open/close hoặc heuristic chỉ ra câu bị cắt
            need_join = (
                ("TAIL_OPEN" in prev_ops) or
                ("HEAD_OPEN" in curr_ops) or
                bool(getattr(buf, "continues_flag", False)) or
                _looks_cut_sentence(prev_txt, curr_txt)
            )

            if need_join:
                joined = (prev_txt + " " + curr_txt).strip()
                meta = dict(buf.metadata or {})
                meta.setdefault("ops", []).append("join_cut_boundary")
                merged_ids = meta.setdefault("merged_ids", [])
                merged_ids.append(c.id)
                buf = type(c)(
                    id=buf.id, document_id=buf.document_id, page=buf.page, offset=buf.offset,
                    raw_text=_normalize_soft(joined), type=buf.type, bbox=buf.bbox,
                    continues_flag=c.continues_flag, metadata=meta
                )
                continue

            _append_buf(buf)
            buf = c
        _append_buf(buf)

        # 3) Khử lặp/đè: ưu tiên bản dài hơn/đủ dấu kết
        deduped: List[Chunk] = []
        for c in fused:
            if deduped:
                prev = deduped[-1]
                a = (prev.raw_text or "")
                b = (c.raw_text or "")

                # substring logic (ngưỡng độ dài để tránh xoá nhầm mẩu rất ngắn)
                if len(a) >= 20 and a in b:
                    meta = dict(c.metadata or {})
                    meta.setdefault("ops", []).append("replace_with_longer_substring")
                    deduped[-1] = type(c)(
                        id=c.id, document_id=c.document_id, page=c.page, offset=c.offset,
                        raw_text=b, type=c.type, bbox=c.bbox,
                        continues_flag=c.continues_flag, metadata=meta
                    )
                    continue
                if len(b) >= 20 and b in a:
                    meta = dict(prev.metadata or {})
                    meta.setdefault("ops", []).append("drop_shorter_substring_following")
                    deduped[-1] = type(prev)(
                        id=prev.id, document_id=prev.document_id, page=prev.page, offset=prev.offset,
                        raw_text=a, type=prev.type, bbox=prev.bbox,
                        continues_flag=prev.continues_flag, metadata=meta
                    )
                    continue

                # near-duplicate (k-gram)
                if _near_duplicate(a, b, k=5, thr=0.85):
                    meta = dict(prev.metadata or {})
                    meta.setdefault("ops", []).append("drop_near_duplicate_following")
                    deduped[-1] = type(prev)(
                        id=prev.id, document_id=prev.document_id, page=prev.page, offset=prev.offset,
                        raw_text=a, type=prev.type, bbox=prev.bbox,
                        continues_flag=prev.continues_flag, metadata=meta
                    )
                    continue

            deduped.append(c)

        return deduped
