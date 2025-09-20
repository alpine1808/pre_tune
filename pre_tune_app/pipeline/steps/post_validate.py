from __future__ import annotations
from typing import List, Dict, Any
import re
from pre_tune_app.domain.models import Chunk
from pre_tune_app.pipeline.step import IPipelineStep

_ODD = re.compile(r"[^\s\w\.\,\-\–\—\(\)\[\]{}:;\'\"/\\\+\*\=\&\%\$€£°©®™§|!?…ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]")
_MIN_LEN = 5

def _quality_score(text: str, flags: Dict[str, Any]) -> float:
    if not text or len(text.strip()) < _MIN_LEN:
        return 0.6
    bad = 1 if flags.get("error") else 0
    odd_ratio = len(_ODD.findall(text)) / max(1, len(text))
    score = 0.95 - 0.25 * bad - 0.2 * min(odd_ratio, 0.05)
    return max(0.0, min(1.0, score))

class PostValidationStep(IPipelineStep):
    def name(self) -> str:
        return "post_validation"

    def process(self, chunks: List[Chunk], context: Dict[str, Any]) -> List[Chunk]:
        out: List[Chunk] = []
        for c in chunks:
            meta = dict(c.metadata or {})
            flags = meta.get("flags") or {}
            if not isinstance(flags, dict):
                flags = {}
            q = _quality_score(c.raw_text or "", flags)
            meta["quality_score"] = q
            out.append(Chunk(
                id=c.id, document_id=c.document_id, page=c.page, offset=c.offset,
                raw_text=c.raw_text, type=c.type, bbox=c.bbox, continues_flag=c.continues_flag,
                metadata=meta
            ))
        return out
