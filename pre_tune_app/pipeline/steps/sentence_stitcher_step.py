from __future__ import annotations
import logging
from typing import List, Dict, Any
import re

from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.domain.sentences import SentenceSCU, ContentVariant, ContentGroup

_LOG = logging.getLogger(__name__)

# Dấu ngắt câu tiếng Việt/Anh
_SENT_END = re.compile(r"[.!?…](['\"”’)]*)$")

class SentenceStitcherStep(IPipelineStep):
    """
    Ghép nối các fragment liền kề thành câu/đoạn hoàn chỉnh (stitch broken fragments).
    - Có cấu hình threshold merge
    - Không merge enum/list hoặc tiêu đề
    - Đánh dấu metadata/log các trường hợp merge
    """
    def __init__(self,
                 max_merge: int = 3,
                 min_len: int = 8,
                 max_len: int = 512,
                 allow_flattened: bool = False):
        self._max_merge = max_merge          # số fragment tối đa merge liên tiếp
        self._min_len = min_len              # độ dài min để xét fragment
        self._max_len = max_len              # không merge vượt quá length này
        self._allow_flattened = allow_flattened

    def name(self) -> str: return "sentence_stitcher"

    def process(self, groups: List[ContentGroup], context: PipelineContext) -> List[ContentGroup]:
        out: List[ContentGroup] = []
        i = 0
        while i < len(groups):
            g = groups[i]
            # Nếu group là flatten list (enum) hoặc tiêu đề: skip
            if not self._allow_flattened and g.flags.get("flattened_list"):
                out.append(g)
                i += 1
                continue

            # Lấy canonical text
            cur_text = g.canonical.sentence.text.strip()
            merged = [g]
            total_len = len(cur_text)
            j = i + 1

            while (
                j < len(groups) and
                len(merged) < self._max_merge
            ):
                next_g = groups[j]
                next_text = next_g.canonical.sentence.text.strip()
                # Không nối enum/list/flatten
                if not self._allow_flattened and next_g.flags.get("flattened_list"):
                    break
                # Không nối nếu current đã kết thúc câu, hoặc next là tiêu đề (bắt đầu bằng "Điều", "Chương", in hoa, số thứ tự...)
                if _SENT_END.search(cur_text):
                    break
                if self._is_likely_header_or_enum(next_text):
                    break
                # Merge nếu chiều dài hợp lý
                if total_len + len(next_text) > self._max_len:
                    break
                merged.append(next_g)
                cur_text = cur_text + " " + next_text
                total_len = len(cur_text)
                j += 1

            # Nếu merge >1 chunk thì build group mới
            if len(merged) > 1:
                _LOG.info(f"Stitcher: Merged {len(merged)} fragments at group {i}-{j-1}")
                # Tạo SentenceSCU và ContentVariant mới cho merged text
                base = merged[0].canonical.sentence
                stitched_text = " ".join(g.canonical.sentence.text.strip() for g in merged)
                stitched_scu = SentenceSCU(
                    id=base.id + "-stitched",
                    document_id=base.document_id,
                    page=base.page,
                    offset=base.offset,
                    text=stitched_text,
                    chunk_id=base.chunk_id,
                    anchor=base.anchor,
                    metadata=dict(base.metadata or {}),
                )
                stitched_var = ContentVariant(sentence=stitched_scu, completeness=None, flags={"stitched": True})
                stitched_group = ContentGroup(
                    content_id=merged[0].content_id + "-stitched",
                    variants=[stitched_var],
                    canonical_index=0,
                    embedding=merged[0].embedding,
                    coverage=0.0,
                    flags={"stitched_from": [g.content_id for g in merged]}
                )
                out.append(stitched_group)
                i += len(merged)
            else:
                out.append(g)
                i += 1
        return out

    def _is_likely_header_or_enum(self, text: str) -> bool:
        # Chặn tiêu đề/quy phạm/chương, số thứ tự hoặc enum list
        if text.isupper() and len(text) < 32:
            return True
        if re.match(r"^(Chương|Điều|Mục|Section)\s+\d+", text, re.I):
            return True
        if re.match(r"^\d+[\.\)]\s*", text):
            return True
        if re.match(r"^[a-zA-Z]\)\s+", text):
            return True
        if len(text) < self._min_len:
            return True
        return False
