# pre_tune_app/pipeline/steps/classify_completeness_step.py
from __future__ import annotations
from typing import List, Dict, Any, Sequence
import copy

from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.llm.gemini_classifier import GeminiCompletenessClassifier
from pre_tune_app.domain.sentences import SentenceSCU
from pre_tune_app.domain.models import Chunk  # để tương thích ngược nếu upstream còn trả Chunk

def _coerce_to_scu(items: Sequence[object]) -> List[SentenceSCU]:
    """Chấp nhận SentenceSCU hoặc Chunk; trả về List[SentenceSCU]."""
    out: List[SentenceSCU] = []
    for x in items:
        if isinstance(x, SentenceSCU):
            out.append(x)
        elif isinstance(x, Chunk):
            meta = copy.deepcopy(x.metadata) if x.metadata else {}
            out.append(
                SentenceSCU(
                    id=x.id,
                    document_id=x.document_id or "",
                    page=x.page,
                    offset=x.offset,
                    text=(x.raw_text or ""),
                    chunk_id=x.id,
                    metadata=meta,
                )
            )
        else:
            # vật thể lạ: bỏ qua an toàn
            continue
    return out

class ClassifyCompletenessStep(IPipelineStep):
    """
    Gắn nhãn độ hoàn thiện 0/1 vào SentenceSCU.metadata['complete'].
    KHÔNG lọc câu 0.
    """
    def __init__(self, classifier: GeminiCompletenessClassifier) -> None:
        self._clf = classifier

    def name(self) -> str:
        return "classify_completeness"

    def process(self, sentences: List[SentenceSCU] | List[Chunk], context: Dict[str, Any]) -> List[SentenceSCU]:
        if not sentences:
            return []

        scus: List[SentenceSCU] = _coerce_to_scu(sentences)
        if not scus:
            return []

        texts = [(s.text or "").strip() for s in scus]

        # Nếu tất cả rỗng -> gán 0 hết
        if all(t == "" for t in texts):
            out: List[SentenceSCU] = []
            for s in scus:
                meta = copy.deepcopy(s.metadata) if s.metadata else {}
                meta["complete"] = 0
                out.append(
                    SentenceSCU(
                        id=s.id,
                        document_id=s.document_id,
                        page=s.page,
                        offset=s.offset,
                        text=s.text,
                        chunk_id=s.chunk_id,
                        metadata=meta,
                    )
                )
            return out

        # Gọi LLM: luôn trả List[int] 0/1
        preds = self._clf.classify_batch(texts)

        # Phòng lệch độ dài (hiếm)
        if len(preds) != len(scus):
            if len(preds) < len(scus):
                preds = preds + [0] * (len(scus) - len(preds))
            else:
                preds = preds[: len(scus)]

        out: List[SentenceSCU] = []
        for s, p in zip(scus, preds):
            meta = copy.deepcopy(s.metadata) if s.metadata else {}
            meta["complete"] = 1 if int(p) == 1 else 0
            out.append(
                SentenceSCU(
                    id=s.id,
                    document_id=s.document_id,
                    page=s.page,
                    offset=s.offset,
                    text=s.text,
                    chunk_id=s.chunk_id,
                    metadata=meta,
                )
            )
        return out
