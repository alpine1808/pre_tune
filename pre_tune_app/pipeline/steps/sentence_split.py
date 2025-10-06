# pre_tune_app/pipeline/steps/sentence_split.py
from __future__ import annotations
from typing import List, Dict, Any
import copy

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.domain.models import Chunk
from pre_tune_app.domain.sentences import SentenceSCU
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.llm.gemini_splitter import GeminiSentenceSplitter

class SentenceSplitStep(IPipelineStep):
    """
    Tách mỗi Chunk thành nhiều SentenceSCU bằng GeminiSentenceSplitter.
    Giữ cả câu chưa hoàn chỉnh (nhờ coverage spans ở splitter).
    """
    def __init__(self, cfg: AppConfig, splitter: GeminiSentenceSplitter | None = None) -> None:
        self._cfg = cfg
        self._splitter = splitter or GeminiSentenceSplitter(cfg)

    def name(self) -> str:
        return "sentence_split"

    def process(self, chunks: List[Chunk], context: Dict[str, Any]) -> List[SentenceSCU]:
        out: List[SentenceSCU] = []
        if not chunks:
            return out

        for c in chunks:
            text = c.raw_text or ""
            if not text:
                # vẫn tạo 1 SCU rỗng nếu bạn muốn, mặc định bỏ qua dòng rỗng
                continue

            # splitter đảm bảo không drop nội dung (spans coverage)
            sentences: List[str] = self._splitter.split(text)

            # tạo SentenceSCU cho từng câu
            for i, s in enumerate(sentences):
                s_norm = s.strip()
                if not s_norm:
                    # nếu muốn giữ cả câu trắng, thay vì continue -> vẫn tạo SCU
                    continue
                meta = copy.deepcopy(c.metadata) if c.metadata else {}
                meta["split_from_chunk"] = c.id
                meta["sentence_index"] = i

                out.append(
                    SentenceSCU(
                        id=f"{c.id}#s{i}",
                        document_id=c.document_id or "",
                        page=c.page,
                        offset=c.offset,       # nếu cần offset chính xác theo spans, mình có thể thêm logic suy ra offset
                        text=s_norm,
                        chunk_id=c.id,
                        metadata=meta,
                    )
                )

        return out
