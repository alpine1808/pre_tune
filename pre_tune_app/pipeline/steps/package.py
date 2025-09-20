from __future__ import annotations
from typing import List, Dict, Any
from pre_tune_app.domain.models import Chunk, CleanedChunk
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.config.settings import AppConfig

class PackageStep(IPipelineStep):
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

    def name(self) -> str:
        return "package"

    def process(self, chunks: List[Chunk], context: Dict[str, Any]) -> List[CleanedChunk]:
        out: List[CleanedChunk] = []
        for c in chunks:
            meta = dict(c.metadata or {})
            flags = meta.get("flags") or {}
            ops = meta.get("ops") or []
            quality = float(meta.get("quality_score") or 0.0)
            cc = CleanedChunk.from_pipeline(
                id=c.id,
                document_id=c.document_id,
                page=c.page,
                offset=c.offset,
                text=c.raw_text or "",
                ops=ops,
                flags=flags,
                quality_score=quality,
                metadata=meta,
            )
            out.append(cc)
        return out
