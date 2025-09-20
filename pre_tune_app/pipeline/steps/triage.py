from __future__ import annotations
from typing import List
from pre_tune_app.domain.models import Chunk, ChunkType
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext

class TriageStep(IPipelineStep):
    def name(self) -> str: return "triage"
    def process(self, chunks: List[Chunk], context: PipelineContext) -> List[Chunk]:
        needs_vision_ids = {c.id for c in chunks if c.type == ChunkType.TABLE}
        context["needs_vision_ids"] = needs_vision_ids
        return chunks
