from __future__ import annotations
from typing import List
from pre_tune_app.domain.models import Chunk
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext

class DisambiguationStep(IPipelineStep):
    def name(self) -> str: return "disambiguation"
    def process(self, chunks: List[Chunk], context: PipelineContext) -> List[Chunk]:
        return chunks
