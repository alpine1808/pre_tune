from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List
from pre_tune_app.domain.models import Chunk
from pre_tune_app.pipeline.context import PipelineContext

class IPipelineStep(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def process(self, chunks: List[Chunk] | Any, context: PipelineContext) -> Any: ...
