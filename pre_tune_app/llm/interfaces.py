from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List
from pre_tune_app.domain.models import Chunk, OpsFlags

class ITextCleanModel(ABC):
    @abstractmethod
    def clean_text_batch(self, items: Sequence[Chunk]) -> List[Tuple[str, OpsFlags, float]]: ...

class IVisionFixModel(ABC):
    @abstractmethod
    def fix_with_image_batch(self, items: Sequence[Chunk]) -> List[Tuple[str, OpsFlags, float]]: ...
