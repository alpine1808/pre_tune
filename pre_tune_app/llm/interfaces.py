from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List
from pre_tune_app.domain.models import Chunk, OpsFlags

class ITextCleanModel(ABC):
    @abstractmethod
    def clean_text_batch(self, items: Sequence[Chunk]) -> List[Tuple[str, OpsFlags, float]]: ...

class ICompletenessClassifier(ABC):
    @abstractmethod
    def classify_batch(self, sentences: Sequence[str]) -> List[int]:
        """Return 1 if complete sentence, 0 if incomplete."""

class IGroupMergeDecider(ABC):
    @abstractmethod
    def confirm_membership(self, canonical: str, candidates: Sequence[str]) -> List[int]:
        """
        For the given canonical sentence and a list of candidate sentences,
        return a list of 1/0 where 1 = same content (merge), 0 = different.
        """        