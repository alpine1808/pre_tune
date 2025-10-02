from __future__ import annotations
from typing import List
import logging

from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.domain.sentences import SentenceSCU

_LOG = logging.getLogger(__name__)

# Optional deps
_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

class EmbedSentencesStep(IPipelineStep):
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self._model_name = model_name
        self._model = None

    def name(self) -> str: return "embed_sentences"

    def _ensure_model(self):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not installed. Please install to use embedding.")
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)

    def process(self, sentences: List[SentenceSCU], context: PipelineContext) -> List[tuple[SentenceSCU, list[float]]]:
        self._ensure_model()
        texts = [s.text for s in sentences]
        vecs = self._model.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
        out: List[tuple[SentenceSCU, list[float]]] = []
        for s, v in zip(sentences, vecs):
            out.append((s, list(map(float, v))))
        return out
