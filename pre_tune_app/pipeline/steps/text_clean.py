from __future__ import annotations
from typing import List
from pre_tune_app.config.settings import AppConfig
from pre_tune_app.domain.models import Chunk, OpsFlags
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.llm.interfaces import ITextCleanModel
from pre_tune_app.domain.models import Chunk, OpsFlags

class TextCleanStep(IPipelineStep):
    def __init__(self, model: ITextCleanModel, cfg: AppConfig) -> None:
        self._model = model
        self._cfg = cfg
    def name(self) -> str: return "text_clean"
    def _coerce_batch_results(self, batch, results):
        safe = []
        results = results if isinstance(results, list) else []
        for idx, c in enumerate(batch):
            r = results[idx] if idx < len(results) else None
            if not (isinstance(r, (list, tuple)) and len(r) == 3 and hasattr(r[1], "ops") and hasattr(r[1], "flags")):
                safe.append((c.raw_text or "", OpsFlags(ops=["fallback_passthrough"], flags={"error": True, "llm_mismatch": True}), 0.0))
            else:
                txt, of, q = r
                q = float(q) if q is not None else 0.0
                safe.append((txt, of, q))
        return safe
    def process(self, chunks: List[Chunk], context: PipelineContext) -> List[Chunk]:
        out: List[Chunk] = []
        for i in range(0, len(chunks), self._cfg.max_batch_size):
            batch = chunks[i:i + self._cfg.max_batch_size]
            results = self._model.clean_text_batch(batch)
            results = self._coerce_batch_results(batch, results)
            for c, (clean_text, opsflags, q) in zip(batch, results):
                meta = dict(c.metadata or {})
                meta.setdefault("ops", []).extend(opsflags.ops)
                meta.setdefault("flags", {}).update(opsflags.flags)
                meta["quality_text_clean"] = q
                out.append(type(c)(
                    id=c.id, document_id=c.document_id, page=c.page, offset=c.offset,
                    raw_text=clean_text, type=c.type, bbox=c.bbox,
                    continues_flag=c.continues_flag, metadata=meta
                ))
        return out
