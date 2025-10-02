from __future__ import annotations
from typing import List
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.domain.models import Chunk, ChunkType
from pre_tune_app.domain.sentences import ContentGroup

class PrepareCanonForPreTuneStep(IPipelineStep):
    """
    Convert ContentGroup canonical sentences into synthetic Chunk objects so that
    downstream steps (text_clean, dedupe, package) can be reused without changes.
    """
    def __init__(self, include_all_variants: bool = False) -> None:
        self._include_all = include_all_variants

    def name(self) -> str: return "prepare_canon_for_pretune"

    def process(self, groups: List[ContentGroup], context: PipelineContext) -> List[Chunk]:
        out: List[Chunk] = []
        for g in groups:
            variants = g.variants if self._include_all else [g.canonical]
            for cv in variants:
                s = cv.sentence
                meta = dict(s.metadata or {})
                meta.setdefault("ops", []).append("canonicalize_from_variants" if not self._include_all else "from_variant")
                meta.setdefault("flags", {}).update(cv.flags or {})
                meta["content_group_id"] = g.content_id
                out.append(Chunk(
                    id=s.id,
                    document_id=s.document_id,
                    page=s.page,
                    offset=s.offset,
                    raw_text=s.text,
                    type=ChunkType.TEXT,
                    bbox=None,
                    continues_flag=False,
                    metadata=meta
                ))
        return out
