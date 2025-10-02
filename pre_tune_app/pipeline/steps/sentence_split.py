from __future__ import annotations
import re
from typing import List
from pre_tune_app.domain.models import Chunk
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.domain.sentences import SentenceSCU, Anchor

# Strict sentence splitter: split on '.' regardless, as requested.
# Keep page/offset/chunk linkage.

_DOT = re.compile(r'\.')

# Simple heuristics for legal anchors (Vietnamese):
_RE_ARTICLE = re.compile(r'\bĐiều\s+\d+\b', re.IGNORECASE)
_RE_CLAUSE = re.compile(r'\bKhoản\s+\d+\b', re.IGNORECASE)
_RE_CHAPTER = re.compile(r'\bChương\s+[IVXLC]+\b', re.IGNORECASE)
_RE_SECTION = re.compile(r'\bMục\s+\w+\b', re.IGNORECASE)

def _scan_anchor(text: str) -> Anchor:
    # Lightweight extraction; best-effort only
    chap = _RE_CHAPTER.search(text)
    sec = _RE_SECTION.search(text)
    art = _RE_ARTICLE.search(text)
    cla = _RE_CLAUSE.search(text)
    return Anchor(
        chapter=chap.group(0) if chap else None,
        section=sec.group(0) if sec else None,
        article=art.group(0) if art else None,
        clause=cla.group(0) if cla else None,
    )

class SentenceSplitStep(IPipelineStep):
    def name(self) -> str: return "sentence_split"

    def process(self, chunks: List[Chunk], context: PipelineContext) -> List[SentenceSCU]:
        out: List[SentenceSCU] = []
        doc_id = context.get("document_id") or (chunks[0].document_id if chunks else "")
        idx = 0
        for c in chunks:
            text = (c.raw_text or "").strip()
            if not text:
                continue
            # Strict split on '.', but keep the '.' at end of each sentence
            parts = []
            start = 0
            for m in _DOT.finditer(text):
                end = m.end()
                parts.append(text[start:end])
                start = end
            if start < len(text):
                parts.append(text[start:])

            anchor = _scan_anchor(text)
            for p in parts:
                s = p.strip()
                if not s:
                    continue
                sid = f"scu-{c.id}-{idx}"
                # carry chunk linkage in metadata for later grouping
                meta = dict(c.metadata or {})
                meta["source_linkage"] = {
                    "page": c.page, "offset": c.offset, "chunk_id": c.id
                }
                out.append(SentenceSCU(
                    id=sid, document_id=doc_id, page=c.page, offset=c.offset,
                    text=s, chunk_id=c.id, anchor=anchor, metadata=meta
                ))
                idx += 1
        return out
