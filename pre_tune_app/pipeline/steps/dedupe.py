from __future__ import annotations
from typing import List, Dict, Any, Set, Tuple
import hashlib, unicodedata, re
from pre_tune_app.domain.models import Chunk
from pre_tune_app.pipeline.step import IPipelineStep

_WS = re.compile(r"\s+")

def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip()
    s = _WS.sub(" ", s)
    return s

def _hash_norm(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

_ALLOWED_OPS = {
    "remove_header","merge_lines","fix_bullets","fix_hyphenation",
    "normalize_space","fallback","vision_fix"
}

def _normalize_ops(meta: Dict[str, Any]) -> None:
    ops = meta.get("ops") or []
    if not isinstance(ops, list):
        ops = [ops]
    ops = [str(x).strip().lower().replace(" ", "_") for x in ops if x]
    ops = [x for x in ops if x in _ALLOWED_OPS]
    meta["ops"] = sorted(set(ops))

def _default_flags(meta: Dict[str, Any]) -> None:
    flags = meta.get("flags") or {}
    if not isinstance(flags, dict):
        flags = {}
    flags.setdefault("continues", False)
    flags.setdefault("error", False)
    meta["flags"] = flags

class DedupeNormalizeStep(IPipelineStep):
    def name(self) -> str:
        return "dedupe_normalize"

    def process(self, chunks: List[Chunk], context: Dict[str, Any]) -> List[Chunk]:
        seen: Set[Tuple[str, int | None, str]] = set()
        out: List[Chunk] = []
        for c in chunks:
            meta = dict(c.metadata or {})
            _normalize_ops(meta)
            _default_flags(meta)

            t = _norm_text(c.raw_text)
            h = _hash_norm(t)
            key = (c.document_id, c.page, h)
            if key in seen:
                # ghi vết trùng để QA, nhưng bỏ chunk
                dropped = meta.get("dropped_as_duplicate", 0)
                meta["dropped_as_duplicate"] = dropped + 1
                # không append
                continue
            seen.add(key)
            out.append(Chunk(
                id=c.id, document_id=c.document_id, page=c.page, offset=c.offset,
                raw_text=t, type=c.type, bbox=c.bbox, continues_flag=c.continues_flag,
                metadata=meta
            ))
        return out
