from __future__ import annotations
from typing import List, Dict, Any
import logging

from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.domain.sentences import SentenceSCU, ContentVariant, ContentGroup

_LOG = logging.getLogger(__name__)

# Optional FAISS
_HAS_FAISS = False
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

def _cosine(u: List[float], v: List[float]) -> float:
    num = sum(a*b for a,b in zip(u,v))
    # vectors are normalized already if using sentence-transformers normalize
    return float(num)

class GroupVariantsStep(IPipelineStep):
    """
    Input: list[(SentenceSCU, embedding)]
    Output: list[ContentGroup] (each is "one content" with many variants)
    """
    def __init__(self, cos_threshold: float = 0.92, same_anchor_only: bool = True) -> None:
        self._thr = cos_threshold
        self._same_anchor_only = same_anchor_only

    def name(self) -> str: return "group_variants"

    def process(self, items: List[tuple[SentenceSCU, List[float]]], context: PipelineContext) -> List[ContentGroup]:
        if not items:
            return []
        if not _HAS_FAISS:
            raise RuntimeError("faiss not installed. Please install faiss-cpu to use GroupVariantsStep.")

        # Build FAISS index (inner product for cosine with normalized vectors)
        d = len(items[0][1])
        index = faiss.IndexFlatIP(d)
        mat = []
        for s, vec in items:
            mat.append(vec)
        import numpy as np
        xb = np.array(mat, dtype='float32')
        index.add(xb)

        # Query kNN for each sentence
        k = min(20, len(items))
        D, I = index.search(xb, k)

        # Union-Find to create groups
        parent = list(range(len(items)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # merge edges above threshold and (optionally) same anchor
        for i,(s_i, v_i) in enumerate(items):
            anc_i = s_i.anchor
            for j_idx, dist in zip(I[i], D[i]):
                if j_idx == i: 
                    continue
                if dist < self._thr:
                    continue
                s_j, _ = items[j_idx]
                if self._same_anchor_only:
                    if (s_i.anchor.article != s_j.anchor.article) or (s_i.anchor.clause != s_j.anchor.clause):
                        continue
                union(i, j_idx)

        # Build groups
        groups_map: Dict[int, List[int]] = {}
        for i in range(len(items)):
            r = find(i)
            groups_map.setdefault(r, []).append(i)

        out: List[ContentGroup] = []
        for gidx, idxs in groups_map.items():
            # choose canonical: longest text
            variants: List[ContentVariant] = []
            longest_len = -1
            canonical_index = 0
            for k_i in idxs:
                s, vec = items[k_i]
                flags: Dict[str, Any] = {}
                variants.append(ContentVariant(sentence=s, completeness=None, flags=flags))
                if len(s.text) > longest_len:
                    longest_len = len(s.text)
                    canonical_index = len(variants)-1
            cg = ContentGroup(
                content_id=f"cg-{gidx}",
                variants=variants,
                canonical_index=canonical_index,
                embedding=list(map(float, items[idxs[0]][1])),
                coverage=0.0,
                flags={}
            )
            out.append(cg)
        return out
