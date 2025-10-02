from __future__ import annotations
from typing import List
import numpy as np

from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.domain.sentences import ContentGroup
from pre_tune_app.llm.interfaces import IGroupMergeDecider

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

class GeminiMergeConfirmStep(IPipelineStep):
    """
    Sau FAISS grouping: dùng Gemini để 'gom lại' các ContentGroup rất gần nhau
    (ngưỡng T2 < T1) và lọc biến thể biên trong từng group.
    """
    def __init__(self, decider: IGroupMergeDecider, assist_thr: float = 0.86) -> None:
        self._decider = decider
        self._thr = assist_thr

    def name(self) -> str: return "gemini_merge_confirm"

    def process(self, groups: List[ContentGroup], context: PipelineContext) -> List[ContentGroup]:
        if not groups:
            return groups

        # 1) Gộp giữa các group gần nhau theo embedding canonical
        emb = []
        for g in groups:
            emb.append(np.array(g.embedding or [], dtype=np.float32))
        E = np.vstack(emb)
        n = len(groups)

        # Tìm láng giềng gần cho từng group (top-3) và hỏi Gemini
        merged = [False]*n
        id_map = list(range(n))  # đại diện group sau khi merge
        for i in range(n):
            if merged[i]: 
                continue
            sims = E @ E[i]
            # ứng viên: khác i, trên ngưỡng T2, lấy top-3
            cand_idx = [j for j in np.argsort(-sims)[:6] if j != i and sims[j] >= self._thr]
            if not cand_idx:
                continue
            canonical = groups[i].canonical.sentence.text
            cands = [groups[j].canonical.sentence.text for j in cand_idx]
            decisions = self._decider.confirm_membership(canonical, cands)
            for keep, j in zip(decisions, cand_idx):
                if keep and not merged[j]:
                    # hợp nhất j vào i
                    base_len = len(groups[i].variants)
                    groups[i].variants.extend(groups[j].variants)
                    # giữ canonical dài hơn
                    if len(groups[j].canonical.sentence.text) > len(groups[i].canonical.sentence.text):
                        groups[i].canonical_index = base_len + groups[j].canonical_index
                    merged[j] = True
                    id_map[j] = i

        groups = [g for i,g in enumerate(groups) if not merged[i]]

        # 2) (tùy chọn) lọc biến thể biên trong từng group
        #    Ở đây ta chỉ hỏi Gemini cho các biến thể ngắn hơn nhiều so với canonical.
        filtered: List[ContentGroup] = []
        for g in groups:
            can_text = g.canonical.sentence.text
            short_candidates_idx = [k for k,v in enumerate(g.variants)
                                    if len(v.sentence.text) < max(20, int(0.5*len(can_text))) and k != g.canonical_index]
            if not short_candidates_idx:
                filtered.append(g); continue
            cands = [g.variants[k].sentence.text for k in short_candidates_idx]
            decisions = self._decider.confirm_membership(can_text, cands)
            # giữ lại những biến thể mà Gemini xác nhận "cùng nội dung"
            keep_set = set(short_candidates_idx[i] for i,d in enumerate(decisions) if d == 1)
            new_vars = [v for idx,v in enumerate(g.variants)
                        if idx == g.canonical_index or idx in keep_set or len(v.sentence.text) >= max(20, int(0.5*len(can_text)))]
            # cập nhật canonical_index
            can_pos = [i for i,v in enumerate(new_vars) if v is g.canonical][0] if g.canonical in new_vars else 0
            g.variants = new_vars
            g.canonical_index = can_pos
            filtered.append(g)

        return filtered
