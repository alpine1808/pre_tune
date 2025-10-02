from __future__ import annotations
from typing import List
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.domain.sentences import ContentGroup
from pre_tune_app.llm.interfaces import ICompletenessClassifier

class ClassifyCompletenessStep(IPipelineStep):
    def __init__(self, clf: ICompletenessClassifier) -> None:
        self._clf = clf
    def name(self) -> str: return "classify_completeness"

    def process(self, groups: List[ContentGroup], context: PipelineContext) -> List[ContentGroup]:
        # classify each variant independently
        texts = []
        idxs = []
        for gi, g in enumerate(groups):
            for vi, v in enumerate(g.variants):
                texts.append(v.sentence.text)
                idxs.append((gi, vi))
        preds = self._clf.classify_batch(texts)
        for (gi, vi), pred in zip(idxs, preds):
            old_v = groups[gi].variants[vi]
            groups[gi].variants[vi] = type(old_v)(**{**old_v.__dict__, 'completeness': int(pred)})

        return groups
