from __future__ import annotations
import re
from typing import List
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.domain.sentences import ContentGroup, ListItem

_RE_LABEL = re.compile(r"^([a-zA-Z])\)\s*")

class ListDetectAndFlattenStep(IPipelineStep):
    """
    Detect a) b) c) d) style lists inside ContentGroups and produce a flattened sentence variant,
    while preserving structured items.
    """
    def __init__(self, enable_flatten: bool = True) -> None:
        self._enable_flatten = enable_flatten

    def name(self) -> str: return "list_detect_flatten"

    def process(self, groups: List[ContentGroup], context: PipelineContext) -> List[ContentGroup]:
        if not self._enable_flatten:
            return groups

        out: List[ContentGroup] = []
        i = 0
        while i < len(groups):
            g = groups[i]
            text = g.canonical.sentence.text.strip()
            # Heuristic: if canonical looks like a header ("...:"), try to consume following labeled items
            if text.endswith(":") and i+1 < len(groups):
                # Collect consecutive groups whose canonical starts with label
                items: List[ListItem] = []
                j = i+1
                while j < len(groups):
                    g2 = groups[j]
                    t2 = g2.canonical.sentence.text.strip()
                    m = _RE_LABEL.match(t2)
                    if not m: break
                    label = m.group(1).lower() + ")"
                    items.append(ListItem(label=label, group=g2))
                    j += 1
                if len(items) >= 2:
                    # Build flattened sentence: "Header: a)...., b)...., c)...."
                    parts = []
                    for it in items:
                        parts.append(f"{it.label}{it.group.canonical.sentence.text[len(it.label)+1:].strip()}")
                    flat = text + " " + ", ".join(parts)
                    # Create a new ContentGroup representing the flattened version
                    flat_group = ContentGroup(
                        content_id=f"{g.content_id}-flat",
                        variants=[*g.variants],  # keep header variants but this is just a carrier
                        canonical_index=0,
                        embedding=g.embedding,
                        flags={"flattened_list": True}
                    )
                    # overwrite canonical text via a synthetic variant
                    from pre_tune_app.domain.sentences import SentenceSCU, ContentVariant
                    header_scu = g.canonical.sentence
                    synthetic = SentenceSCU(
                        id=header_scu.id + "-flat",
                        document_id=header_scu.document_id,
                        page=header_scu.page,
                        offset=header_scu.offset,
                        text=flat,
                        chunk_id=header_scu.chunk_id,
                        anchor=header_scu.anchor,
                        metadata=dict(header_scu.metadata or {})
                    )
                    flat_group.variants = [ContentVariant(sentence=synthetic, completeness=None, flags={"flattened": True})]
                    out.append(flat_group)
                    # Skip consumed items but also keep original items/groups for structure
                    out.append(g)  # keep header group
                    out.extend([it.group for it in items])
                    i = j
                    continue
            out.append(g)
            i += 1

        return out
