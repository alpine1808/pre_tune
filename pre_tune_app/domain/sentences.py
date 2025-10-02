from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

class AnchorLevel(str, Enum):
    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLE = "article"  # Điều
    CLAUSE = "clause"    # Khoản
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class Anchor:
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter, "section": self.section,
            "article": self.article, "clause": self.clause
        }

@dataclass(frozen=True)
class SentenceSCU:
    id: str
    document_id: str
    page: Optional[int]
    offset: Optional[int]
    text: str
    chunk_id: Optional[str] = None
    anchor: Anchor = field(default_factory=Anchor)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["anchor"] = self.anchor.to_dict()
        return d

@dataclass(frozen=True)
class ContentVariant:
    sentence: SentenceSCU
    completeness: Optional[int] = None  # 1=complete, 0=incomplete, None=unknown
    flags: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentGroup:
    content_id: str
    variants: List[ContentVariant] = field(default_factory=list)
    canonical_index: int = 0
    embedding: Optional[List[float]] = None
    coverage: float = 0.0
    topic_id: Optional[str] = None
    topic_label: Optional[str] = None
    flags: Dict[str, Any] = field(default_factory=dict)

    @property
    def canonical(self) -> ContentVariant:
        return self.variants[self.canonical_index]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "canonical_text": self.canonical.sentence.text,
            "variant_count": len(self.variants),
            "coverage": self.coverage,
            "topic_id": self.topic_id,
            "topic_label": self.topic_label,
            "flags": self.flags,
        }

@dataclass
class TopicGroup:
    topic_id: str
    topic_label: Optional[str]
    content_groups: List[ContentGroup] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_label": self.topic_label,
            "content_groups": [cg.to_summary() for cg in self.content_groups],
        }

@dataclass
class ListItem:
    label: str  # 'a)', 'b)' etc.
    group: ContentGroup

@dataclass
class ListBlock:
    id: str
    header: Optional[str]
    items: List[ListItem]
    anchor: Anchor
    flattened_text: Optional[str] = None
    flags: Dict[str, Any] = field(default_factory=dict)
