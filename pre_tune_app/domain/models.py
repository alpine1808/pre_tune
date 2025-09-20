from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    OTHER = "other"

@dataclass(frozen=True)
class BBox:
    x0: float; y0: float; x1: float; y1: float

@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    page: Optional[int]
    offset: Optional[int]
    raw_text: str
    type: ChunkType = ChunkType.TEXT
    bbox: Optional[BBox] = None
    continues_flag: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class OpsFlags:
    ops: List[str] = field(default_factory=list)
    flags: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CleanedChunk:
    chunk_id: str
    document_id: str
    clean_text: str
    ops: List[str]
    flags: Dict[str, Any]
    quality_score: float
    source_linkage: Dict[str, Any]
    page_range: Tuple[Optional[int], Optional[int]]
    version: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    @classmethod
    def from_pipeline(
        cls,
        *,
        id: str,
        document_id: str,
        page: int | None,
        offset: int | None,
        text: str,
        ops: list[str],
        flags: dict,
        quality_score: float,
        metadata: dict,
    ) -> "CleanedChunk":
        return cls(
            chunk_id=id,
            document_id=document_id,
            clean_text=text or "",
            ops=ops or [],
            flags=flags or {},
            quality_score=float(quality_score or 0.0),
            source_linkage={"page": page, "offset": offset, "metadata": metadata or {}},
            page_range=(page, page),
            version="v1",
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )
