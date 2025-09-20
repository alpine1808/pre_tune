from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine, Result

from pre_tune_app.domain.models import Chunk, ChunkType, BBox


class IChunksRepository(ABC):
    """Interface để lấy danh sách chunks theo document_id."""
    @abstractmethod
    def get_chunks_by_document_id(self, document_id: str) -> List[Chunk]:
        raise NotImplementedError


class SqlChunksRepository(IChunksRepository):
    """Triển khai Postgres/SQLAlchemy, có xử lý cột reserved keyword."""
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_chunks_by_document_id(self, document_id: str) -> List[Chunk]:
        # Lưu ý: "offset" và "text" là từ khóa → cần quote.
        sql = text("""
            SELECT id, document_id, page, "offset", "text", metadata
            FROM chunks
            WHERE document_id = :doc_id
            ORDER BY page NULLS LAST, "offset" NULLS LAST, id
        """)

        out: List[Chunk] = []
        with self._engine.begin() as conn:
            res: Result = conn.execute(sql, {"doc_id": document_id})
            for r in res.mappings():
                meta: dict[str, Any] = r.get("metadata") or {}

                # Chunk type
                ctype_val = meta.get("type")
                if isinstance(ctype_val, str):
                    try:
                        ctype = ChunkType(ctype_val)
                    except Exception:
                        ctype = ChunkType.TEXT
                else:
                    ctype = ChunkType.TEXT

                # BBox (tùy chọn)
                bbox: Optional[BBox] = None
                bb = meta.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    try:
                        x0, y0, x1, y1 = bb
                        bbox = BBox(float(x0), float(y0), float(x1), float(y1))
                    except Exception:
                        bbox = None

                # continues_flag (tùy chọn)
                cont = meta.get("continues_flag")
                cont_bool: Optional[bool] = None
                if cont is not None:
                    try:
                        cont_bool = bool(cont)
                    except Exception:
                        cont_bool = None

                out.append(Chunk(
                    id=str(r["id"]),
                    document_id=str(r["document_id"]),
                    page=r.get("page"),
                    offset=r.get("offset"),
                    raw_text=r.get("text") or "",
                    type=ctype,
                    bbox=bbox,
                    continues_flag=cont_bool,
                    metadata=meta,
                ))
        return out


# Alias để code mới/cũ đều dùng được nếu có nơi import ChunksRepository
ChunksRepository = SqlChunksRepository

__all__ = [
    "IChunksRepository",
    "SqlChunksRepository",
    "ChunksRepository",
]
