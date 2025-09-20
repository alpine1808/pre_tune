from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine, Result


class IDocumentsRepository(ABC):
    """Interface để tra cứu document theo filename (trong cột JSONB data)."""
    @abstractmethod
    def get_document_id_by_filename(self, filename: str) -> Optional[str]:
        raise NotImplementedError


class SqlDocumentsRepository(IDocumentsRepository):
    """Triển khai Postgres/SQLAlchemy, tương thích import cũ."""
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_document_id_by_filename(self, filename: str) -> Optional[str]:
        # data: JSONB, khóa filename nằm ở data->>'filename'
        sql = text("""
            SELECT data->>'id' AS doc_id
            FROM documents
            WHERE COALESCE(data->'metadata'->>'filename', data->>'filename') = :filename
            ORDER BY 1
            LIMIT 1
        """)
        with self._engine.begin() as conn:
            res: Result = conn.execute(sql, {"filename": filename})
            row = res.mappings().fetchone()
            return row["doc_id"] if row else None


# Alias để code mới/cũ đều dùng được nếu có nơi import DocumentsRepository
DocumentsRepository = SqlDocumentsRepository

__all__ = [
    "IDocumentsRepository",
    "SqlDocumentsRepository",
    "DocumentsRepository",
]
