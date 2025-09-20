from typing import Dict, List, Any, Sequence
from pre_tune_app.domain.models import CleanedChunk, Chunk
from pre_tune_app.infra.repositories.documents_repo import IDocumentsRepository
from pre_tune_app.infra.repositories.chunks_repo import IChunksRepository
from pre_tune_app.pipeline.step import IPipelineStep

class PreTuneService:
    def __init__(
        self,
        docs_repo: IDocumentsRepository,
        chunks_repo: IChunksRepository,
        steps: Sequence[IPipelineStep],
    ) -> None:
        self._docs_repo = docs_repo
        self._chunks_repo = chunks_repo
        self._steps = list(steps)

    def run_for_filename(self, filename: str) -> List[CleanedChunk]:
        doc_id = self._docs_repo.get_document_id_by_filename(filename)
        if not doc_id:
            raise SystemExit(f"Không tìm thấy document cho filename='{filename}'")
        chunks: List[Chunk] = self._chunks_repo.get_chunks_by_document_id(doc_id)
        ctx: Dict[str, Any] = {"document_id": doc_id, "filename": filename}

        current: Any = chunks
        for step in self._steps:
            current = step.process(current, ctx)

        if not current or not isinstance(current[0], CleanedChunk):
            raise RuntimeError("Pipeline không kết thúc bằng CleanedChunk.")
        return current
