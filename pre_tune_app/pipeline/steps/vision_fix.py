# pre_tune_app/pipeline/steps/vision_fix.py
from __future__ import annotations
from typing import List
import logging

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.domain.models import Chunk, OpsFlags
from pre_tune_app.pipeline.step import IPipelineStep
from pre_tune_app.pipeline.context import PipelineContext
from pre_tune_app.llm.interfaces import IVisionFixModel

_LOG = logging.getLogger(__name__)

class VisionFixStep(IPipelineStep):
    def __init__(self, model: IVisionFixModel, cfg: AppConfig) -> None:
        self._model = model
        self._cfg = cfg

    def name(self) -> str:
        return "vision_fix"

    def _coerce_batch_results(self, batch, results):
        """
        Chuẩn hoá kết quả trả về từ model về dạng List[Tuple[text, OpsFlags, quality]],
        đảm bảo không làm vỡ pipeline nếu LLM trả về thiếu/mismatch.
        """
        safe = []
        results = results if isinstance(results, list) else []
        for idx, c in enumerate(batch):
            r = results[idx] if idx < len(results) else None
            if not (isinstance(r, (list, tuple)) and len(r) == 3 and hasattr(r[1], "ops") and hasattr(r[1], "flags")):
                safe.append(
                    (
                        c.raw_text or "",
                        OpsFlags(ops=["fallback_passthrough"], flags={"error": True, "llm_mismatch": True}),
                        0.0,
                    )
                )
            else:
                txt, of, q = r
                q = float(q) if q is not None else 0.0
                safe.append((txt, of, q))
        return safe

    def _append_ops_flags(self, meta: dict, of: OpsFlags) -> dict:
        """
        Gộp ops/flags vào metadata một cách an toàn (ép kiểu).
        """
        meta = dict(meta or {})

        # ops: luôn là list
        ops_existing = meta.get("ops", [])
        if isinstance(ops_existing, list):
            ops_list = ops_existing
        elif ops_existing is None:
            ops_list = []
        else:
            ops_list = [ops_existing]
        ops_list.extend(list(getattr(of, "ops", []) or []))
        meta["ops"] = ops_list

        # flags: luôn là dict
        flags_existing = meta.get("flags", {})
        if not isinstance(flags_existing, dict):
            flags_existing = {}
        of_flags = getattr(of, "flags", {}) or {}
        if isinstance(of_flags, dict):
            flags_existing.update(of_flags)
        meta["flags"] = flags_existing

        return meta

    def process(self, chunks: List[Chunk], context: PipelineContext) -> List[Chunk]:
        # Gate OFF: text-only mode
        if not getattr(self._cfg, "use_vision_gate", False):
            return chunks

        # Lấy danh sách id cần xử lý vision (nếu không có -> bỏ qua)
        try:
            target_ids = context.get("needs_vision_ids", set())  # type: ignore[attr-defined]
        except Exception:
            target_ids = set()

        if not target_ids:
            return chunks

        targets = [c for c in chunks if c.id in target_ids]
        passthrough = [c for c in chunks if c.id not in target_ids]

        if not targets:
            return chunks

        fixed: List[Chunk] = []
        batch_size = int(getattr(self._cfg, "max_batch_size", 16) or 16)

        for i in range(0, len(targets), batch_size):
            batch = targets[i : i + batch_size]
            try:
                results = self._model.fix_with_image_batch(batch)
            except Exception as e:
                _LOG.warning("Vision model batch failed (%s). Falling back passthrough for this batch.", e)
                results = []

            results = self._coerce_batch_results(batch, results)

            for c, (clean_text, opsflags, q) in zip(batch, results):
                meta = self._append_ops_flags(c.metadata, opsflags)
                meta["quality_vision_fix"] = float(q)

                fixed.append(
                    type(c)(
                        id=c.id,
                        document_id=c.document_id,
                        page=c.page,
                        offset=c.offset,
                        raw_text=clean_text,
                        type=c.type,
                        bbox=c.bbox,
                        continues_flag=c.continues_flag,
                        metadata=meta,
                    )
                )

        # Giữ nguyên thứ tự gốc
        merged = {c.id: c for c in passthrough + fixed}
        return [merged[c.id] for c in chunks]
