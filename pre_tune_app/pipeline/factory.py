# pre_tune_app/pipeline/factory.py
from __future__ import annotations
import logging
from typing import List

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.pipeline.step import IPipelineStep

from pre_tune_app.pipeline.steps.triage import TriageStep
from pre_tune_app.pipeline.steps.text_clean import TextCleanStep
from pre_tune_app.pipeline.steps.merge_continues import MergeContinuesStep
from pre_tune_app.pipeline.steps.dedupe import DedupeNormalizeStep
from pre_tune_app.pipeline.steps.disambiguation import DisambiguationStep
from pre_tune_app.pipeline.steps.locale_normalize import LocaleNormalizeStep
from pre_tune_app.pipeline.steps.post_validate import PostValidationStep
from pre_tune_app.pipeline.steps.package import PackageStep
from pre_tune_app.pipeline.steps.vn_header_filter_llm import VNGovHeaderFilterLLMStep

from pre_tune_app.llm.gemini_text import GeminiTextModel
# LƯU Ý: GeminiVisionModel sẽ import *bên trong* khi gate bật để tránh khởi tạo/ phụ thuộc không cần thiết.

_LOG = logging.getLogger(__name__)

def build_pipeline(cfg: AppConfig) -> List[IPipelineStep]:
    steps: List[IPipelineStep] = []

    # 1) (Tuỳ chọn) Lọc tiêu ngữ/quốc hiệu trước để giảm nhiễu cho các bước sau
    #    Tôn trọng cờ cfg.use_header_filter_llm, mặc định True trong settings.py
    if getattr(cfg, "use_header_filter_llm", True):
        steps.append(VNGovHeaderFilterLLMStep(cfg=cfg, include_org_headers=False))

    # 2) Phân loại ban đầu
    steps.append(TriageStep())

    # 3) Làm sạch text (luôn có, trừ khi tắt rõ ràng)
    if getattr(cfg, "use_text_clean", True):
        text_model = GeminiTextModel(cfg)
        steps.append(TextCleanStep(text_model, cfg))

    # 4) (Gate) Hiệu chỉnh dựa vào "vision" — chỉ bật khi cfg.use_vision_gate = True
    if getattr(cfg, "use_vision_gate", False):
        try:
            from pre_tune_app.llm.gemini_vision import GeminiVisionModel
            from pre_tune_app.pipeline.steps.vision_fix import VisionFixStep
            vision_model = GeminiVisionModel(cfg)
            steps.append(VisionFixStep(vision_model, cfg))
        except Exception as e:
            # Không làm vỡ pipeline nếu Vision lỗi/thiếu SDK — chỉ cảnh báo và tiếp tục text-only
            _LOG.warning("VisionFixStep is disabled due to init/import failure: %s", e)

    # 5) Hợp nhất các chunk có cờ continues
    steps.append(MergeContinuesStep())

    # 6) Chuẩn hoá + khử trùng lặp
    steps.append(DedupeNormalizeStep())

    # 7) Gỡ nhập nhằng
    steps.append(DisambiguationStep())

    # 8) Chuẩn hoá locale (ví dụ: số, ngày tháng, ký hiệu)
    steps.append(LocaleNormalizeStep())

    # 9) Hậu kiểm
    steps.append(PostValidationStep())

    # 10) Đóng gói output
    steps.append(PackageStep(cfg))

    return steps
