# pre_tune_app/pipeline/factory.py
from __future__ import annotations
import logging
from typing import List

from pre_tune_app.config.settings import AppConfig
from pre_tune_app.pipeline.step import IPipelineStep

from pre_tune_app.pipeline.steps.vn_header_filter_llm import VNGovHeaderFilterLLMStep
from pre_tune_app.pipeline.steps.text_clean import TextCleanStep
from pre_tune_app.pipeline.steps.dedupe import DedupeNormalizeStep
from pre_tune_app.pipeline.steps.post_validate import PostValidationStep
from pre_tune_app.pipeline.steps.package import PackageStep
from pre_tune_app.llm.gemini_text import GeminiTextModel
from pre_tune_app.llm.gemini_splitter import GeminiSentenceSplitter

# SCU / BI core
from pre_tune_app.pipeline.steps.sentence_split import SentenceSplitStep
from pre_tune_app.pipeline.steps.embed_sentences import EmbedSentencesStep
from pre_tune_app.pipeline.steps.group_variants import GroupVariantsStep
from pre_tune_app.pipeline.steps.gemini_merge_confirm import GeminiMergeConfirmStep
from pre_tune_app.llm.gemini_group_merge import GeminiGroupMergeDecider
from pre_tune_app.pipeline.steps.classify_completeness import ClassifyCompletenessStep
from pre_tune_app.pipeline.steps.list_detect_flatten import ListDetectAndFlattenStep
from pre_tune_app.pipeline.steps.prepare_canon_for_pretune import PrepareCanonForPreTuneStep
from pre_tune_app.llm.gemini_classifier import GeminiCompletenessClassifier
from pre_tune_app.pipeline.steps.sentence_stitcher_step import SentenceStitcherStep

_LOG = logging.getLogger(__name__)

def _build_scu_pipeline(cfg: AppConfig) -> List[IPipelineStep]:
    steps: List[IPipelineStep] = []

    # 0) (tuỳ chọn) lọc tiêu ngữ VN (fail-safe, rất hữu ích cho pháp lý VN)
    if getattr(cfg, "use_header_filter_llm", False):
        steps.append(VNGovHeaderFilterLLMStep(cfg))

    # A) Tách câu từ chunks (bảo vệ viết tắt/số thập phân theo triển khai trong step)
    splitter = GeminiSentenceSplitter(cfg)
    steps.append(SentenceSplitStep(cfg, splitter=splitter))

    clf = GeminiCompletenessClassifier(
        cfg,
        model_id=cfg.classifier_model_id,
        api_key_env=cfg.gemini_api_key_classifier_env,
        rpm=cfg.rpm_global_gemini,
    )
    steps.append(ClassifyCompletenessStep(clf))

    # B) Vector hoá câu
    steps.append(EmbedSentencesStep(model_name=cfg.embedding_model_name))

    # C) Gom biến thể bằng FAISS (với tuỳ chọn chỉ gom trong cùng anchor)
    steps.append(
        GroupVariantsStep(
            cos_threshold=cfg.group_cosine_t1,
            same_anchor_only=cfg.group_same_anchor_only,
        )
    )

    # C') (tuỳ chọn) LLM xác nhận các trường hợp biên rìa ngưỡng
    if getattr(cfg, "use_gemini_merge_assist", False):
        gdecider = GeminiGroupMergeDecider(
            cfg,
            model_id=cfg.model_text,
            api_key_env=cfg.gemini_api_key_group_env,
            rpm=cfg.rpm_global_gemini,
        )
        steps.append(GeminiMergeConfirmStep(gdecider, assist_thr=cfg.group_cosine_t2_assist))

    # E) Phát hiện danh sách a) b) c) d) và (tuỳ chọn) flatten
    steps.append(ListDetectAndFlattenStep(enable_flatten=cfg.flatten_lists_to_sentence))
    steps.append(SentenceStitcherStep(
        max_merge=cfg.stitcher_max_merge if hasattr(cfg, "stitcher_max_merge") else 3,
        min_len=cfg.stitcher_min_len if hasattr(cfg, "stitcher_min_len") else 8,
        max_len=cfg.stitcher_max_len if hasattr(cfg, "stitcher_max_len") else 512,
        allow_flattened=False
    ))

    # F) Chuyển canonical → synthetic Chunk để tái dùng các bước hạ nguồn
    steps.append(PrepareCanonForPreTuneStep(include_all_variants=(cfg.pretune_targets == "all")))

    # G) Làm sạch bằng Gemini Text (để sửa bể chữ/space rác lần cuối)
    if getattr(cfg, "use_text_clean", True):
        steps.append(TextCleanStep(GeminiTextModel(cfg), cfg))

    # H) Khử trùng lặp & chuẩn hoá nhẹ
    steps.append(DedupeNormalizeStep())

    # I) Hậu kiểm & đóng gói output thống nhất
    steps.append(PostValidationStep())
    steps.append(PackageStep(cfg))

    return steps


def build_pipeline(cfg: AppConfig) -> List[IPipelineStep]:
    """
    Force SCU-only pipeline. 'object_level' is still logged for transparency,
    but we always build the SCU pipeline to avoid the legacy chunk flow.
    """
    mode = (getattr(cfg, "object_level", "scu") or "scu").lower()
    _LOG.info("Building SCU-only pipeline (requested mode=%s)", mode)
    return _build_scu_pipeline(cfg)
