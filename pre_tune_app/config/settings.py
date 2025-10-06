# pre_tune_app/config/settings.py
from __future__ import annotations
import os, logging
from dataclasses import dataclass

# ---------- helpers ----------
def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


# ---------- config ----------
@dataclass(frozen=True)
class AppConfig:
    # Core
    database_url: str
    dry_run: bool
    log_level: str

    # Gemini keys / endpoints
    gemini_api_key_env: str = "GEMINI_API_KEY"
    use_vertex: bool = False
    gcp_project: str | None = None
    gcp_location: str | None = None

    # Model names (text/vision)
    model_text: str = "gemini-2.0-flash-lite"
    model_text_fallback: str = "gemini-2.0-flash"

    # Header-filter LLM (bộ lọc tiêu ngữ/quốc hiệu)
    header_model_primary: str = "gemini-2.0-flash-lite"
    header_model_fallback: str = "gemini-2.0-flash"
    header_rpm: int = 10
    header_max_retries: int = 5
    header_max_retry_time_sec: float = 120.0

    # Gating (bật/tắt bước trong pipeline)
    use_header_filter_llm: bool = True
    use_text_clean: bool = True  # (giữ để step kiểm tra không lỗi)

    # Throttle & Retry (dùng cho Text/Vision clean)
    rpm_text: int = 10
    max_retries: int = 5
    base_backoff_sec: float = 1.0
    jitter_sec: float = 0.35
    max_retry_time_sec: float = 180.0
    adapt_slowdown_factor: float = 1.4
    min_rpm_floor: int = 2

    # Batching
    max_batch_size: int = 24  # dùng bởi text_clean step

    # Embedding / Grouping / Flow flags
    object_level: str = "chunk"
    embedding_model_name: str = "BAAI/bge-m3"
    group_cosine_t1: float = 0.92
    group_same_anchor_only: bool = True
    use_gemini_merge_assist: bool = True
    group_cosine_t2_assist: float = 0.86
    flatten_lists_to_sentence: bool = True

    # API keys per op
    gemini_api_key_classifier_env: str = "GEMINI_API_KEY_CLASSIFIER"
    gemini_api_key_group_env: str = "GEMINI_API_KEY_GROUP"
    gemini_api_key_header_env: str = "GEMINI_API_KEY_HEADER"
    gemini_api_key_splitter_env: str = "GEMINI_API_KEY_SPLITTER"

    # RPM per op (optional)
    rpm_group_merge: int | None = None
    rpm_classifier: int | None = None

    # Classifier / targets
    classifier_model_id: str | None = None
    pretune_targets: str = 'canonical'
    classifier_max_chars: int = 400
    classifier_max_batch: int = 8

    # Cache / Redis
    cache_ttl_sec: int = 86400
    use_redis_cache: bool = False
    redis_url: str = ""

    # Global RPM override
    rpm_global_gemini: int = 10

    # Back-compat property cho chỗ gọi cfg.db_url
    @property
    def db_url(self) -> str:
        return self.database_url

    # Loader chính (đọc ENV)
    @staticmethod
    def load() -> "AppConfig":
        return AppConfig(
            # Core
            database_url=os.getenv("DATABASE_URL", ""),
            dry_run=_bool_env("PRE_TUNE_DRY_RUN", False),
            log_level=os.getenv("LOG_LEVEL", "INFO"),

            # Endpoints
            use_vertex=_bool_env("GOOGLE_GENAI_USE_VERTEXAI", False),
            gcp_project=os.getenv("GOOGLE_CLOUD_PROJECT") or None,
            gcp_location=os.getenv("GOOGLE_CLOUD_LOCATION") or None,

            # Models
            model_text=os.getenv("PRE_TUNE_MODEL_TEXT", "gemini-2.0-flash-lite"),
            model_text_fallback=os.getenv("PRE_TUNE_MODEL_TEXT_FALLBACK", "gemini-2.0-flash"),

            # Header filter LLM
            header_model_primary=os.getenv("PRE_TUNE_HEADER_MODEL_PRIMARY", "gemini-2.0-flash-lite"),
            header_model_fallback=os.getenv("PRE_TUNE_HEADER_MODEL_FALLBACK", "gemini-2.0-flash"),
            header_rpm=_int_env("PRE_TUNE_HEADER_RPM", 10),
            header_max_retries=_int_env("PRE_TUNE_HEADER_MAX_RETRIES", 5),
            header_max_retry_time_sec=_float_env("PRE_TUNE_HEADER_MAX_RETRY_TIME_SEC", 120.0),

            # Gates
            use_header_filter_llm=_bool_env("PRE_TUNE_USE_HEADER_FILTER_LLM", True),
            use_text_clean=_bool_env("PRE_TUNE_USE_TEXT_CLEAN", True),

            # Throttle & Retry
            rpm_text=_int_env("PRE_TUNE_RPM_TEXT", 10),
            max_retries=_int_env("PRE_TUNE_MAX_RETRIES", 5),
            base_backoff_sec=_float_env("PRE_TUNE_BASE_BACKOFF_SEC", 1.0),
            jitter_sec=_float_env("PRE_TUNE_JITTER_SEC", 0.35),
            max_retry_time_sec=_float_env("PRE_TUNE_MAX_RETRY_TIME_SEC", 180.0),
            adapt_slowdown_factor=_float_env("PRE_TUNE_ADAPT_SLOWDOWN", 1.4),
            min_rpm_floor=_int_env("PRE_TUNE_MIN_RPM_FLOOR", 2),
            rpm_global_gemini=int(os.getenv("RPM_GLOBAL_GEMINI", "10")),

            # Batching
            max_batch_size=_int_env("PRE_TUNE_MAX_BATCH_SIZE", 24),

            # Embedding / Grouping / Flow flags
            object_level=os.getenv("PRE_TUNE_OBJECT_LEVEL", "chunk"),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
            group_cosine_t1=_float_env("GROUP_COSINE_T1", 0.92),
            group_same_anchor_only=_bool_env("GROUP_SAME_ANCHOR_ONLY", True),
            use_redis_cache=_bool_env("USE_REDIS_CACHE", False),
            redis_url=os.getenv("REDIS_URL", ""),
            cache_ttl_sec=_int_env("CACHE_TTL_SEC", 86400),
            flatten_lists_to_sentence=_bool_env("FLATTEN_LISTS_TO_SENTENCE", True),
            use_gemini_merge_assist=_bool_env("USE_GEMINI_MERGE_ASSIST", True),
            group_cosine_t2_assist=_float_env("GROUP_COSINE_T2_ASSIST", 0.86),

            # API keys per op
            gemini_api_key_classifier_env=os.getenv("GEMINI_API_KEY_CLASSIFIER"),
            gemini_api_key_group_env=os.getenv("GEMINI_API_KEY_GROUP"),
            gemini_api_key_header_env=os.getenv("GEMINI_API_KEY_HEADER"),
            gemini_api_key_splitter_env=os.getenv("GEMINI_API_KEY_SPLITTER"),

            # RPM per op (optional)
            rpm_group_merge=_int_env("RPM_GROUP_MERGE", _int_env("PRE_TUNE_RPM_TEXT", 40)),
            rpm_classifier=_int_env("RPM_CLASSIFIER", _int_env("PRE_TUNE_RPM_TEXT", 40)),

            # Classifier / targets
            classifier_model_id=os.getenv("CLASSIFIER_MODEL_ID") or None,
            pretune_targets=os.getenv("PRE_TUNE_TARGETS", "canonical"),
            classifier_max_chars=_int_env("CLASSIFIER_MAX_CHARS", 400),
            classifier_max_batch=_int_env("CLASSIFIER_MAX_BATCH", 8),
        )

    # Alias để tương thích với code cũ (nếu chỗ khác còn gọi from_env)
    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig.load()


# ---------- logging ----------
def setup_logging(level: str | None = None) -> None:
    """
    Cấu hình logging dùng LOG_LEVEL từ env nếu không truyền tham số.
    Ví dụ: LOG_LEVEL=DEBUG/INFO/WARNING/ERROR
    """
    lvl_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, lvl_name, logging.INFO)

    # format thời gian an toàn (tránh ValueError do format sai)
    datefmt = os.getenv("LOG_DATEFMT") or "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt=datefmt,
    )

    # Giảm ồn từ httpx/genai nếu cần
    quiet = _bool_env("LOG_HTTPX_QUIET", True)
    if quiet:
        logging.getLogger("httpx").setLevel(max(lvl, logging.WARNING))
        logging.getLogger("google_genai").setLevel(max(lvl, logging.INFO))
