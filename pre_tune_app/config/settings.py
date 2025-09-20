# pre_tune_app/config/settings.py
from __future__ import annotations
import os, logging, time
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
    model_vision: str = "gemini-2.0-flash"
    model_vision_fallback: str = "gemini-2.0-flash-lite"

    # Header-filter LLM (bộ lọc tiêu ngữ/quốc hiệu)
    header_model_primary: str = "gemini-2.0-flash-lite"
    header_model_fallback: str = "gemini-2.0-flash"
    header_rpm: int = 10
    header_max_retries: int = 5
    header_max_retry_time_sec: float = 120.0

    # Gating (bật/tắt bước trong pipeline)
    use_header_filter_llm: bool = True
    # Lưu ý: có thể đặt bằng 2 biến môi trường (ưu tiên theo thứ tự):
    #   1) USE_VISION_GATE (mới, khuyến nghị)
    #   2) PRE_TUNE_USE_VISION (tương thích ngược)
    use_vision_gate: bool = True
    # (các bước khác thường luôn bật qua factory, nhưng thêm ở đây để không lỗi nếu step kiểm tra)
    use_text_clean: bool = True

    # Throttle & Retry (dùng cho Text/Vision clean)
    rpm_text: int = 10
    rpm_vision: int = 8
    max_retries: int = 5
    base_backoff_sec: float = 1.0
    jitter_sec: float = 0.35
    max_retry_time_sec: float = 180.0
    adapt_slowdown_factor: float = 1.4
    min_rpm_floor: int = 2

    # Batching
    max_batch_size: int = 24  # dùng bởi text_clean step

    # Back-compat property cho chỗ gọi cfg.db_url
    @property
    def db_url(self) -> str:
        return self.database_url

    # Loader chính (đọc ENV)
    @staticmethod
    def load() -> "AppConfig":
        return AppConfig(
            database_url=os.getenv("DATABASE_URL", ""),
            dry_run=_bool_env("PRE_TUNE_DRY_RUN", False),
            log_level=os.getenv("LOG_LEVEL", "INFO"),

            # endpoints
            use_vertex=_bool_env("GOOGLE_GENAI_USE_VERTEXAI", False),
            gcp_project=os.getenv("GOOGLE_CLOUD_PROJECT") or None,
            gcp_location=os.getenv("GOOGLE_CLOUD_LOCATION") or None,

            # models
            model_text=os.getenv("PRE_TUNE_MODEL_TEXT", "gemini-2.0-flash-lite"),
            model_text_fallback=os.getenv("PRE_TUNE_MODEL_TEXT_FALLBACK", "gemini-2.0-flash"),
            model_vision=os.getenv("PRE_TUNE_MODEL_VISION", "gemini-2.0-flash"),
            model_vision_fallback=os.getenv("PRE_TUNE_MODEL_VISION_FALLBACK", "gemini-2.0-flash-lite"),

            # header filter llm
            header_model_primary=os.getenv("PRE_TUNE_HEADER_MODEL_PRIMARY", "gemini-2.0-flash-lite"),
            header_model_fallback=os.getenv("PRE_TUNE_HEADER_MODEL_FALLBACK", "gemini-2.0-flash"),
            header_rpm=_int_env("PRE_TUNE_HEADER_RPM", 10),
            header_max_retries=_int_env("PRE_TUNE_HEADER_MAX_RETRIES", 5),
            header_max_retry_time_sec=_float_env("PRE_TUNE_HEADER_MAX_RETRY_TIME_SEC", 120.0),

            # gates
            use_header_filter_llm=_bool_env("PRE_TUNE_USE_HEADER_FILTER_LLM", True),
            # Ưu tiên USE_VISION_GATE; nếu không đặt thì rơi về PRE_TUNE_USE_VISION (giữ nguyên hành vi cũ)
            use_vision_gate=_bool_env("USE_VISION_GATE", _bool_env("PRE_TUNE_USE_VISION", True)),
            use_text_clean=_bool_env("PRE_TUNE_USE_TEXT_CLEAN", True),

            # throttle & retry
            rpm_text=_int_env("PRE_TUNE_RPM_TEXT", 10),
            rpm_vision=_int_env("PRE_TUNE_RPM_VISION", 8),
            max_retries=_int_env("PRE_TUNE_MAX_RETRIES", 5),
            base_backoff_sec=_float_env("PRE_TUNE_BASE_BACKOFF_SEC", 1.0),
            jitter_sec=_float_env("PRE_TUNE_JITTER_SEC", 0.35),
            max_retry_time_sec=_float_env("PRE_TUNE_MAX_RETRY_TIME_SEC", 180.0),
            adapt_slowdown_factor=_float_env("PRE_TUNE_ADAPT_SLOWDOWN", 1.4),
            min_rpm_floor=_int_env("PRE_TUNE_MIN_RPM_FLOOR", 2),

            # batching
            max_batch_size=_int_env("PRE_TUNE_MAX_BATCH_SIZE", 24),
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
