from __future__ import annotations
import time, random, logging
from typing import Any

from pre_tune_app.config.settings import AppConfig

_LOG = logging.getLogger(__name__)

# Optional Redis bucket
try:
    from pre_tune_app.infra.ratelimit.redis_bucket import (
        DistributedRedisRateLimiter, RedisBucketConfig
    )
    _HAS_REDIS = True
except Exception:
    DistributedRedisRateLimiter = None  # type: ignore
    RedisBucketConfig = None  # type: ignore
    _HAS_REDIS = False


class _LocalGlobalLimiter:
    """Limiter cục bộ nếu không có Redis. Giữ tương thích với wait(jitter_sec)."""
    def __init__(self, rpm: int, jitter_sec: float = 0.0, floor: int = 1, slowdown: float = 2.0):
        self._rpm = max(1, int(rpm))
        self._floor = max(1, int(floor))
        self._slow = float(slowdown)
        self._min_interval = 60.0 / self._rpm
        self._next_t = 0.0
        self._jitter = float(jitter_sec)

    def wait(self, jitter_sec: float | None = None):
        now = time.monotonic()
        if self._next_t > now:
            time.sleep(self._next_t - now)
        self._next_t = max(now, self._next_t) + self._min_interval
        j = self._jitter if jitter_sec is None else float(jitter_sec)
        if j > 0:
            time.sleep(random.uniform(0.0, j))

    def penalize(self):
        new_rpm = max(int(self._rpm / self._slow), self._floor)
        if new_rpm != self._rpm:
            self._rpm = new_rpm
            self._min_interval = 60.0 / self._rpm

class _RedisLimiterAdapter:
    """Bọc Redis limiter để chấp nhận wait(jitter_sec) & có penalize()."""
    def __init__(self, inner, default_jitter: float):
        self._inner = inner
        self._jitter = float(default_jitter)
    def wait(self, jitter_sec: float | None = None):
        # Redis limiter tự quản lý token; mình thêm jitter mềm ở client cho tương thích.
        self._inner.wait()
        j = self._jitter if jitter_sec is None else float(jitter_sec)
        if j > 0:
            time.sleep(random.uniform(0.0, j))
    def penalize(self):
        try:
            self._inner.penalize()
        except Exception:
            # Nếu inner không có penalize, bỏ qua an toàn.
            pass


def build_global_limiter(cfg: AppConfig) -> Any:
    """
    Trả về limiter toàn cục dùng chung cho MỌI lời gọi Gemini.
    Ưu tiên Redis để đa tiến trình chia sẻ, key = pre_tune:gemini:GLOBAL:{project_id}
    """
    rpm = int(getattr(cfg, "rpm_global_gemini", 10) or 10)
    jitter = float(getattr(cfg, "jitter_sec", 0.0) or 0.0)
    proj = getattr(cfg, "project_id", "local")
    floor = int(getattr(cfg, "min_rpm_floor", 1) or 1)
    slowdown = float(getattr(cfg, "adapt_slowdown_factor", 2.0) or 2.0)

    redis_url = getattr(cfg, "redis_url", None)
    if _HAS_REDIS and redis_url:
        try:
            inner = DistributedRedisRateLimiter(
                url=redis_url,
                cfg=RedisBucketConfig(
                    key_prefix=f"pre_tune:gemini:GLOBAL:{proj}",
                    capacity=float(rpm),
                    refill_per_sec=float(rpm) / 60.0,
                    jitter_ms=int(max(0, int(jitter * 1000))),
                ),
            )
            return _RedisLimiterAdapter(inner, default_jitter=jitter)
        except Exception:
            _LOG.exception("Init GLOBAL Redis limiter failed; fallback to local.")

    return _LocalGlobalLimiter(rpm=rpm, jitter_sec=jitter, floor=floor, slowdown=slowdown)
