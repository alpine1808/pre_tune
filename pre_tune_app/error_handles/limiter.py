# pre_tune_app/error_handles/limiter.py
from __future__ import annotations
from typing import Protocol, Optional
import time, random, threading

# --- Optional: Redis bucket y như file gốc ---
_HAS_REDIS_BUCKET = False
try:
    # Giữ đúng import như file gốc để tương thích
    from pre_tune_app.infra.ratelimit.redis_bucket import (
        DistributedRedisRateLimiter, RedisBucketConfig
    )
    _HAS_REDIS_BUCKET = True
except Exception:
    DistributedRedisRateLimiter = None  # type: ignore
    RedisBucketConfig = None  # type: ignore


class Limiter(Protocol):
    """Giao diện limiter tối giản để nơi khác chỉ phụ thuộc vào Protocol này."""
    def wait(self, jitter_sec: float = 0.0) -> None: ...
    def penalize(self) -> None: ...


class _LocalLimiter(Limiter):
    """
    Limiter nội bộ kiểu token-bucket dựa trên RPM (request/min).
    Dùng khi không có Redis bucket.
    """
    def __init__(self, rpm: float, floor: float, slowdown_factor: float):
        self._rpm = max(float(rpm or 0.0), 0.0)
        self._floor = max(float(floor or 0.0), 0.0)
        self._slowdown = max(float(slowdown_factor or 1.0), 1.0)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self, jitter_sec: float = 0.0) -> None:
        if self._rpm <= 0:
            if jitter_sec > 0:
                time.sleep(random.uniform(0.0, jitter_sec))
            return
        with self._lock:
            now = time.time()
            base_gap = 60.0 / max(self._rpm, self._floor)  # giãn cách tối thiểu
            gap = base_gap * self._slowdown
            sleep_s = max(0.0, (self._last + gap) - now)
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._last = max(now, self._last + gap)
        if jitter_sec > 0:
            time.sleep(random.uniform(0.0, jitter_sec))

    def penalize(self) -> None:
        # tăng slowdown nhẹ tạm thời (có trần)
        with self._lock:
            self._slowdown = min(self._slowdown * 1.2, 8.0)


class _RedisBucketLimiter(Limiter):
    """
    Adapter bao quanh DistributedRedisRateLimiter để khớp Protocol Limiter.
    Ưu tiên dùng nếu module redis_bucket khả dụng và có redis_url.
    """
    def __init__(self, url: str, rpm: float, floor: float, key_prefix: str, jitter_sec: float):
        if not _HAS_REDIS_BUCKET or DistributedRedisRateLimiter is None or RedisBucketConfig is None:
            raise RuntimeError("redis bucket module not available")

        # cấu hình token-bucket
        capacity = max(1, int(max(rpm, floor)))
        refill_per_sec = max(0.0, float(rpm) / 60.0)
        jitter_ms = max(0, int((jitter_sec or 0.0) * 1000))

        cfg = RedisBucketConfig(
            capacity=capacity,
            refill_per_sec=refill_per_sec,
            jitter_ms=jitter_ms,
            key_prefix=key_prefix,
        )
        self._limiter = DistributedRedisRateLimiter(url, cfg)

        # nếu lớp gốc có set_key_prefix
        if hasattr(self._limiter, "set_key_prefix"):
            try:
                self._limiter.set_key_prefix(key_prefix)  # type: ignore[attr-defined]
            except Exception:
                pass

        self._fallback_jitter = float(jitter_sec or 0.0)

    def wait(self, jitter_sec: float = 0.0) -> None:
        if hasattr(self._limiter, "wait"):
            self._limiter.wait()
        elif hasattr(self._limiter, "acquire"):
            self._limiter.acquire()  # type: ignore[attr-defined]
        elif hasattr(self._limiter, "throttle"):
            self._limiter.throttle()  # type: ignore[attr-defined]
        else:
            time.sleep(0.05)

        # jitter phía client (trong trường hợp bucket không xử lý jitter)
        j = jitter_sec if jitter_sec is not None else self._fallback_jitter
        if j and j > 0:
            time.sleep(random.uniform(0.0, j))

    def penalize(self) -> None:
        if hasattr(self._limiter, "penalize"):
            try:
                self._limiter.penalize()  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        time.sleep(0.25)


# ---------- helpers đọc AppConfig ----------
def _rpm_for_op(cfg, op: str | None) -> float:
    g = getattr(cfg, "rpm_global_gemini", None)
    try:
        if g is not None and float(g) > 0.0:
            return float(g)
    except Exception:
        pass
    if op == "text":
        v = getattr(cfg, "rpm_text", None)
        if v is not None: return float(v)
    if op == "header":
        v = getattr(cfg, "header_rpm", None)
        if v is not None: return float(v)
    if op == "merge":
        v = getattr(cfg, "rpm_group_merge", None)
        if v is not None: return float(v)
    if op == "classifier":
        v = getattr(cfg, "rpm_classifier", None)
        if v is not None: return float(v)
    # fallback toàn cục
    return float(getattr(cfg, "rpm_global_gemini", 0.0) or 0.0)


def make_limiter(cfg, *, key_prefix: str = "gemini", op: str | None = None) -> Limiter:
    rpm = _rpm_for_op(cfg, op)
    jitter_sec = float(getattr(cfg, "jitter_sec", 0.0) or 0.0)
    floor = float(getattr(cfg, "min_rpm_floor", 1.0) or 1.0)
    slowdown = float(getattr(cfg, "adapt_slowdown_factor", 1.0) or 1.0)
    redis_url = getattr(cfg, "redis_url", "") or None
    project = getattr(cfg, "gcp_project", None) or "default"
    location = getattr(cfg, "gcp_location", None) or "global"

    key = f"{key_prefix}:{project}:{location}{(':'+op) if op else ''}"

    if _HAS_REDIS_BUCKET and redis_url:
        try:
            return _RedisBucketLimiter(redis_url, rpm, floor, key, jitter_sec)
        except Exception:
            # rơi về local nếu khởi tạo Redis limiter thất bại
            pass

    return _LocalLimiter(rpm=rpm, floor=floor, slowdown_factor=slowdown)
