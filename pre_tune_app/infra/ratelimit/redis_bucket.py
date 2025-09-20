# pre_tune_app/infra/ratelimit/redis_bucket.py
from __future__ import annotations
import time, random
from dataclasses import dataclass

try:
    import redis  # optional
except Exception:
    redis = None

# Atomic token bucket via Lua
REDIS_LUA = """
local tokens_key = KEYS[1]
local ts_key = KEYS[2]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local take = tonumber(ARGV[4])

local last_ts = tonumber(redis.call("GET", ts_key) or now)
local tokens = tonumber(redis.call("GET", tokens_key) or capacity)

local elapsed = math.max(0, now - last_ts) / 1000.0
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
if tokens >= take then
  tokens = tokens - take
  allowed = 1
else
  allowed = 0
end

redis.call("SET", tokens_key, tokens)
redis.call("SET", ts_key, now)
return {allowed, tokens}
"""

@dataclass
class RedisBucketConfig:
    key_prefix: str
    capacity: float               # e.g., rpm
    refill_per_sec: float         # rpm/60
    jitter_ms: int = 120

class DistributedRedisRateLimiter:
    """
    Distributed token bucket (project-level). If Redis not installed/available,
    raise at __init__ so caller can fallback to local limiter.
    """
    def __init__(self, url: str, cfg: RedisBucketConfig):
        if redis is None:
            raise RuntimeError("redis library not installed")
        self._r = redis.from_url(url)
        self._cfg = cfg
        self._lua = self._r.register_script(REDIS_LUA)
        # adaptive targets
        self._target_capacity = cfg.capacity
        self._cooldown_until = 0.0

    def penalize(self, factor: float = 0.5, cooldown_sec: int = 10):
        now = time.time()
        self._cooldown_until = max(self._cooldown_until, now + cooldown_sec)
        self._cfg.capacity = max(1.0, self._cfg.capacity * factor)
        self._cfg.refill_per_sec = self._cfg.capacity / 60.0

    def success(self, warmup: float = 1.1):
        # recover gradually if out of cooldown
        if time.time() < self._cooldown_until:
            return
        self._cfg.capacity = min(self._target_capacity, self._cfg.capacity * warmup)
        self._cfg.refill_per_sec = self._cfg.capacity / 60.0

    def wait(self, take: int = 1):
        while True:
            now_ms = int(time.time() * 1000)
            allowed, _ = self._lua(
                keys=[f"{self._cfg.key_prefix}:tokens", f"{self._cfg.key_prefix}:ts"],
                args=[self._cfg.capacity, self._cfg.refill_per_sec, now_ms, take],
            )
            if int(allowed) == 1:
                time.sleep(random.randint(0, self._cfg.jitter_ms) / 1000.0)  # small jitter
                return
            time.sleep(0.2 + random.random() * 0.2)
