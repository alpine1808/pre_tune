from __future__ import annotations
import hashlib
from typing import Optional

class RedisCache:
    def __init__(self, url: str, prefix: str = "pretune:", ttl_sec: int = 86400, enabled: bool = False):
        self.enabled = bool(enabled and url)
        self.prefix = prefix
        self.ttl = ttl_sec
        self._r = None
        if self.enabled:
            import redis  # lazy import
            self._r = redis.Redis.from_url(url, decode_responses=True)

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        return self._r.get(self._k(key))

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if not self.enabled:
            return
        self._r.setex(self._k(key), int(ttl or self.ttl), value)

    @staticmethod
    def sha1(*parts: str) -> str:
        h = hashlib.sha1()
        for p in parts:
            h.update(p.encode("utf-8"))
        return h.hexdigest()
