# pre_tune_app/error_handles/runner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence
import time, random

from .limiter import Limiter
from .errors import ErrorHandler, GeminiErrorType

@dataclass
class CallPolicy:
    max_retries: int
    max_retry_time_sec: float
    base_backoff_sec: float
    jitter_sec: float
    allow_model_switch: bool = True

@dataclass
class CallContext:
    op: str
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CallOutcome:
    ok: bool
    model_used: Optional[str]
    attempts: int
    elapsed_sec: float
    response: Any = None
    error: Optional[BaseException] = None
    switched_models: int = 0


def make_policy_from_cfg(cfg, *, op: str | None = None) -> CallPolicy:
    if op == "header":
        max_retries = int(getattr(cfg, "header_max_retries", getattr(cfg, "max_retries", 5)) or 5)
        max_retry_time_sec = float(getattr(cfg, "header_max_retry_time_sec",
                                           getattr(cfg, "max_retry_time_sec", 180.0)) or 180.0)
    else:
        max_retries = int(getattr(cfg, "max_retries", 5) or 5)
        max_retry_time_sec = float(getattr(cfg, "max_retry_time_sec", 180.0) or 180.0)

    return CallPolicy(
        max_retries=max_retries,
        max_retry_time_sec=max_retry_time_sec,
        base_backoff_sec=float(getattr(cfg, "base_backoff_sec", 1.0) or 1.0),
        jitter_sec=float(getattr(cfg, "jitter_sec", 0.35) or 0.35),
        allow_model_switch=True,
    )


def _default_backoff(attempt: int, base: float) -> float:
    return base * (1.5 ** max(0, attempt))


def execute_with_retry(
    models_cycle: Sequence[str],
    policy: CallPolicy,
    limiter: Limiter,
    error_handler: ErrorHandler,
    call_fn: Callable[[str], Any],
    ctx: Optional[CallContext] = None,
) -> CallOutcome:
    start = time.time()
    attempts = 0
    model_idx = 0
    switched = 0

    if not models_cycle:
        return CallOutcome(False, None, 0, 0.0, error=RuntimeError("models_cycle is empty"))

    while True:
        now = time.time()
        if attempts > policy.max_retries or (now - start) >= policy.max_retry_time_sec:
            return CallOutcome(False, models_cycle[min(model_idx, len(models_cycle)-1)], attempts, now - start,
                               error=RuntimeError("Retry/Time budget exhausted"))

        model_id = models_cycle[model_idx]

        # Điều tiết RPM toàn cục
        try:
            limiter.wait(policy.jitter_sec)
        except Exception:
            pass

        try:
            resp = call_fn(model_id)
            return CallOutcome(True, model_id, attempts + 1, time.time() - start, response=resp, switched_models=switched)

        except BaseException as exc:
            attempts += 1
            try:
                info = error_handler.extract_info(exc)
                et = error_handler.classify(info)
                decision = error_handler.make_decision(info, attempts - 1, policy.max_retries, len(models_cycle), exc)

                # sleep theo quyết định; nếu None → backoff mặc định
                sleep_s = getattr(decision, "sleep", None)
                if sleep_s is None:
                    sleep_s = _default_backoff(attempts - 1, policy.base_backoff_sec)
                sleep_s = max(0.0, float(sleep_s)) + random.uniform(0.0, policy.jitter_sec)
                time.sleep(sleep_s)

                # quota/infra → giảm tốc tạm thời
                if et in (GeminiErrorType.RATE_LIMIT, GeminiErrorType.UNAVAILABLE):
                    try:
                        limiter.penalize()
                    except Exception:
                        pass

                # đổi model nếu được yêu cầu và cho phép
                if getattr(decision, "switch_model", False) and policy.allow_model_switch and len(models_cycle) > 1:
                    model_idx = (model_idx + 1) % len(models_cycle)
                    switched += 1

            except Exception:
                # nếu error handler lỗi → dùng backoff mặc định
                time.sleep(_default_backoff(attempts - 1, policy.base_backoff_sec))
            # lặp lại cho attempt tiếp theo
