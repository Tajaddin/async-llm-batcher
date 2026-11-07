"""Asyncio token-bucket rate limiter.

Drops tokens into a bucket at ``rate`` per second, capped at ``capacity``.
Callers ``await acquire()`` before each call. An empty bucket sleeps the
caller until the next refill.

The implementation is deliberately tiny (~40 lines, no external deps) so a
reviewer confirms correctness at a glance.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """Simple async token bucket.

    Parameters
    ----------
    rate:
        Tokens added per second.
    capacity:
        Max tokens the bucket can hold. Defaults to ``max(rate, 1)``: a
        1-second burst, and at least one call even when ``rate`` is below 1/s.
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else max(rate, 1.0))
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + delta * self.rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then decrement."""
        if tokens <= 0:
            return
        if tokens > self.capacity:
            # The bucket never holds that many tokens, so waiting would never end.
            raise ValueError(f"cannot acquire {tokens} tokens from a bucket of capacity {self.capacity}")
        async with self._lock:
            self._refill()
            while self._tokens < tokens:
                deficit = tokens - self._tokens
                wait_secs = deficit / self.rate
                # release lock while sleeping so other waiters proceed in FIFO
                self._lock.release()
                try:
                    await asyncio.sleep(wait_secs)
                finally:
                    await self._lock.acquire()
                self._refill()
            self._tokens -= tokens

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens
