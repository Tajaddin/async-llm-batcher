"""Rate limiter tests."""

from __future__ import annotations

import time

import pytest

from async_llm_batcher import TokenBucketRateLimiter


async def test_acquires_immediately_when_bucket_full() -> None:
    rl = TokenBucketRateLimiter(rate=10, capacity=10)
    start = time.perf_counter()
    await rl.acquire(1)
    assert time.perf_counter() - start < 0.05  # near-instant


async def test_throttles_to_rate() -> None:
    rl = TokenBucketRateLimiter(rate=10, capacity=1)
    # Drain the bucket once, then measure how long 5 more tokens take.
    await rl.acquire(1)
    start = time.perf_counter()
    for _ in range(5):
        await rl.acquire(1)
    elapsed = time.perf_counter() - start
    # 5 tokens at 10/s = ~0.5 s; allow generous slack for asyncio scheduling.
    assert 0.3 < elapsed < 1.5


async def test_invalid_rate_rejected() -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate=0)
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate=-1)


async def test_zero_token_request_is_noop() -> None:
    rl = TokenBucketRateLimiter(rate=10, capacity=10)
    await rl.acquire(0)
    assert rl.available_tokens == pytest.approx(10.0, abs=1.0)
