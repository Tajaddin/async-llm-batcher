"""BatchRunner tests."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from async_llm_batcher import (
    BatchRunner,
    PermanentError,
    RetryPolicy,
    SqliteCheckpointer,
    TokenBucketRateLimiter,
    TransientError,
)


def _fast_retry() -> RetryPolicy:
    return RetryPolicy(max_attempts=4, base_delay_secs=0.01, max_delay_secs=0.05, jitter_secs=0.01)


async def test_all_success_when_handler_never_fails() -> None:
    async def handler(pid, text):
        return f"ok:{text}"

    cp = SqliteCheckpointer(":memory:")
    runner = BatchRunner(handler=handler, retry=_fast_retry(), checkpointer=cp, concurrency=4)
    res = await runner.run([(f"p{i}", f"text{i}") for i in range(10)])
    assert res.n_succeeded == 10
    assert res.n_dlq == 0
    assert res.completion_rate == 1.0


async def test_transient_error_retries_then_succeeds() -> None:
    counts = {"calls": 0}

    async def handler(pid, text):
        counts["calls"] += 1
        if counts["calls"] < 3:
            raise TransientError("blip")
        return text

    cp = SqliteCheckpointer(":memory:")
    runner = BatchRunner(handler=handler, retry=_fast_retry(), checkpointer=cp, concurrency=1)
    res = await runner.run([("p", "hi")])
    assert res.n_succeeded == 1
    assert res.total_attempts == 3


async def test_permanent_error_goes_straight_to_dlq() -> None:
    calls = {"n": 0}

    async def handler(pid, text):
        calls["n"] += 1
        raise PermanentError("schema rejection")

    cp = SqliteCheckpointer(":memory:")
    runner = BatchRunner(handler=handler, retry=_fast_retry(), checkpointer=cp, concurrency=1)
    res = await runner.run([("p", "hi")])
    assert res.n_dlq == 1
    assert calls["n"] == 1  # no retries


async def test_max_retries_exhausted_routes_to_dlq() -> None:
    async def handler(pid, text):
        raise TransientError("never recovers")

    cp = SqliteCheckpointer(":memory:")
    runner = BatchRunner(handler=handler, retry=_fast_retry(), checkpointer=cp, concurrency=1)
    res = await runner.run([("p", "hi")])
    assert res.n_dlq == 1
    assert res.n_succeeded == 0
    assert res.total_attempts == _fast_retry().max_attempts


async def test_resume_skips_completed_prompts() -> None:
    cp = SqliteCheckpointer(":memory:")
    cp.enqueue(["already-done"])
    cp.mark_success("already-done", {"r": 1})

    calls = {"ids": []}

    async def handler(pid, text):
        calls["ids"].append(pid)
        return text

    runner = BatchRunner(handler=handler, retry=_fast_retry(), checkpointer=cp, concurrency=2)
    res = await runner.run([("already-done", "x"), ("new", "y")])
    # Only "new" was actually attempted.
    assert calls["ids"] == ["new"]
    assert res.n_succeeded == 2  # both end in SUCCESS state in DB


async def test_resume_from_disk_after_kill() -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)

    # Simulate "killed mid-run": p1 already SUCCESS, p2 IN_PROGRESS, p3 still PENDING.
    cp1 = SqliteCheckpointer(path)
    cp1.enqueue(["p1", "p2", "p3"])
    cp1.mark_success("p1", {"r": "from-run-1"})
    cp1.mark_in_progress("p2")  # got interrupted while running this
    cp1.close()

    # Run 2: fresh checkpointer reads the same DB. Only p2 (reset from IN_PROGRESS
    # back to PENDING by the runner) and p3 should be attempted.
    seen: list[str] = []

    async def handler(pid, text):
        seen.append(pid)
        return f"done:{pid}"

    cp2 = SqliteCheckpointer(path)
    runner2 = BatchRunner(handler=handler, retry=_fast_retry(), checkpointer=cp2, concurrency=1)
    res = await runner2.run([("p1", "a"), ("p2", "b"), ("p3", "c")])
    assert sorted(seen) == ["p2", "p3"]
    assert res.n_succeeded == 3
    cp2.close()
    path.unlink()


async def test_concurrency_cap_respected() -> None:
    in_flight = {"current": 0, "peak": 0}

    async def handler(pid, text):
        in_flight["current"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
        await asyncio.sleep(0.02)
        in_flight["current"] -= 1
        return text

    runner = BatchRunner(
        handler=handler, retry=_fast_retry(),
        checkpointer=SqliteCheckpointer(":memory:"), concurrency=3,
    )
    await runner.run([(f"p{i}", "x") for i in range(20)])
    assert in_flight["peak"] <= 3


async def test_rate_limiter_actually_slows_us_down() -> None:
    async def handler(pid, text):
        return text

    runner = BatchRunner(
        handler=handler,
        retry=_fast_retry(),
        checkpointer=SqliteCheckpointer(":memory:"),
        concurrency=20,
        rate_limit=TokenBucketRateLimiter(rate=10, capacity=2),
    )
    import time

    start = time.perf_counter()
    res = await runner.run([(f"p{i}", "x") for i in range(20)])
    elapsed = time.perf_counter() - start
    # 20 tokens at 10/s, with initial capacity of 2, should take ~1.8s
    assert res.n_succeeded == 20
    assert elapsed > 1.0


async def test_in_progress_reset_on_resume() -> None:
    cp = SqliteCheckpointer(":memory:")
    cp.enqueue(["a"])
    cp.mark_in_progress("a")

    async def handler(pid, text):
        return "ok"

    runner = BatchRunner(handler=handler, retry=_fast_retry(), checkpointer=cp, concurrency=1)
    res = await runner.run([("a", "x")])
    assert res.n_succeeded == 1
