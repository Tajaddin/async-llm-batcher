"""BatchRunner: drives N prompts through an async handler with rate limit + retry + DLQ + checkpoint.

The handler is any ``async (prompt_id, prompt) -> result`` callable. It may
raise :class:`TransientError` to ask for a retry, :class:`PermanentError` to
route straight to DLQ, or any other ``Exception`` which is treated as
transient by default.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from async_llm_batcher.checkpointer import PromptStatus, SqliteCheckpointer
from async_llm_batcher.rate_limit import TokenBucketRateLimiter
from async_llm_batcher.retry import PermanentError, RetryPolicy

Handler = Callable[[str, str], Awaitable[Any]]


_logger = logging.getLogger("async_llm_batcher")


@dataclass
class BatchResult:
    """Summary of a single :meth:`BatchRunner.run` call."""

    n_total: int
    n_attempted: int
    n_succeeded: int
    n_dlq: int
    total_attempts: int
    elapsed_secs: float
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def completion_rate(self) -> float:
        if self.n_attempted == 0:
            return 0.0
        return (self.n_succeeded + self.n_dlq) / self.n_attempted

    @property
    def mean_attempts_per_success(self) -> float:
        if self.n_succeeded == 0:
            return 0.0
        return self.total_attempts / self.n_succeeded


@dataclass
class BatchRunner:
    """Process a batch of prompts with rate limit + retry + DLQ + checkpoint.

    Parameters
    ----------
    handler:
        Async function ``async (prompt_id, prompt) -> result``.
    rate_limit:
        :class:`TokenBucketRateLimiter`. Pass ``None`` to disable rate limiting.
    retry:
        :class:`RetryPolicy`. Defaults to 5 attempts with exponential backoff.
    checkpointer:
        :class:`SqliteCheckpointer`. ``None`` builds an in-memory one (no
        resume across processes).
    concurrency:
        Max number of in-flight handler calls.
    """

    handler: Handler
    rate_limit: TokenBucketRateLimiter | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    checkpointer: SqliteCheckpointer | None = None
    concurrency: int = 10

    def __post_init__(self) -> None:
        if self.checkpointer is None:
            self.checkpointer = SqliteCheckpointer(":memory:")
        self._sem = asyncio.Semaphore(self.concurrency)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def run(self, prompts: Iterable[tuple[str, str]]) -> BatchResult:
        """Process ``(prompt_id, prompt_text)`` tuples.

        Idempotent: prompts already in SUCCESS or DLQ are skipped. Prompts
        stuck in IN_PROGRESS from a previous interrupted run are reset to
        PENDING before this run starts.
        """
        cp = self.checkpointer
        assert cp is not None
        items = list(prompts)
        ids = [pid for pid, _ in items]
        cp.enqueue(ids)
        cp.reset_in_progress()

        pending_ids = set(cp.all_pending_ids())
        to_run = [(pid, text) for pid, text in items if pid in pending_ids]

        total_attempts = 0
        async def _worker(pid: str, text: str) -> None:
            nonlocal total_attempts
            async with self._sem:
                cp.mark_in_progress(pid)
                attempt = 0
                while True:
                    attempt += 1
                    total_attempts += 1
                    if self.rate_limit is not None:
                        await self.rate_limit.acquire(1.0)
                    try:
                        result = await self.handler(pid, text)
                        cp.mark_success(pid, result)
                        return
                    except PermanentError as exc:
                        cp.mark_dlq(pid, f"PermanentError: {exc}")
                        return
                    except Exception as exc:  # noqa: BLE001
                        # Transient or unknown. Count and retry.
                        cp.increment_attempt(pid, error=f"{type(exc).__name__}: {str(exc)[:200]}")
                        if attempt >= self.retry.max_attempts:
                            cp.mark_dlq(pid, f"exhausted retries, last={exc!r}")
                            return
                        await self.retry.sleep(attempt)

        start = time.perf_counter()
        await asyncio.gather(*[_worker(pid, text) for pid, text in to_run])
        elapsed = time.perf_counter() - start

        counts = cp.counts()
        return BatchResult(
            n_total=len(items),
            n_attempted=len(to_run),
            n_succeeded=counts.get(PromptStatus.SUCCESS.value, 0),
            n_dlq=counts.get(PromptStatus.DLQ.value, 0),
            total_attempts=total_attempts,
            elapsed_secs=elapsed,
            counts=counts,
        )

    def dead_letter_queue(self) -> list:
        return self.checkpointer.dead_letters() if self.checkpointer else []
