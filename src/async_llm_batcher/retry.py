"""Retry policy with exponential backoff + jitter.

Distinguishes two failure modes:

* :class:`TransientError`: retryable. Network timeouts, 429s, 5xx.
* :class:`PermanentError`: not retryable. 400s, schema violations.

Anything else (a plain ``Exception``) is treated as transient by default.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass


class TransientError(Exception):
    """Retryable error. The runner backs off and tries again."""


class PermanentError(Exception):
    """Non-retryable error. The runner routes the prompt straight to DLQ."""


@dataclass
class RetryPolicy:
    """Exponential backoff with jitter.

    Total wall time for the worst-case retry storm on max_attempts=5,
    base=0.5, factor=2, max_delay=10 is ~31 s.
    """

    max_attempts: int = 5
    base_delay_secs: float = 0.5
    factor: float = 2.0
    max_delay_secs: float = 10.0
    jitter_secs: float = 0.25

    def delay_for(self, attempt: int) -> float:
        """``attempt`` is 1-indexed. The first retry uses ``base_delay_secs``."""
        if attempt < 1:
            return 0.0
        exp = self.base_delay_secs * (self.factor ** (attempt - 1))
        capped = min(exp, self.max_delay_secs)
        jitter = random.uniform(0, self.jitter_secs)
        return capped + jitter

    async def sleep(self, attempt: int) -> None:
        await asyncio.sleep(self.delay_for(attempt))
