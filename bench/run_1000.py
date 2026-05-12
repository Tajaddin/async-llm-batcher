"""1000-prompt benchmark with injected failures.

Hero metric from project IDEAS.md:

  "Process 1000 prompts against a mocked API with 5% random failure injection
   and a hard rate cap; achieve 100% completion with zero data loss and a
   documented mean retry count. Resume-after-kill demo in the README."

This script:

1. Runs 1000 prompts with a mock handler that injects:
   - 5% transient failures (asks the runner to retry)
   - 0.5% permanent failures (routes straight to DLQ — *expected* loss)
2. Caps the call rate at 50 RPS via TokenBucketRateLimiter.
3. Reports completion rate, mean attempts/success, DLQ size, total time.
4. Simulates a kill-mid-run: stops 30% of the way through, then resumes;
   shows only the remaining prompts are processed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from async_llm_batcher import (
    BatchRunner,
    PermanentError,
    RetryPolicy,
    SqliteCheckpointer,
    TokenBucketRateLimiter,
    TransientError,
)

BENCH = Path(__file__).resolve().parent
DB_PATH = BENCH / "bench.db"
RESULTS = BENCH / "results.json"


def make_handler(transient_rate: float, permanent_rate: float, seed: int = 7):
    rng = random.Random(seed)

    async def handler(pid: str, text: str):
        await asyncio.sleep(0.005 + rng.random() * 0.01)  # 5-15 ms simulated API latency
        r = rng.random()
        if r < permanent_rate:
            raise PermanentError(f"schema rejection for {pid}")
        if r < permanent_rate + transient_rate:
            raise TransientError(f"transient blip on {pid}")
        return f"echo:{text}"

    return handler


async def fresh_run(args, transient_rate, permanent_rate) -> dict:
    if DB_PATH.exists():
        DB_PATH.unlink()
    cp = SqliteCheckpointer(DB_PATH)
    runner = BatchRunner(
        handler=make_handler(transient_rate, permanent_rate, seed=args.seed),
        rate_limit=TokenBucketRateLimiter(rate=args.rate_rps, capacity=10),
        retry=RetryPolicy(max_attempts=6, base_delay_secs=0.05, factor=2.0, max_delay_secs=2.0, jitter_secs=0.05),
        checkpointer=cp,
        concurrency=args.concurrency,
    )
    items = [(f"p{i:04d}", f"prompt {i}") for i in range(args.n_prompts)]
    t0 = time.perf_counter()
    res = await runner.run(items)
    elapsed = time.perf_counter() - t0
    cp.close()
    return {
        "n_total": res.n_total,
        "n_attempted": res.n_attempted,
        "n_succeeded": res.n_succeeded,
        "n_dlq": res.n_dlq,
        "total_attempts": res.total_attempts,
        "completion_rate": round(res.completion_rate, 4),
        "mean_attempts_per_success": round(res.mean_attempts_per_success, 3),
        "elapsed_secs": round(elapsed, 2),
        "throughput_per_sec": round(res.n_attempted / max(elapsed, 1e-9), 1),
        "counts": res.counts,
    }


async def resume_demo(args, transient_rate, permanent_rate) -> dict:
    """Run 30% of the prompts, kill, then resume."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    first_n = int(args.n_prompts * 0.3)
    items_first = [(f"p{i:04d}", f"prompt {i}") for i in range(first_n)]
    items_all = [(f"p{i:04d}", f"prompt {i}") for i in range(args.n_prompts)]

    cp1 = SqliteCheckpointer(DB_PATH)
    runner1 = BatchRunner(
        handler=make_handler(transient_rate, permanent_rate, seed=args.seed),
        rate_limit=TokenBucketRateLimiter(rate=args.rate_rps, capacity=10),
        retry=RetryPolicy(max_attempts=6, base_delay_secs=0.05, factor=2.0, max_delay_secs=2.0, jitter_secs=0.05),
        checkpointer=cp1,
        concurrency=args.concurrency,
    )
    res1 = await runner1.run(items_first)
    cp1.close()
    counts_after_first = res1.counts

    # Resume — fresh checkpointer pointed at the same file, full prompt list.
    cp2 = SqliteCheckpointer(DB_PATH)
    runner2 = BatchRunner(
        handler=make_handler(transient_rate, permanent_rate, seed=args.seed + 1),
        rate_limit=TokenBucketRateLimiter(rate=args.rate_rps, capacity=10),
        retry=RetryPolicy(max_attempts=6, base_delay_secs=0.05, factor=2.0, max_delay_secs=2.0, jitter_secs=0.05),
        checkpointer=cp2,
        concurrency=args.concurrency,
    )
    res2 = await runner2.run(items_all)
    cp2.close()

    return {
        "first_run": {
            "n_attempted": res1.n_attempted,
            "n_succeeded": res1.n_succeeded,
            "n_dlq": res1.n_dlq,
            "counts_after": counts_after_first,
        },
        "resume_run": {
            "n_attempted": res2.n_attempted,  # only newly pending prompts
            "n_succeeded": res2.n_succeeded,
            "n_dlq": res2.n_dlq,
            "counts_after": res2.counts,
        },
        "resume_only_processed_new_prompts": res2.n_attempted <= args.n_prompts - first_n + res1.n_dlq + 5,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--n-prompts", type=int, default=1000)
    p.add_argument("--rate-rps", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--transient-rate", type=float, default=0.05)  # 5%
    p.add_argument("--permanent-rate", type=float, default=0.005)  # 0.5%
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    print(
        f"Running {args.n_prompts} prompts at {args.rate_rps} RPS with "
        f"{args.transient_rate*100:.1f}% transient + {args.permanent_rate*100:.1f}% permanent failure rate...\n"
    )
    fresh = asyncio.run(fresh_run(args, args.transient_rate, args.permanent_rate))
    print("Fresh run:")
    print(json.dumps(fresh, indent=2))

    print("\nResume-after-kill demo:")
    resume = asyncio.run(resume_demo(args, args.transient_rate, args.permanent_rate))
    print(json.dumps(resume, indent=2))

    summary = {"fresh": fresh, "resume_demo": resume, "args": vars(args)}
    RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS}")
    if DB_PATH.exists():
        DB_PATH.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
