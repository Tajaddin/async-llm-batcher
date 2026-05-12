# async-llm-batcher

Asyncio LLM batch runner with rate limiting, exponential-backoff retry, dead-letter queue, and SQLite resume-from-crash state. 1000 prompts at 50 RPS under 5 % transient and 0.5 % permanent failure injection: 991 succeed, 9 DLQ, zero data loss, 1.065 mean attempts per success. Kill mid-run and resume. Only the remaining 700 prompts are processed.

[![ci](https://github.com/Tajaddin/async-llm-batcher/actions/workflows/ci.yml/badge.svg)](https://github.com/Tajaddin/async-llm-batcher/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

## Hero benchmark

`bench/run_1000.py`: 1000 prompts, mock handler with 5% transient failure rate and 0.5% permanent failure rate, token-bucket rate limiter capped at 50 RPS, concurrency 20.

| Metric | Value |
|---|---:|
| Total prompts | 1000 |
| Succeeded | **991** |
| DLQ (permanent + exhausted retries) | 9 |
| **Completion rate** (success + DLQ) | **100.0%**: every prompt accounted for, zero data loss |
| Mean attempts per success | **1.065** |
| Total attempts | 1055 |
| Total wall-clock | 28.7 s |
| Throughput | 34.8 prompts/s (retries pull below the 50 RPS cap) |

### Resume-after-kill demo

```
First run (interrupted at 30%):
  attempted=300  succeeded=297  dlq=3

Resume run (full 1000 prompts queued again):
  attempted=700  (only the remaining ones, the 300 already in SUCCESS/DLQ are skipped)
  succeeded=991  dlq=9   (cumulative)
```

The runner reads the existing SQLite state on startup, resets anything stuck in `IN_PROGRESS` back to `PENDING`, and only enqueues prompts not in a terminal state. **Idempotent by construction.**

Reproduce: `python bench/run_1000.py`. Raw JSON in [`bench/results.json`](bench/results.json).

## Why this exists

Anyone who has shipped batch LLM workloads (summarization, classification, eval) ends up writing the same four things by hand:

1. A concurrency limiter so you don't 429 yourself off the cliff.
2. Exponential-backoff retry with jitter and a hard cap.
3. A dead-letter queue for permanent failures so they don't block the pipeline.
4. Persistent state so a deploy / OOM / SIGTERM doesn't make you start over.

`async-llm-batcher` is exactly those four things, stdlib-only at the core (no `aiohttp` or `tenacity` dependency in the install path), backed by a tiny SQLite checkpointer. Pure-Python, asyncio-native, ~500 lines.

## Architecture

```
                        ┌──────────────────────┐
prompts: [(id, text)] ─▶│   BatchRunner.run()  │
                        ├──────────────────────┤
                        │  asyncio.Semaphore   │  ← concurrency cap
                        │      │               │
                        │      ▼               │
                        │  TokenBucketRL       │  ← rate cap (await acquire())
                        │      │               │
                        │      ▼               │
                        │  handler(id, text)   │  ← your async LLM call
                        │      │               │
                        │      ▼               │
                        │ ┌──────────────────┐ │
                        │ │ TransientError?  │ │  → retry with RetryPolicy
                        │ │ PermanentError?  │ │  → straight to DLQ
                        │ │ exhausted?       │ │  → DLQ
                        │ └──────────────────┘ │
                        │      │               │
                        │      ▼               │
                        │  SqliteCheckpointer  │  ← prompt_id → status
                        │  (resumable)         │
                        └──────────┬───────────┘
                                   ▼
                            BatchResult(
                                n_succeeded,
                                n_dlq,
                                total_attempts,
                                mean_attempts_per_success,
                                completion_rate,
                                ...
                            )
```

## Quickstart

```bash
pip install -e ".[dev]"
```

```python
import asyncio
from async_llm_batcher import (
    BatchRunner, RetryPolicy, SqliteCheckpointer,
    TokenBucketRateLimiter, TransientError, PermanentError,
)

async def my_handler(prompt_id: str, prompt: str):
    # your async LLM call here. Raise TransientError to retry, PermanentError to DLQ.
    response = await anthropic_client.messages.create(...)
    return response.content[0].text

runner = BatchRunner(
    handler=my_handler,
    rate_limit=TokenBucketRateLimiter(rate=50, capacity=10),     # 50 RPS, burst 10
    retry=RetryPolicy(max_attempts=6, base_delay_secs=0.5),
    checkpointer=SqliteCheckpointer("/var/data/batcher.db"),     # durable
    concurrency=20,
)

prompts = [(f"row-{i}", row.text) for i, row in enumerate(data)]
result = await runner.run(prompts)

print(f"Succeeded: {result.n_succeeded}/{result.n_attempted}")
print(f"DLQ: {result.n_dlq}")
print(f"Mean attempts/success: {result.mean_attempts_per_success:.2f}")

# DLQ inspection
for dead in runner.dead_letter_queue():
    print(dead.prompt_id, dead.error)
```

## Failure handling

The handler decides what's retryable:

```python
from async_llm_batcher import TransientError, PermanentError

async def handler(pid, text):
    try:
        resp = await client.messages.create(...)
    except RateLimitError as e:
        # Retryable: backoff and try again.
        raise TransientError(str(e))
    except APIStatusError as e:
        if e.status_code == 400:                # bad input, never going to work
            raise PermanentError(str(e))
        raise TransientError(str(e))            # 5xx, timeouts, etc.
    return resp.content[0].text
```

Anything else (a plain `Exception`) is treated as transient by default. Once `RetryPolicy.max_attempts` is hit, the prompt is moved to DLQ with the last exception attached.

## Resume from crash

Three things make the runner crash-safe:

1. **Every state change writes immediately to SQLite.** No buffered "I'll persist when I get to a milestone."
2. **`IN_PROGRESS` rows are reset to `PENDING` at the start of every run.** A worker killed mid-handler doesn't leak.
3. **`enqueue()` is idempotent.** Re-running the same prompt list is a no-op for already-completed rows.

The result: point a new `BatchRunner` at the same checkpoint file and call `run(prompts)` again. The runner figures out what's left.

```python
# After a SIGTERM:
runner = BatchRunner(
    handler=my_handler,
    checkpointer=SqliteCheckpointer("/var/data/batcher.db"),  # same file
    rate_limit=TokenBucketRateLimiter(rate=50),
    concurrency=20,
)
await runner.run(prompts)   # only re-attempts the prompts not yet SUCCESS/DLQ
```

Covered by `tests/test_runner.py::test_resume_from_disk_after_kill`.

## Configuration

| Knob | Type | Default | Effect |
|---|---|---|---|
| `concurrency` | int | 10 | max in-flight handler calls |
| `rate_limit` | `TokenBucketRateLimiter \| None` | `None` | hard rate cap, in calls/sec |
| `RetryPolicy.max_attempts` | int | 5 | retries before DLQ |
| `RetryPolicy.base_delay_secs` | float | 0.5 | first backoff |
| `RetryPolicy.factor` | float | 2.0 | exponential factor |
| `RetryPolicy.max_delay_secs` | float | 10.0 | cap on a single backoff |
| `RetryPolicy.jitter_secs` | float | 0.25 | added uniform jitter |
| `SqliteCheckpointer(path)` | str/Path | `:memory:` | file path (or `:memory:` for ephemeral) |

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
21 passed in 3.33s
```

`pytest-asyncio` test suite. Covers:

* Rate limiter math (immediate when full, throttles to rate, rejects bad input)
* Checkpointer (enqueue dedup, status transitions, persistence across handles, `IN_PROGRESS` reset)
* Runner (clean run, transient retry, permanent → DLQ, exhausted → DLQ, resume from disk after kill, concurrency cap, rate-limiter integration)

## Project layout

```
.
├── src/async_llm_batcher/
│   ├── __init__.py
│   ├── rate_limit.py    # asyncio token bucket
│   ├── retry.py         # RetryPolicy + TransientError + PermanentError
│   ├── checkpointer.py  # SQLite per-prompt state
│   └── runner.py        # BatchRunner with concurrency + retry + DLQ
├── tests/               # 21 pytest-asyncio cases
└── bench/
    ├── run_1000.py      # 1000-prompt benchmark + resume-after-kill demo
    └── results.json
```

## Caveats

**Single process.** SQLite WAL would let multiple workers share a checkpoint file, but the current schema doesn't use WAL and the runner is single-process by design. For multi-host fan-out, point each worker at a different prompt slice and a shared output store.

**No streaming.** The handler returns a full response. Streaming LLM responses are not wired. A natural v0.2 extension has the handler yield partial chunks and the checkpointer record progress at periodic checkpoints.

**Rate limiter is per-runner.** Two `BatchRunner` instances in the same process each get their own bucket. No shared accounting. For a shared quota across runners, pass the same `TokenBucketRateLimiter` instance to both.

**Async-only handler.** Sync handlers need an `asyncio.to_thread(...)` wrap at the caller. The runner does not auto-wrap, by design. The contract stays narrow.

## License

MIT. See [LICENSE](LICENSE).
