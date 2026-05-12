# Improvement plan

## What changes

- Rewrite README prose in spartan voice. Strip em dashes, semicolons, and banned filler words. Keep every section (hero benchmark, resume demo, why this exists, architecture, quickstart, failure handling, resume, configuration, tests, layout, caveats). Keep file-tree comments.
- Rename "Limitations" to "Caveats" for cross-repo heading consistency.
- Strip em dashes and inline semicolons from `src/async_llm_batcher/*.py` docstrings and comments. The core code is stdlib-only and behavior-stable, so cleaning prose is safe.
- Strip em dashes from `bench/run_1000.py` comments.
- Pin dev dependencies to exact versions (`pytest==8.3.3`, `pytest-asyncio==0.24.0`, `ruff==0.6.9`, `mypy==1.13.0`). Runtime deps stay empty (stdlib-only).
- Add `.github/workflows/ci.yml` with ruff plus pytest matrix on Python 3.10 / 3.11 / 3.12.
- Re-run the 1000-prompt bench under the pinned dev set to confirm the README numbers still hold.

## Why

- README has 10 em dashes and 2 semicolons. Style rules forbid both.
- Dev pins are `>=` ranges. A future pytest or pytest-asyncio release breaks the suite without warning.
- No CI workflow. The "21 passing tests" claim has no badge.

## Out of scope

- Adding a multi-process WAL mode, streaming handler, or shared-bucket rate limiter. All are listed as caveats.
- Changing the SQLite schema or the BatchRunner public API. Both are part of the value the repo demonstrates.
