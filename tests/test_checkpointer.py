"""Checkpointer tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

from async_llm_batcher import PromptStatus, SqliteCheckpointer


def test_enqueue_only_inserts_missing() -> None:
    cp = SqliteCheckpointer(":memory:")
    assert cp.enqueue(["a", "b", "c"]) == 3
    assert cp.enqueue(["a", "b", "c"]) == 0  # all already there
    assert cp.enqueue(["a", "d"]) == 1


def test_mark_success_persists_result() -> None:
    cp = SqliteCheckpointer(":memory:")
    cp.enqueue(["a"])
    cp.mark_success("a", {"answer": 42})
    st = cp.get("a")
    assert st.status == PromptStatus.SUCCESS
    assert st.result == {"answer": 42}


def test_mark_dlq_persists_error() -> None:
    cp = SqliteCheckpointer(":memory:")
    cp.enqueue(["a"])
    cp.mark_dlq("a", "test failure")
    st = cp.get("a")
    assert st.status == PromptStatus.DLQ
    assert st.error == "test failure"


def test_increment_attempt() -> None:
    cp = SqliteCheckpointer(":memory:")
    cp.enqueue(["a"])
    cp.increment_attempt("a", error="timeout")
    cp.increment_attempt("a", error="timeout")
    st = cp.get("a")
    assert st.attempts == 2
    assert st.error == "timeout"
    assert st.status == PromptStatus.PENDING


def test_counts_by_status() -> None:
    cp = SqliteCheckpointer(":memory:")
    cp.enqueue([f"p{i}" for i in range(5)])
    cp.mark_success("p0", {"r": 1})
    cp.mark_success("p1", {"r": 1})
    cp.mark_dlq("p2", "bad")
    counts = cp.counts()
    assert counts.get("success") == 2
    assert counts.get("dlq") == 1
    assert counts.get("pending") == 2


def test_all_pending_ids_excludes_terminal() -> None:
    cp = SqliteCheckpointer(":memory:")
    cp.enqueue(["a", "b", "c"])
    cp.mark_success("a", {})
    cp.mark_dlq("c", "x")
    assert cp.all_pending_ids() == ["b"]


def test_reset_in_progress_returns_to_pending() -> None:
    cp = SqliteCheckpointer(":memory:")
    cp.enqueue(["a", "b"])
    cp.mark_in_progress("a")
    cp.mark_in_progress("b")
    assert cp.reset_in_progress() == 2
    assert sorted(cp.all_pending_ids()) == ["a", "b"]


def test_persists_across_handles() -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)
    try:
        cp1 = SqliteCheckpointer(path)
        cp1.enqueue(["a", "b"])
        cp1.mark_success("a", {"ok": True})
        cp1.close()

        cp2 = SqliteCheckpointer(path)
        assert cp2.get("a").status == PromptStatus.SUCCESS
        assert cp2.get("a").result == {"ok": True}
        assert cp2.all_pending_ids() == ["b"]
        cp2.close()
    finally:
        if path.exists():
            path.unlink()
