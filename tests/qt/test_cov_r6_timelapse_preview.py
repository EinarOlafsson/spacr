"""Timelapse preview: the cache paths the memory budget actually walks.

Pins what ``FrameSequence`` and ``TimelapsePreviewPanel`` do when the
process-wide cache-budget sweep meets a cache that is *busy* or a token that
is *stale* — a frame two workers decoded at the same instant, a lock a
decoding worker is holding while the GUI thread sweeps, a frame that was
never decoded, and the two mask stacks (the one on screen and the one a
worker is re-linking) that refuse to be evicted through the owner protocol.
"""
from __future__ import annotations

import threading
import types

import numpy as np
import pytest

from spacr.qt.widgets import timelapse_preview as TP


H = W = 8


def _frame_files(tmp_path, n: int = 2):
    """``n`` single-frame ``.npy`` files, each filled with its own index."""
    paths = []
    for index in range(n):
        path = tmp_path / f"frame_{index}.npy"
        np.save(path, np.full((H, W), index + 1, dtype=np.uint16))
        paths.append(path)
    return paths


def _sequence(tmp_path, n: int = 2):
    return TP.FrameSequence("files", _frame_files(tmp_path, n), n,
                            list(range(n)))


# ---------------------------------------------------------------------------
# FrameSequence: the LRU order under concurrent decodes
# ---------------------------------------------------------------------------

def test_two_workers_decoding_one_frame_leave_a_single_lru_entry(tmp_path):
    """The cache lock is dropped while a frame decodes, so two movie-field
    workers can both miss on the same frame and both come back to insert it.
    The second insert must move the existing order entry, not duplicate it —
    a duplicate would make the LRU evict a frame that is still cached.
    """
    sequence = _sequence(tmp_path, 1)
    both_decoding = threading.Barrier(2, timeout=10)
    real_read = sequence._read

    def read_in_lockstep(real):
        # Both threads are past the cache miss before either one inserts.
        both_decoding.wait()
        return real_read(real)

    sequence._read = read_in_lockstep
    decoded = []
    threads = [threading.Thread(target=lambda: decoded.append(sequence.frame(0)))
               for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert sequence.read_count == 2, "only one thread ever reached the decode"
    assert sequence._cache_order == [0], "the LRU gained a duplicate entry"
    assert list(sequence._cache) == [0]
    assert len(decoded) == 2
    assert all(int(frame[0, 0]) == 1 for frame in decoded)


# ---------------------------------------------------------------------------
# FrameSequence: the budget protocol against a busy cache
# ---------------------------------------------------------------------------

def test_a_sweep_reports_nothing_and_evicts_nothing_from_a_busy_sequence(
        tmp_path):
    """The sweep runs on the GUI thread; a worker may hold the cache lock.

    Both halves of the owner protocol take the lock without blocking, so a
    busy sequence is skipped for this pass rather than freezing the GUI.
    """
    sequence = _sequence(tmp_path, 1)
    assert int(sequence.frame(0)[0, 0]) == 1

    free_rows = sequence._cache_budget_entries()
    assert [row[0] for row in free_rows] == [0]
    assert free_rows[0][1] == sequence._cache[0].nbytes

    holding = threading.Event()
    finish = threading.Event()

    def hold_the_lock():
        with sequence._cache_lock:
            holding.set()
            finish.wait(timeout=10)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    try:
        assert holding.wait(timeout=10)
        assert sequence._cache_budget_entries() == []
        assert sequence._drop_cache_budget_entry(0) is False
        assert 0 in sequence._cache, "a busy sequence was evicted anyway"
    finally:
        finish.set()
        holder.join(timeout=10)

    assert [row[0] for row in sequence._cache_budget_entries()] == [0]
    assert sequence._drop_cache_budget_entry(0) is True
    assert sequence._cache == {}


def test_dropping_a_frame_that_was_never_decoded_drops_nothing(tmp_path):
    """The policy holds tokens from an earlier pass, so it can name a frame
    this sequence has since re-read or never cached at all."""
    sequence = _sequence(tmp_path, 2)
    assert int(sequence.frame(0)[0, 0]) == 1

    assert sequence._drop_cache_budget_entry(1) is False
    assert sequence._cache_order == [0], "the untouched frame disturbed the LRU"
    assert 0 in sequence._cache

    assert sequence._drop_cache_budget_entry(0) is True
    assert sequence._cache_order == []
    assert sequence._cache_last_used == {}
    assert int(sequence.frame(0)[0, 0]) == 1, "the frame did not come back"


# ---------------------------------------------------------------------------
# TimelapsePreviewPanel: the two pinned mask stacks
# ---------------------------------------------------------------------------

def test_the_shown_stack_and_the_one_a_worker_wants_refuse_eviction(qtbot):
    """``_cache_budget_entries`` pins the drawn stack and the worker's, and
    ``_drop_cache_budget_entry`` refuses them a second time.

    The sweep filters pinned rows out before it evicts, so the guards inside
    the eviction callback are only reachable by calling it — which is exactly
    what a policy token held across a pass does once the panel has moved on.
    """
    panel = TP.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    shown = np.zeros((2, H, W), dtype=np.int32)
    pending = np.ones((2, H, W), dtype=np.int32)
    cold = np.full((2, H, W), 2, dtype=np.int32)
    panel._mask_cache.update({("shown",): shown, ("pending",): pending,
                              ("cold",): cold})
    panel._masks = shown
    panel._pending_signature = ("pending",)
    # Only ``is not None`` is ever asked of it here; ``closeEvent`` waits on
    # it, so the stand-in answers that too.
    panel._worker = types.SimpleNamespace(wait=lambda _ms: None)
    try:
        pinned = {row[0]: row[3] for row in panel._cache_budget_entries()}
        assert pinned == {("shown",): True, ("pending",): True,
                          ("cold",): False}

        assert panel._drop_cache_budget_entry(("shown",)) is False
        assert panel._drop_cache_budget_entry(("pending",)) is False
        assert panel._drop_cache_budget_entry(("never-cached",)) is False
        assert panel._drop_cache_budget_entry(("cold",)) is True

        assert set(panel._mask_cache) == {("shown",), ("pending",)}
        assert panel._mask_cache[("shown",)] is shown
    finally:
        panel._worker = None

    # With the worker gone its signature is no longer pinned, and the same
    # token that was refused above now evicts.
    pinned = {row[0]: row[3] for row in panel._cache_budget_entries()}
    assert pinned == {("shown",): True, ("pending",): False}
    assert panel._drop_cache_budget_entry(("pending",)) is True
    assert set(panel._mask_cache) == {("shown",)}
    panel.shutdown()
