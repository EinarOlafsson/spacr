"""An asynchronous drag must finish before another press can replace it."""

import threading

import numpy as np
import pytest

from tests.qt.test_the_magnifier_drag_merges_what_the_cursor_passes_over import (
    CodedStub,
    drag_along,
    fields,  # noqa: F401 - fixture imported for screen
    press,
    pull,
    release,
    screen,  # noqa: F401 - shared real canvas fixture
    settle_on,
    switch_on,
    wait_for_edits,
)


@pytest.mark.parametrize("finish_before_release", [False, True])
def test_a_second_press_cannot_replace_a_drag_waiting_for_its_last_frame(
    qtbot, screen, finish_before_release,  # noqa: F811 - shared pytest fixture
):
    stub = CodedStub({1: (20, 28, 35, 36), 2: (46, 28, 54, 36)})
    switch_on(screen, stub, save="touching")
    settle_on(qtbot, screen, 22, 32)
    stub.gate = threading.Event()
    try:
        drag_along(screen, list(range(22, 31)), 32)
        pending = screen._magnifier._stroke
        assert pending is not None and pending.released and pending.waiting()

        press(screen, 50, 32)
        assert screen._magnifier._stroke is pending
        pull(screen, 52, 32)
        assert screen._magnifier._stroke is pending
        if finish_before_release:
            stub.gate.set()
            wait_for_edits(qtbot, screen)
            assert screen._magnifier._stroke is None
        release(screen, 50, 32)
        if not finish_before_release:
            assert screen._magnifier._stroke is pending
    finally:
        stub.gate.set()
        qtbot.waitUntil(screen._magnifier._worker.idle, timeout=10_000)

    wait_for_edits(qtbot, screen)
    assert screen._log.counts().get("magnifier") == 1
    assert screen._canvas.mask[32, 22] > 0
    assert screen._canvas.mask[32, 50] == 0

    # Positive counterpart: once the pending edit is committed, a fresh click
    # at the very same position must work, rather than remaining disabled.
    settle_on(qtbot, screen, 50, 32)
    press(screen, 50, 32)
    release(screen, 50, 32)
    wait_for_edits(qtbot, screen, 2)
    assert screen._canvas.mask[32, 50] > 0
    assert len(np.unique(screen._canvas.mask)) == 3
    screen._on_undo()
    assert screen._canvas.mask[32, 22] > 0
    assert screen._canvas.mask[32, 50] == 0
