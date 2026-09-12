"""A stall watch that loses the last stall loses the only one that matters.

`spacr/qt/stall_watch.py` samples the GUI thread while it is stuck and then
counts the innermost frames, because ONE SNAPSHOT NAMES WHERE THE THREAD WAS,
NOT WHERE THE TIME WENT -- and those differ whenever the stall is a loop
rather than a single blocking call. The distribution is the useful half.

IT WAS WRITTEN OUT ONLY WHEN THE NEXT STALL BEGAN. So a process that wedges
once and is then killed -- a run against a sleeping autofs mount, which is
the failure this tool exists for -- left a traceback and NO distribution at
all, because there was never a next stall to trigger the write.
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QTimer                            # noqa: E402
from PySide6.QtWidgets import QApplication                   # noqa: E402


def _watch_one_stall(tmp_path, monkeypatch, seconds=3.0):
    """Hold the GUI thread once, then let the loop answer again."""
    from spacr.qt import stall_watch

    log = tmp_path / "stalls.log"
    monkeypatch.setattr(stall_watch, "LOG_PATH", log)
    # A quarter of the real poll, so one short stall still yields several
    # samples without making the test slow.
    monkeypatch.setattr(stall_watch, "POLL_SECONDS", 0.05)

    app = QApplication.instance()
    # echo=False: the watcher writes from a DAEMON THREAD, and writing
    # into a stream pytest is swapping underneath it crashes pytest
    # inside its own capture.py. The file is the record either way.
    stall_watch.watch_this_application(app, stall_seconds=0.4,
                                       echo=False)

    def wedge():
        """Block the GUI thread the way a synchronous call does."""
        time.sleep(seconds * 0.4)

    QTimer.singleShot(50, wedge)
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
        if log.exists() and "WHERE THAT STALL SPENT" in log.read_text():
            break
    return log.read_text() if log.exists() else ""


@pytest.mark.slow
def test_a_single_stall_is_summarised_without_waiting_for_a_second(
        tmp_path, monkeypatch, qapp):
    """THE LAST STALL OF A SESSION IS THE ONE ANYBODY RUNS THIS FOR.

    A process that freezes and is killed has exactly one stall. If the
    summary waits for the next one it is never written, and what survives is
    a single snapshot -- which names a frame that may merely have been
    executing when the sampler fired.
    """
    text = _watch_one_stall(tmp_path, monkeypatch)

    assert "GUI THREAD STALLED" in text, (
        "the watch did not report the stall at all, so this test is not "
        "measuring what it claims")
    assert "WHERE THAT STALL SPENT ITS TIME" in text, (
        "one stall happened and no distribution was written: the summary is "
        "still waiting for a SECOND stall that a wedged process never has.\n"
        + text[-400:])


@pytest.mark.slow
def test_the_summary_counts_more_than_one_sample(tmp_path, monkeypatch, qapp):
    """A distribution of one sample is a snapshot wearing a percentage.

    The whole reason for sampling through the stall is that a single frame
    is not evidence about where the time went. A summary that reports 1/1
    would satisfy the test above while restoring the defect it was written
    for.
    """
    text = _watch_one_stall(tmp_path, monkeypatch)
    marker = "WHERE THAT STALL SPENT ITS TIME"
    assert marker in text, text[-400:]
    header = text[text.index(marker):text.index(marker) + 120]
    assert "samples at" in header, header
    count = int(header.split("(")[1].split(" samples")[0])
    assert count >= 2, (
        f"the stall was summarised from {count} sample(s), which is a "
        "snapshot with a percent sign on it rather than a distribution")
