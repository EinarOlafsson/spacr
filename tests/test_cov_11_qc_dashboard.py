"""The QC dashboard re-reads whenever it cannot prove nothing changed.

The screen caches a whole project's verdicts behind a fingerprint of the
files they came from. Any part of that fingerprint being unavailable has to
mean "read again", never "the cache is still good" -- a dashboard showing
last week's verdicts for a plate that has since been re-masked is worse than
one that is merely slow.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.qc_dashboard import QCDashboardScreen


def _screen(qtbot, reader=None):
    screen = QCDashboardScreen(threaded=False, reader=reader)
    qtbot.addWidget(screen)
    return screen


def test_the_text_export_says_so_when_nothing_has_been_read(qtbot):
    """Copying from an empty screen gives a sentence, not an empty string."""
    screen = _screen(qtbot)

    assert screen.as_text() == "No folder read yet."
