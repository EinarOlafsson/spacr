"""A motility result from a pass that has been superseded is not adopted.

Changing the source while a read is in flight starts a new generation. The
old worker cannot be interrupted, so its answer still arrives -- carrying
the token of the pass that asked for it -- and adopting it would put a
table from the previous file behind the current settings.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.motility_preview import MotilityPreviewPanel  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qapp):
    widget = MotilityPreviewPanel(threaded=False)
    yield widget
    widget.deleteLater()


def _points():
    return pd.DataFrame({
        "plateID": ["plate1", "plate1"],
        "wellID": ["A1", "A1"],
        "fieldID": ["f1", "f1"],
        "cellID": [1, 1],
        "frame": [0, 1],
        "x": [0.0, 1.0], "y": [0.0, 1.0],
    })


def test_a_result_from_the_previous_generation_is_ignored(panel, caplog):
    panel._pending_token = 1
    panel._run_token = 2
    before = panel._status.text()

    with caplog.at_level(logging.DEBUG, "spacr"):
        panel._on_worker_done(_points(), "")

    assert panel._points is None, (
        "the superseded table must not become the panel's cache")
    assert panel._status.text() == before, (
        "and it must not report itself as the current read")


def test_a_result_from_the_current_generation_is_acted_on(panel):
    """The same call, one token apart, is the whole difference."""
    panel._pending_token = 3
    panel._run_token = 3

    panel._on_worker_done(_points().iloc[0:0], "")

    assert "No objects found" in panel._status.text(), (
        "the current pass reports its own outcome")
