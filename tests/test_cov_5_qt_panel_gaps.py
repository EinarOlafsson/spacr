"""Cancelling, superseding and handing over — the panel seams nobody watches.

A live preview is a pass the user can outrun. Each of these is what happens
when they do: a crop pass abandoned mid-flight, a tracking result that lands
after its question changed, a fit offered a frame that is no longer there.
None may leave a stale picture on screen, and none may raise out of a slot.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.widgets import measure_preview as MP                 # noqa: E402
from spacr.qt.widgets import measurement_scan_panel as msp         # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Measure preview: the crop pass runs on the shared runner, not on a QThread
# ---------------------------------------------------------------------------

@pytest.fixture()
def measure_panel(qtbot):
    panel = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    return panel


def test_a_panel_with_no_runner_reports_no_extra_work(measure_panel):
    """The hook is asked before every cancel, including on a bare panel."""
    assert measure_panel._extra_work_in_flight() is False
    measure_panel._jobs = None
    assert measure_panel._extra_work_in_flight() is False
    measure_panel._cancel_extra_work()   # must not raise without a runner


# ---------------------------------------------------------------------------
# Column regression: the merged frame handed over without a second read
# ---------------------------------------------------------------------------

@pytest.fixture()
def column_panel(qtbot):
    def make(provider):
        panel = msp.ColumnRegressionPanel(frame_provider=provider,
                                          threaded=False)
        qtbot.addWidget(panel)
        return panel
    return make


def test_a_provider_with_an_empty_frame_makes_no_offer(column_panel):
    """An empty frame is not the merge; the fits must read the file instead."""
    panel = column_panel(lambda: pd.DataFrame())

    assert panel._offer_frame("/plate/merged.csv") is False


def test_a_provider_with_no_frame_at_all_makes_no_offer(column_panel):
    """A run whose artefact was written in an earlier session has none."""
    panel = column_panel(lambda: None)

    assert panel._offer_frame("/plate/merged.csv") is False


def test_a_provider_that_raises_makes_no_offer(column_panel):
    """The offer is an optimisation; failing it must not fail the queue."""
    def refuse():
        raise RuntimeError("the merging panel is gone")

    panel = column_panel(refuse)

    assert panel._offer_frame("/plate/merged.csv") is False


def test_a_live_merged_frame_is_offered_under_the_path_the_fits_read(
        column_panel):
    """The fits are pointed at a path; the offer has to use that same name."""
    from spacr import frame_handoff

    frame = pd.DataFrame({"gene": ["a"], "score": [1.0]})
    panel = column_panel(lambda: frame)

    try:
        assert panel._offer_frame("/plate/merged.csv") is True
        assert frame_handoff.held("/plate/merged.csv") is frame
    finally:
        frame_handoff.release("/plate/merged.csv")
    assert panel._offer_frame("") is False, "no path is no offer"


def test_a_panel_with_no_provider_makes_no_offer(column_panel):
    panel = column_panel(None)

    assert panel._offer_frame("/plate/merged.csv") is False
