"""Cancelling, superseding and handing over — the panel seams nobody watches.

A live preview is a pass the user can outrun. Each of these is what happens
when they do: a crop pass abandoned mid-flight, a tracking result that lands
after its question changed, a fit offered a frame that is no longer there.
None may leave a stale picture on screen, and none may raise out of a slot.
"""
from __future__ import annotations

import os
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.widgets import measure_preview as MP                 # noqa: E402
from spacr.qt.widgets import measurement_scan_panel as msp         # noqa: E402
from spacr.qt.widgets import timelapse_preview as TP               # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Measure preview: the crop pass runs on the shared runner, not on a QThread
# ---------------------------------------------------------------------------

def _merged(tmp_path, name="plate1_A01_f1.npy"):
    data = np.zeros((48, 48, 8), np.float32)
    data[..., :3] = 20
    cell = np.zeros((48, 48), np.int32)
    cell[2:18, 2:18] = 1
    cell[24:42, 24:42] = 2
    data[..., 4] = cell
    path = tmp_path / name
    np.save(path, data)
    return str(path)


@pytest.fixture()
def measure_panel(qtbot):
    panel = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    return panel


def test_running_the_preview_with_no_array_says_so_instead_of_nothing(
        measure_panel):
    """Every live view answers to ``run_preview``; a silent one looks broken."""
    measure_panel.run_preview()

    assert measure_panel.preview_status().startswith(
        MP.MeasurePreviewPanel.PREVIEW_SOURCE_HINT)
    assert measure_panel.can_preview() is False


def test_running_the_preview_on_a_loaded_array_re_crops_it(measure_panel,
                                                            tmp_path):
    """``run_preview`` is the shared name for what the crop knobs also do."""
    measure_panel.load_array(_merged(tmp_path))
    assert measure_panel.can_preview() is True
    landed = []
    measure_panel.preview_ready.connect(landed.append)

    measure_panel.run_preview()

    assert landed, "the pass produced no result"
    assert measure_panel._crops, "the array holds two objects to crop"


def test_cancelling_drops_the_answer_of_the_pass_in_flight(measure_panel,
                                                            tmp_path):
    """A crop pass cannot be killed, so its result has to land as a no-op."""
    measure_panel.load_array(_merged(tmp_path))
    measure_panel.run_preview()
    before = list(measure_panel._crops)
    token = measure_panel._crop_token

    measure_panel.cancel_preview()

    assert measure_panel._crop_token != token
    # The superseded result arrives late and must change nothing.
    measure_panel._on_crops_ready(token, {"crops": [], "error": ""})
    assert measure_panel._crops == before


def test_a_panel_with_no_runner_reports_no_extra_work(measure_panel):
    """The hook is asked before every cancel, including on a bare panel."""
    assert measure_panel._extra_work_in_flight() is False
    measure_panel._jobs = None
    assert measure_panel._extra_work_in_flight() is False
    measure_panel._cancel_extra_work()   # must not raise without a runner


# ---------------------------------------------------------------------------
# Timelapse preview: a result whose question has changed
# ---------------------------------------------------------------------------

@pytest.fixture()
def timelapse_panel(qtbot):
    panel = TP.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    return panel


class _StubWorker:
    """Stands in for the QThread the panel is waiting on."""

    def isRunning(self):
        return True

    def wait(self, *_args):
        return True


def test_a_superseded_tracking_result_is_dropped_not_drawn(timelapse_panel):
    """Re-linking twice must not repaint the first answer over the second.

    The pass in flight cannot be stopped, so the panel bumps its token and
    the old answer has to recognise itself as stale and leave the screen
    alone — including the panel's own record of the worker it is waiting on.
    """
    timelapse_panel.set_preview_status("linking…")
    timelapse_panel._pending_token = 1
    timelapse_panel._run_token = 2          # a newer pass was started
    sentinel = _StubWorker()
    timelapse_panel._worker = sentinel

    timelapse_panel._on_worker_done(pd.DataFrame({"x": [1]}), "")

    assert timelapse_panel._worker is sentinel, "the live pass was forgotten"
    assert "linking" in timelapse_panel.preview_status()
    timelapse_panel._worker = None


def test_the_current_tracking_result_is_adopted(timelapse_panel):
    """The control: a result whose token is current does reach the panel."""
    timelapse_panel._pending_token = 0
    timelapse_panel._run_token = 0
    timelapse_panel._worker = None

    timelapse_panel._on_worker_done(None, "cellpose would not load")

    assert "cellpose would not load" in timelapse_panel.preview_status()


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
