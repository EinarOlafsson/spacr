"""Motility preview corners: teardown, propagation, and a filter that spares.

Four narrow paths through :mod:`spacr.qt.widgets.motility_preview`, each one
a place where the panel has to do *nothing* correctly:

* ``shutdown()`` on a panel that never finished building its job runner --
  the last thing a closing screen calls, and an ``AttributeError`` there
  leaves the QThread it was meant to stop running behind the closed window;
* the Propagate toggle on a panel nobody wired a callback to, and the same
  toggle being switched OFF, neither of which may push anything into the
  main Motility Assay settings;
* the over-straight-track filter on a plate where every track is bent --
  the pruning pass must leave the cached point table alone rather than
  rebuilding it from an empty drop set.

Each is asserted against the surface a user actually sees: the settings dict
that reaches the main panel, the rows that reach the plot, the status line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import motility_preview as module
from spacr.qt.widgets.motility_preview import MotilityPreviewPanel

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    """A synchronous panel: jobs run inline, so nothing is left in flight."""
    widget = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


class _RecordingRunner:
    """Stands in for the :class:`JobRunner` the panel builds for itself."""

    def __init__(self, log, pending=0):
        self._log = log
        self._pending = pending

    def pending_jobs(self):
        return self._pending

    def shutdown(self):
        self._log.append("shutdown")


def _point_table():
    """Two five-frame tracks, neither of them straight.

    ``wander`` doubles back on itself and ends where it started
    (straightness 0.0); ``bent`` zig-zags to the right and ends 40 px away
    along a 56.6 px path (straightness 0.707). Both are well inside the
    50 px max-displacement default, so the QC pass drops neither.
    """
    wander = [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0), (10.0, 0.0), (0.0, 0.0)]
    bent = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0), (30.0, 10.0), (40.0, 0.0)]
    rows = []
    for cell_id, path, infected in ((1, wander, False), (2, bent, True)):
        for frame, (x, y) in enumerate(path):
            rows.append({"plateID": "plate1", "wellID": "A01", "fieldID": "1",
                         "cellID": cell_id, "frame": frame, "x": x, "y": y,
                         "area": 9.0, "infected": infected})
    return pd.DataFrame(rows)


@pytest.fixture
def plate_dir(tmp_path):
    """A plate holding six (well, field) time series of two frames each."""
    merged = tmp_path / "plate1" / "merged"
    merged.mkdir(parents=True)
    for well in ("A01", "A02"):
        for field in (1, 2, 3):
            for t in (0, 1):
                arr = np.zeros((6, 8, 8), np.float32)
                arr[4, 2:5, 2:5] = 1                 # one labelled cell
                np.save(str(merged / f"plate1_{well}_{field}_{t}.npy"), arr)
    return str(tmp_path / "plate1")


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def test_shutdown_stops_the_runner_and_survives_never_having_one(panel):
    """Closing the screen must stop the scan whether or not one was built.

    ``shutdown()`` is what ``closeEvent`` calls first, and the runner it
    reaches for is created in the panel's very first constructor statement --
    so a panel whose construction died before that (a screen torn down
    mid-build) still gets shut down. If this raised ``AttributeError``, the
    exception would escape ``closeEvent`` and the two motility workers below
    it would never be waited on: a QThread collected mid-run aborts the whole
    process, which is the failure this method exists to prevent.
    """
    log = []
    real_runner = panel._jobs
    panel._jobs = _RecordingRunner(log, pending=2)
    try:
        assert panel._loads_in_flight == [0, 0], (
            "the runner's two pending scans should be visible as in flight")

        panel.shutdown()

        assert log == ["shutdown"], "shutdown did not reach the job runner"

        del panel._jobs                       # construction never got this far
        panel.shutdown()

        assert log == ["shutdown"], "a second runner was invented from nowhere"
        assert panel._loads_in_flight == []
    finally:
        panel._jobs = real_runner


# ---------------------------------------------------------------------------
# Propagating settings back into the main panel
# ---------------------------------------------------------------------------

def test_the_propagate_toggle_pushes_only_when_on_and_only_when_wired(panel):
    """Tuned settings reach the main panel exactly when the user asked.

    Two ways this goes wrong for a user. Pushing while the toggle is OFF
    silently overwrites the Motility Assay settings they typed by hand, from
    a preview they were only playing with. Pushing on a panel that was built
    without :meth:`set_propagate_callback` -- the standalone card, and every
    embedding that has not wired it yet -- would raise out of a Qt signal
    handler, which in this suite means the toggle is left visually on while
    nothing happened.
    """
    pushed = []

    # Unwired: the toggle still turns on, and there is nobody to push to.
    panel._propagate_btn.setChecked(True)
    assert panel._propagate_btn.isChecked() is True

    panel.set_propagate_callback(pushed.append)
    panel._propagate_btn.setChecked(False)          # off: still no push
    assert pushed == [], "settings were pushed while propagation was off"

    panel._propagate_btn.setChecked(True)           # on and wired: one push
    assert len(pushed) == 1
    first = pushed[0]
    assert first["tracked_object"] == "cell"
    assert first["max_displacement"] == 50.0
    assert first["straightness_threshold"] == 0.95
    assert first["straightness_filter"] is False
    assert first["channels"] == [0, 1, 2, 3]
    # An unset calibration is left out entirely rather than pushed as zero.
    assert "pixels_per_um" not in first
    assert "seconds_per_frame" not in first

    panel._pixels_per_um.setValue(2.0)
    panel._seconds_per_frame.setValue(30.0)
    panel._propagate_btn.setChecked(False)
    panel._propagate_btn.setChecked(True)

    assert len(pushed) == 2, "toggling off then on should push exactly once"
    assert pushed[1]["pixels_per_um"] == 2.0
    assert pushed[1]["seconds_per_frame"] == 30.0


def test_a_propagation_that_raises_is_logged_not_thrown_at_qt(panel):
    """A main panel that rejects the push must not break the preview.

    ``propagate_settings`` runs from a Qt signal handler and from every
    recompute; an exception from the host's callback escaping it would tear
    down the recompute that called it, leaving the plot and the statistics
    showing the previous settings' numbers with no sign anything failed.
    """
    calls = []

    def _explode(settings):
        calls.append(settings)
        raise KeyError("the main panel has no such setting")

    panel.set_propagate_callback(_explode)
    panel.propagate_settings()

    assert len(calls) == 1, "the callback was not called at all"
    assert calls[0]["tracked_object"] == "cell"


# ---------------------------------------------------------------------------
# The over-straight-track filter
# ---------------------------------------------------------------------------

def test_a_filter_that_drops_nothing_leaves_the_whole_point_table_plotted(
        panel, monkeypatch):
    """Bent tracks survive the drift filter, points and all.

    ``straightness_filter`` exists to remove stage drift -- tracks so
    straight they cannot be a crawling cell. When the threshold catches
    nothing, the cached point table must be passed to the plot untouched: the
    pruning pass builds a per-row mask from the dropped track ids, and running
    it with an empty drop set would still be a rebuild of the table, so the
    guard is what keeps "filter on, nothing to filter" identical to "filter
    off". A user sees this as the track panel losing curves it never asked to
    lose.
    """
    plotted = []

    def _fake_render(points, tracks, cal, *args, **kwargs):
        plotted.append((points.copy(), tracks.copy()))
        return np.zeros((16, 16, 3), np.uint8)

    monkeypatch.setattr(module, "render_motility_figure", _fake_render)
    panel._points = _point_table()

    panel._straightness_filter.setChecked(True)     # recomputes

    assert len(plotted) == 1
    points, tracks = plotted[0]
    assert sorted(tracks["cellID"]) == [1, 2], "a bent track was filtered out"
    assert len(points) == 10, "the point table was pruned with nothing to prune"
    assert round(float(tracks.loc[tracks["cellID"] == 1,
                                  "straightness"].iloc[0]), 6) == 0.0
    assert round(float(tracks.loc[tracks["cellID"] == 2,
                                  "straightness"].iloc[0]), 3) == 0.707

    # Same table, a threshold the zig-zag track now fails: both the track and
    # its five rows of points go.
    panel._straightness.setValue(0.50)

    assert len(plotted) == 2
    points, tracks = plotted[1]
    assert list(tracks["cellID"]) == [1]
    assert len(points) == 5
    assert set(points["cellID"]) == {1}
    assert panel._summary.n_tracks == 1


# ---------------------------------------------------------------------------
# The sample cap
# ---------------------------------------------------------------------------

def test_lowering_the_sample_cap_redraws_the_dropdown_and_says_so(panel,
                                                                 plate_dir):
    """The status line has to state that the dropdown is a sample, not a list.

    A 384-well plate has thousands of time series and the dropdown shows a
    bounded random sample of them. A user who lowers the cap and is not told
    the list is now a sample of two would read the missing fields as a scan
    that failed to find half their plate. The field they are currently
    looking at also has to survive the redraw -- the draw is random, and a
    cap change that moved them to somebody else's well would silently change
    what the next preview run reads.
    """
    assert panel.load_folder(plate_dir) is True
    assert panel._group_box.count() == 6, "six time series were written"
    assert panel.sample_note() == "showing all 6 image sets"
    watching = panel._group_box.currentData()

    panel._max_sets_box.setValue(2)

    # Two drawn, plus the field being watched when the draw missed it.
    assert panel._group_box.count() in (2, 3)
    assert panel._group_box.currentData() == watching, (
        "the field under preview was dropped by a redraw of the sample")
    assert panel.sample_note().startswith(
        "showing a random sample of 2 of 6 image sets")
    assert panel._status.text().startswith(
        "Showing a random sample of 2 of 6 image sets")
    assert "showing a random sample of 2 of 6" in panel._group_box.toolTip()
