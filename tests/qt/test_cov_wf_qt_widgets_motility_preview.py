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

The later half of the file walks the same corners along their real routes:
``closeEvent`` on a panel with no job runner, a recompute that reaches for a
callback nobody registered, the Propagate toggle going off, and the drift
filter one thousandth below its threshold.

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


# ---------------------------------------------------------------------------
# The close path, end to end
# ---------------------------------------------------------------------------

class _FakeWorker:
    """A stand-in for :class:`_MotilityWorker` that records its ``wait``."""

    def __init__(self, log, name):
        self._log = log
        self._name = name

    def wait(self, msec):
        self._log.append((self._name, msec))
        return True


def test_closing_a_runnerless_panel_still_waits_on_both_workers(panel):
    """The window close must reach the QThread waits even with no job runner.

    ``closeEvent`` cancels the plate scan *first* and only then waits on the
    motility workers, so anything that raises in the cancel step skips both
    waits. A panel whose construction never got as far as its ``JobRunner``
    -- the screen torn down while it was still being built -- is exactly that
    case: the runner lookup has to come back empty rather than raise. If it
    raised, the two workers below would never be waited on, and a ``QThread``
    garbage-collected while still running aborts the whole application
    instead of closing one window.
    """
    from PySide6.QtGui import QCloseEvent

    log = []
    real_runner = panel._jobs
    panel._worker = _FakeWorker(log, "in-flight")
    panel._retired_worker = _FakeWorker(log, "retired")
    try:
        del panel._jobs                  # construction died before line one

        panel.closeEvent(QCloseEvent())

        assert log == [("in-flight", 5000), ("retired", 5000)], (
            "the close path skipped a worker wait after finding no runner")
    finally:
        panel._worker = None
        panel._retired_worker = None
        panel._jobs = real_runner


# ---------------------------------------------------------------------------
# Propagation from the recompute, and from the toggle going off
# ---------------------------------------------------------------------------

def test_a_recompute_with_propagate_armed_but_nothing_wired_still_finishes(
        panel, monkeypatch):
    """An unwired Propagate button must not break the recompute behind it.

    Every recompute ends by pushing settings when the toggle is armed, and
    the standalone Motility preview card is built without a callback -- the
    host wires one only when the panel is embedded next to the real settings
    form. Reaching for a callback that was never registered has to be a quiet
    no-op: if it raised, it would raise *after* the statistics and the plot
    were computed but *before* ``preview_ready`` is emitted, so the user
    would see the numbers update while everything downstream of the signal
    -- the card's own ready state -- stayed on the previous run.
    """
    pushed = []
    monkeypatch.setattr(
        module, "render_motility_figure",
        lambda *a, **k: np.zeros((16, 16, 3), np.uint8))
    ready = []
    panel.preview_ready.connect(ready.append)

    panel._propagate_btn.setChecked(True)       # armed, and nobody is wired
    panel._points = _point_table()
    panel.recompute()

    assert len(ready) == 1, "preview_ready never fired on the unwired panel"
    assert ready[0].n_tracks == 2
    assert "straightness" in panel._stats_label.text()
    assert pushed == [], "an unwired panel pushed settings from nowhere"

    # The same recompute, once a host has registered itself: now it pushes.
    panel.set_propagate_callback(pushed.append)
    panel.recompute()

    assert len(ready) == 2
    assert len(pushed) == 1, "a wired panel did not push from the recompute"
    assert pushed[0]["straightness_threshold"] == 0.95


def test_switching_propagate_off_pushes_nothing_at_all(panel):
    """Turning the toggle OFF must be silent, not a last parting push.

    The toggle is the user's statement about where the authoritative settings
    live. While it is on, the preview owns them; the moment they switch it
    off they are taking the main Motility Assay form back -- usually because
    they are about to type a value the preview cannot express. A push on the
    way out would overwrite that form with the preview's numbers at the exact
    moment the user said to stop, and they would have no way to tell it
    happened short of re-reading every field.
    """
    pushed = []
    panel.set_propagate_callback(pushed.append)

    panel._on_propagate_toggled(False)

    assert pushed == [], "switching propagation off pushed the settings"

    panel._on_propagate_toggled(True)

    assert len(pushed) == 1, "switching propagation on pushed nothing"
    assert pushed[0]["tracked_object"] == "cell"


# ---------------------------------------------------------------------------
# The drift filter right at its threshold
# ---------------------------------------------------------------------------

def _almost_straight_table():
    """One nearly straight track (0.9991) and one zig-zag (0.7071).

    The first crawls 40 px along x and steps 1 px off the line at the end --
    straight enough to be mistaken for stage drift, but not actually
    straight. Both stay well inside the 50 px max-displacement default.
    """
    creeper = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 1.0)]
    bent = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0), (30.0, 10.0), (40.0, 0.0)]
    rows = []
    for cell_id, path in ((1, creeper), (2, bent)):
        for frame, (x, y) in enumerate(path):
            rows.append({"plateID": "plate1", "wellID": "A01", "fieldID": "1",
                         "cellID": cell_id, "frame": frame, "x": x, "y": y,
                         "area": 9.0, "infected": False})
    return pd.DataFrame(rows)


def test_the_drift_filter_spares_a_track_just_under_its_threshold(
        panel, monkeypatch):
    """The threshold is a strict ``<``, and the point table must follow it.

    A cell that crawls almost in a line is still a cell. The filter drops
    tracks whose straightness reaches the threshold, so a track at 0.9991
    survives a threshold of 1.0 -- and when nothing is dropped, the cached
    point table has to be handed to the plot as it stands rather than rebuilt
    through the per-row prune. Getting the boundary wrong is what makes a
    user's real, slightly-directional cells vanish from the track panel the
    moment they turn the drift filter on, with the summary still counting
    them.
    """
    plotted = []

    def _fake_render(points, tracks, cal, *args, **kwargs):
        plotted.append((points.copy(), tracks.copy()))
        return np.zeros((16, 16, 3), np.uint8)

    monkeypatch.setattr(module, "render_motility_figure", _fake_render)
    panel._straightness.setValue(1.0)
    panel._points = _almost_straight_table()

    panel._straightness_filter.setChecked(True)     # recomputes

    assert len(plotted) == 1
    points, tracks = plotted[0]
    assert sorted(tracks["cellID"]) == [1, 2], (
        "a track under the threshold was treated as stage drift")
    straightness = float(tracks.loc[tracks["cellID"] == 1,
                                    "straightness"].iloc[0])
    assert 0.99 < straightness < 1.0, straightness
    assert len(points) == 10, "the point table was pruned with nothing to drop"
    assert panel._summary.n_tracks == 2

    # Move the threshold below it and the same track is now drift: it goes,
    # and so do its five rows of points.
    panel._straightness.setValue(0.90)

    assert len(plotted) == 2
    points, tracks = plotted[1]
    assert list(tracks["cellID"]) == [2]
    assert set(points["cellID"]) == {2}
    assert len(points) == 5
    assert panel._summary.n_tracks == 1
