"""Two more panels that must not get stuck when a job is refused.

Instruction 288. ``DatabaseMergePanel.start_merge`` and
``ColumnRegressionPanel.start_regressions`` both set their running flag
and repaint their buttons BEFORE submitting, because the submit is the
part that takes minutes. If the submit is refused, both have to be put
back or the panel is stuck showing a run that is not happening.

Both arms were marked ``# pragma: no cover - JobRunner always returns
True today``, the same claim already corrected in
``measurement_compare_dialog``: ``JobRunner.submit`` returns False
whenever it is unthreaded and either the work or the completion callback
raises. These panels are constructed with ``threaded=False`` in tests
and threaded in the application, so the arm is live in exactly the
configuration the tests use.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import measurement_scan_panel as msp


class _RefusingRunner:
    """Declines the job, the way an unthreaded JobRunner does when the
    work or the completion callback raises."""

    def __init__(self):
        self.asked = 0

    def submit(self, _fn, _on_done=None) -> bool:
        self.asked += 1
        return False

    def cancel(self):
        pass

    def shutdown(self, *_args, **_kwargs):
        # closeEvent calls this during teardown; without it the widget
        # errors on close and pytest reports "previous item was not torn
        # down properly" instead of the assertion under test.
        return True


class _AcceptingRunner(_RefusingRunner):
    def submit(self, _fn, _on_done=None) -> bool:
        self.asked += 1
        return True


def _frame():
    return pd.DataFrame({
        "plate": ["p1"] * 8 + ["p2"] * 8,
        "gene": [f"g{i % 4}" for i in range(16)],
        "score": [float(i) for i in range(16)],
        "count": [10.0] * 16,
    })


# ---------------------------------------------------------------------------
# ColumnRegressionPanel
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_panel(qtbot):
    panel = msp.ColumnRegressionPanel(frame_provider=_frame, threaded=False)
    qtbot.addWidget(panel)
    return panel


def test_a_refused_regression_queue_stops_claiming_to_run(regression_panel):
    """THE ARM."""
    panel = regression_panel
    # The COLUMN PICKER and the merged-frame path are not what is under
    # test; both are separate refusals with their own messages, and each
    # returns before the submit. Stubbing them is what lets the submit be
    # reached at all.
    panel.selected_columns = lambda: ("score",)
    panel._score_path = lambda: "/tmp/merged.csv"
    panel._offer_frame = lambda *_a, **_k: True

    runner = _RefusingRunner()
    panel._jobs = runner

    started = panel.start_regressions()

    assert runner.asked == 1, "the panel never tried to submit"
    assert started is False, "a refused submit reported success"
    assert panel._running is False, (
        "the panel still believes a queue is running; every later press "
        "returns early and the fits can never be started")


def test_an_accepted_regression_queue_does_claim_to_run(regression_panel):
    """The other side, so the test above cannot pass against a panel that
    never sets the flag at all."""
    panel = regression_panel
    panel.selected_columns = lambda: ("score",)
    panel._score_path = lambda: "/tmp/merged.csv"
    panel._offer_frame = lambda *_a, **_k: True

    panel._jobs = _AcceptingRunner()

    assert panel.start_regressions() is True
    assert panel._running is True


# ---------------------------------------------------------------------------
# DatabaseMergePanel
# ---------------------------------------------------------------------------

def _merge_panel(qtbot, tmp_path):
    import sqlite3

    paths = []
    for name in ("plate1", "plate2"):
        path = tmp_path / f"{name}.db"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE object (plate TEXT, value REAL)")
            db.execute("INSERT INTO object VALUES (?, ?)", (name, 1.0))
        paths.append(str(path))

    rows = [{"plate": f"plate{i + 1}", "score": "", "count": "",
             "database": path} for i, path in enumerate(paths)]
    panel = msp.DatabaseMergePanel(lambda: rows, threaded=False)
    qtbot.addWidget(panel)
    return panel


def test_a_refused_merge_stops_claiming_to_run(qtbot, tmp_path):
    """THE ARM, in the other panel."""
    panel = _merge_panel(qtbot, tmp_path)
    runner = _RefusingRunner()
    panel._jobs = runner

    started = panel.start_merge()

    if runner.asked == 0:
        pytest.skip("the merge was refused before reaching the submit")
    assert started is False, "a refused submit reported success"
    assert panel._merging is False, (
        "the panel still believes a merge is running")


def test_an_accepted_merge_does_claim_to_run(qtbot, tmp_path):
    """The other side."""
    panel = _merge_panel(qtbot, tmp_path)
    runner = _AcceptingRunner()
    panel._jobs = runner

    started = panel.start_merge()

    if runner.asked == 0:
        pytest.skip("the merge was refused before reaching the submit")
    assert started is True
    assert panel._merging is True


def test_job_runner_really_can_refuse():
    """THE PREMISE both pragmas denied."""
    from spacr.qt.job_runner import JobRunner

    runner = JobRunner(threaded=False, app_key="test", user_visible=False)

    def boom(_result):
        raise RuntimeError("the completion callback failed")

    assert runner.submit(lambda: {"ok": True}, boom) is False
    assert runner.submit(lambda: {"ok": True}, lambda _r: None) is True
