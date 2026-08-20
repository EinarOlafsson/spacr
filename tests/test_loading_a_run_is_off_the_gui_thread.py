"""Loading a run must not stop the window. Instruction 159.

REPORTED 2026-08-18: "i tried to load another run and this seemed to hang
spacr."

`RegressionResultsPanel.load` walked the folder, read the CSV and rebuilt every
view inline, and the file contained no JobRunner at all -- the same defect that
froze the merge before 154 A moved it onto a worker.
"""

import os

import spacr


import pandas as pd
import pytest


def _a_run(tmp_path, name="ols_1", rows=40):
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "feature": [f"g{i}" for i in range(rows)],
        "coefficient": [0.1 * i for i in range(rows)],
        "p_value": [0.01] * rows,
    }).to_csv(folder / "results.csv", index=False)
    return str(folder)


def test_the_panel_has_a_worker_at_all(qtbot):
    """It had none. This is the whole of the finding."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    assert hasattr(panel, "_load_jobs")
    assert hasattr(panel, "start_load")


def test_the_read_half_touches_no_widget(tmp_path):
    """`_read_run` runs on a worker thread, so it must be safe there.

    Called as a plain function with no panel at all -- if it ever reaches a
    widget this fails rather than crashing intermittently inside Qt.
    """
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    out = RegressionResultsPanel._read_run(_a_run(tmp_path))
    assert out.get("error") is None, out
    assert len(out["frame"]) == 40
    assert out["found"].endswith("results.csv")


def test_a_missing_folder_comes_back_as_a_message_not_an_exception(tmp_path):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    out = RegressionResultsPanel._read_run(str(tmp_path / "nope"))
    assert "does not exist" in out["error"]


def test_the_async_load_puts_the_run_on_screen(qtbot, tmp_path):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    seen = []
    panel.load_finished.connect(seen.append)

    assert panel.start_load(_a_run(tmp_path)) is True
    qtbot.waitUntil(lambda: bool(seen), timeout=10000)
    assert seen == [True], seen
    assert not panel.is_loading()


def test_a_failure_still_reports_finished(qtbot, tmp_path):
    """A spinner nothing clears is worse than no spinner."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    seen = []
    panel.load_finished.connect(seen.append)

    assert panel.start_load(str(tmp_path / "missing")) is True
    qtbot.waitUntil(lambda: bool(seen), timeout=10000)
    assert seen == [False]
    assert not panel.is_loading()


def test_a_second_load_is_refused_while_one_runs(qtbot, tmp_path):
    """Two reads of different folders answer out of order."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel._loading = True
    assert panel.start_load(_a_run(tmp_path)) is False


def test_both_load_paths_end_in_one_place(tmp_path):
    """`load` and `start_load` must not drift.

    They did before: the merge was moved onto a worker and the run loader was
    not, because nobody had made them one path.
    """
    import inspect
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    for name in ("load", "_finish_load"):
        src = inspect.getsource(getattr(RegressionResultsPanel, name))
        assert "_apply_loaded_run" in src, (
            f"{name} does not go through the shared ending")
