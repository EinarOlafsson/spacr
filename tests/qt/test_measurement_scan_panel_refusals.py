"""The measurements tab: which columns can be regressed, and every refusal.

:mod:`spacr.qt.widgets.measurement_scan_panel` is a four-step workflow
(attach, choose, merge, regress) where each step can be pressed before the one
before it is done.  These tests walk the column chooser over a table of frames
and drive the merge panel, the regression queue and the tab itself into each
refusal, asserting the exact sentence or return value the panel gives instead
of a traceback.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.qt.widgets import measurement_scan_panel as msp          # noqa: E402

pytestmark = pytest.mark.qt


def _database(directory, plate):
    """One plate's ``measurements.db`` with a cell and a pathogen table."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(str(directory), "measurements.db")
    identity = {"rowID": "r1", "columnID": "c1", "fieldID": "f1"}
    cell = pd.DataFrame({
        "plateID": [plate] * 3, **{k: [v] * 3 for k, v in identity.items()},
        "object_label": [1, 2, 3],
        "area": [100.0, 200.0, 300.0],
        "perimeter": [10.0, 20.0, 30.0],
    })
    pathogen = pd.DataFrame({
        "plateID": [plate] * 4, **{k: [v] * 4 for k, v in identity.items()},
        "cell_id": [1, 1, 2, 2],
        "object_label": [1, 2, 1, 2],
        "pathogen_area": [10.0, 30.0, 50.0, 70.0],
    })
    with sqlite3.connect(path) as database:
        cell.to_sql("cell", database, index=False)
        pathogen.to_sql("pathogen", database, index=False)
    return path


def _rows(paths):
    """Input-table rows, in the shape the paired input table emits."""
    plates = [f"plate{i + 1}" for i in range(len(paths))]
    return [{"plate": plates[i], "score": f"{plates[i]}_scores.csv",
             "count": f"{plates[i]}_counts.csv", "database": path}
            for i, path in enumerate(paths)]


@pytest.fixture()
def two_plates(tmp_path):
    """Two plates' measurement databases."""
    return [_database(tmp_path / "plate1", "plate1"),
            _database(tmp_path / "plate2", "plate2")]


@pytest.fixture()
def frame():
    """A small merged frame with one numeric measurement."""
    return pd.DataFrame({"plateID": ["p1", "p1", "p2"],
                         "rowID": ["r1", "r2", "r1"],
                         "columnID": ["c1", "c1", "c1"],
                         "area": [100.0, 200.0, 300.0]})


def _boom(message, kind=RuntimeError):
    """A provider that raises ``kind(message)``."""
    def provider(*_args, **_kwargs):
        raise kind(message)
    return provider


def test_only_varying_numeric_measurements_are_regressable(subtests):
    """``regressable_columns`` keeps numeric, varying, non-identity columns."""
    cases = [
        ("no frame", None, ()),
        ("two measurements", pd.DataFrame({"cell_area": [1.0, 2.0, 3.0],
                                           "cell_perimeter": [4.0, 5.0, 6.0]}),
         ("cell_area", "cell_perimeter")),
        ("a constant column", pd.DataFrame({"cell_area": [1.0, 1.0, 1.0],
                                            "cell_perimeter": [4.0, 5.0, 6.0]}),
         ("cell_perimeter",)),
        ("identity columns", pd.DataFrame({
            "plateID": ["p1", "p2", "p3"], "object_label": ["1", "2", "3"],
            "source_database": ["a", "b", "c"], "cell_area": [1.0, 2.0, 3.0]}),
         ("cell_area",)),
        ("list values", pd.DataFrame({"cell_area": [1.0, 2.0, 3.0],
                                      "odd": [[1], [2], [3]]}), ("cell_area",)),
        ("set values", pd.DataFrame({"cell_area": [1.0, 2.0, 3.0],
                                     "odd": [{1}, {2}, {3}]}), ("cell_area",)),
        ("a boolean column", pd.DataFrame({"flag": [True, False, True],
                                           "cell_area": [1.0, 2.0, 3.0]}),
         ("cell_area",)),
    ]
    for case, data, expected in cases:
        with subtests.test(msg=case):
            assert msp.regressable_columns(data) == expected


def test_the_regression_queue_says_why_it_cannot_start(qtbot, frame, tmp_path,
                                                       monkeypatch, subtests):
    """``ColumnRegressionPanel``: unreadable frames and settings, missing
    score files, a second start and a refusing job runner."""
    score = tmp_path / "merged.csv"
    frame.to_csv(score, index=False)

    def panel(**kwargs):
        widget = msp.ColumnRegressionPanel(threaded=False, fit=lambda s: {},
                                           **kwargs)
        qtbot.addWidget(widget)
        return widget

    with subtests.test(msg="a frame provider that raises"):
        widget = panel(frame_provider=_boom("the merged frame was deleted",
                                            OSError))
        assert widget.refresh() == 0
        assert widget.columns_list.count() == 0
        assert widget.state.text() == (
            "Could not read the merged frame: the merged frame was deleted")
        assert not widget.run_button.isEnabled()

    with subtests.test(msg="identity columns and a constant only"):
        identity_only = pd.DataFrame({
            "plateID": ["p1", "p2"], "rowID": ["r1", "r1"],
            "columnID": ["c1", "c1"], "constant": [1.0, 1.0]})
        widget = panel(frame_provider=lambda: identity_only)
        assert widget.refresh() == 0
        assert widget.state.text().startswith(
            "The merged frame has 4 columns and none of them is a numeric "
            "measurement that varies.")

    for case, provider in [("no score provider", None),
                           ("a score provider that raises",
                            _boom("no merge has finished"))]:
        with subtests.test(msg=case):
            kwargs = {} if provider is None else {"score_provider": provider}
            widget = panel(frame_provider=lambda: frame, **kwargs)
            assert widget.refresh() == 1
            assert widget._score_path() == ""
            assert not widget.run_button.isEnabled()

    with subtests.test(msg="a queue already running"):
        widget = panel(frame_provider=lambda: frame,
                       score_provider=lambda: str(score))
        widget.refresh()
        assert widget.set_selected_columns(["area"]) == 1
        widget._running = True
        assert widget.start_regressions() is False

    with subtests.test(msg="settings that cannot be read"):
        widget = panel(frame_provider=lambda: frame,
                       settings_provider=_boom("the model field is empty",
                                               ValueError),
                       score_provider=lambda: str(score))
        widget.refresh()
        assert widget.set_selected_columns(["area"]) == 1
        assert widget.start_regressions() is False
        assert widget.progress.text() == (
            "Could not read the run settings: the model field is empty")
        assert widget._running is False

    with subtests.test(msg="a job runner that refuses"):
        widget = panel()
        monkeypatch.setattr(widget._jobs, "submit", lambda *a, **k: False)
        monkeypatch.setattr(widget, "selected_columns", lambda: ("cell_area",))
        monkeypatch.setattr(widget, "_score_path", lambda: "/tmp/merged.csv")
        assert widget.start_regressions() is False
        assert widget._running is False

    for case, provider in [("no provider", None),
                           ("a provider that raises",
                            _boom("the merging panel is gone")),
                           ("an empty frame", lambda: pd.DataFrame()),
                           ("no frame at all", lambda: None)]:
        with subtests.test(msg=f"no offer: {case}"):
            widget = panel(frame_provider=provider)
            assert widget._offer_frame("/plate/merged.csv") is False


def test_the_merge_panel_reports_failures_under_its_plan(qtbot, two_plates,
                                                         tmp_path, monkeypatch,
                                                         subtests):
    """``DatabaseMergePanel``: a failed worker, an unwritable merged frame and
    a refusing job runner each leave the panel idle with a sentence."""
    with subtests.test(msg="a merge worker that fails"):
        widget = msp.DatabaseMergePanel(lambda: _rows(two_plates),
                                        threaded=False)
        qtbot.addWidget(widget)
        widget.start_merge()
        plan = widget._plan_shown
        assert "Anchor: cell" in plan
        widget._merging = True
        widget._on_job_failed("MemoryError: 226,467 rows")
        assert widget.report.toPlainText() == (
            plan + "\n\nThe merge did not finish: MemoryError: 226,467 rows")
        assert widget.merge_button.isEnabled()

    with subtests.test(msg="a merged frame that cannot be written"):
        monkeypatch.setattr(msp, "write_merged_frame",
                            _boom("read-only filesystem", OSError))
        widget = msp.DatabaseMergePanel(
            lambda: _rows(two_plates), threaded=False,
            destination_provider=lambda: str(tmp_path / "out"))
        qtbot.addWidget(widget)
        assert widget.start_merge() is True
        assert len(widget.frame) == 6
        assert widget.merged_frame_path() == ""
        assert ("The merged frame could not be written: read-only filesystem"
                in widget.report.toPlainText())
        monkeypatch.undo()

    with subtests.test(msg="a job runner that refuses"):
        widget = msp.DatabaseMergePanel()
        qtbot.addWidget(widget)
        monkeypatch.setattr(widget._jobs, "submit", lambda *a, **k: False)
        monkeypatch.setattr(widget, "_prepare_merge", lambda **k: object(),
                            raising=False)
        assert widget.start_merge() is False
        assert widget._merging is False


def test_the_scan_tab_layout_and_overlap_line_never_raise(
        qtbot, two_plates, monkeypatch, subtests):
    """``MeasurementScanPanel``: stored layouts that cannot be read or
    written, no sections to share height between, and ``what_is_available``."""
    from spacr.qt import preferences

    def tab(frame_provider=None):
        kwargs = {"database_provider": lambda: _rows(two_plates),
                  "threaded": False}
        if frame_provider is not None:
            kwargs["frame_provider"] = frame_provider
        widget = msp.MeasurementScanPanel(**kwargs)
        qtbot.addWidget(widget)
        return widget

    scan = tab()
    sections = scan.sections()
    assert len(sections) >= 1

    with subtests.test(msg="a stored layout that cannot be read"):
        monkeypatch.setattr(preferences, "get_section_layout",
                            _boom("the stored layout is not JSON", ValueError))
        assert scan.restore_section_layout() is False
        assert scan.sections() == sections

    with subtests.test(msg="a layout that cannot be stored"):
        monkeypatch.setattr(preferences, "set_section_layout",
                            _boom("preferences are read-only", OSError))
        scan.remember_section_layout()
        assert scan.sections() == sections

    with subtests.test(msg="no sections to share the height between"):
        monkeypatch.setattr(type(scan), "sections", lambda self: ())
        scan._share_the_height()
        assert scan.sections() == ()
        monkeypatch.undo()

    run_frame = pd.DataFrame({"plateID": ["plate9"], "rowID": ["r9"],
                              "columnID": ["c9"], "gene": ["a"]})
    with subtests.test(msg="both halves present"):
        widget = tab(lambda: run_frame)
        assert widget.what_is_available() == ""
        widget.databases.start_merge()
        said = widget.what_is_available()
        assert said == msp.describe_key_overlap(
            "merged measurements", widget.databases.frame, "loaded run",
            run_frame)
        assert "merged measurements" in said and "loaded run" in said

    for case, provider in [("the run cannot be read",
                            _boom("the run folder went away")),
                           ("the run has no rows", lambda: pd.DataFrame())]:
        with subtests.test(msg=case):
            widget = tab(provider)
            widget.databases.start_merge()
            assert len(widget.databases.frame) == 6
            assert widget.what_is_available() == ""
