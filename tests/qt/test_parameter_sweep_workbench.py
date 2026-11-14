"""The Parameter Sweep workbench: the space its form builds, seeding, starting, and reading rows.

No real sweep runs here.  Everything up to and after the engine call is
driven for real: the axes the form builds, the settings it seeds from the
regression panel beside it, the worker and containment notes, the status
lines it writes, and what activating a row does when the trial failed, when
its results are on disk, and when they are not.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QTableWidgetItem

import spacr.parameter_sweep as engine
import spacr.qt.theme as theme
from spacr.qt.screens.parameter_sweep import _make_screen


class _Inline:
    """A JobRunner stand-in that runs the job where the test can see it."""

    def __init__(self):
        self.jobs = []

    def submit(self, job, handler):
        self.jobs.append(job)
        handler(job())


def _budget(**kwargs):
    """A fixed worker budget, so notes and estimates do not depend on this machine."""
    return {"workers": 3, "available": 20.0, "per_trial": 6.0,
            "budget_fraction": 0.5, "requested": kwargs.get("requested")}


@pytest.fixture
def make(qtbot, monkeypatch):
    """Build a fresh screen with an inline runner and a fixed worker budget."""
    monkeypatch.setattr(engine, "_recommended_worker_budget", _budget)

    def _build():
        widget = _make_screen()
        qtbot.addWidget(widget)
        widget._runner = _Inline()
        return widget

    return _build


@pytest.fixture
def dialogs(monkeypatch):
    """Record the modal warnings and notices instead of opening them."""
    seen = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args, **kwargs: seen.append(
                            ("warning",) + args[1:])))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *args, **kwargs: seen.append(
                            ("information",) + args[1:])))
    return seen


def _results_frame(rows=2, status="ok", folder=""):
    return pd.DataFrame([{
        "trial_id": index,
        "status": status,
        "regression_type": "ols",
        "inference": "parametric",
        "analysis_unit": "well",
        "n_below_alpha": index,
        "positive_rank": index + 1,
        "folder": folder,
        "error_type": "ValueError",
        "error": "no rows survived filtration",
    } for index in range(rows)])


def _loaded(screen, tmp_path, frame):
    frame.to_csv(tmp_path / "sweep_results.csv", index=False)
    screen.destination.setText(str(tmp_path))
    screen.load_results()


def _set_axis(screen, key, text, ticked):
    include, editor = screen._axis_rows[key]
    include.setChecked(ticked)
    editor.setText(text)
    return editor


def test_the_space_pins_unticked_and_single_valued_axes_and_parses_literals(make, subtests):
    """Ticked multi-value rows become axes, others are pinned to their first value, empty rows vanish."""
    absent = object()
    for case_id, key, text, ticked, axis, fixed in [
            ("an unticked axis is pinned", "regression_type", "ols, rlm",
             False, None, "ols"),
            ("one ticked value is pinned", "regression_type", "ols", True,
             None, "ols"),
            ("an empty row contributes nothing", "regression_type", "   ",
             True, None, absent),
            ("blank entries between commas are skipped", "regression_type",
             "ols, , rlm", True, ["ols", "rlm"], absent),
            ("numbers are literals", "min_cells_per_well", "100, 250", True,
             [100, 250], absent),
            ("names stay strings", "multiple_testing_method",
             "fdr_bh, bonferroni", True, ["fdr_bh", "bonferroni"], absent),
            ("None is a literal", "transform", "None, log", False, None,
             None),
            ("quoted text and floats are literals", "alpha", "'x', 1.5",
             True, ["x", 1.5], absent),
            ("a comma inside brackets still splits", "alpha", "[1, 2], auto",
             True, ["[1", "2]", "auto"], absent)]:
        with subtests.test(msg=case_id):
            screen = make()
            _set_axis(screen, key, text, ticked)
            space = screen.space()
            assert space.axes.get(key) == axis
            assert space.fixed.get(key, absent) == fixed

    with subtests.test(msg="the default form: every unticked row is pinned"):
        screen = make()
        space = screen.space()
        assert sorted(space.axes) == [
            "agg_type", "alpha", "analysis_unit", "batch_correction",
            "inference", "multiple_testing_method",
            "random_row_column_effects", "regression_type",
            "threshold_multiplier", "transform"]
        assert space.fixed == {
            "fdr_alpha": 0.05, "threshold_method": "std",
            "fraction_threshold": 0.02, "min_cells_per_well": 100,
            "min_observations_per_hit": 0, "outlier_detection": False}
        assert space.axes["transform"] == [None, "log"]


def test_apply_settings_seeds_inputs_and_pins_the_users_values(make, subtests, tmp_path):
    """Inputs come across, a typed destination is kept, and every axis gains the user's own value once."""
    with subtests.test(msg="a non-dict is ignored"):
        screen = make()
        before = screen.base_settings()
        texts = {key: editor.text()
                 for key, (_inc, editor) in screen._axis_rows.items()}
        screen.apply_settings(None)
        screen.apply_settings(["regression_type", "glm"])
        assert screen.base_settings() == before
        assert {key: editor.text() for key, (_inc, editor)
                in screen._axis_rows.items()} == texts
        assert screen.destination.text() == ""

    with subtests.test(msg="the inputs already on screen come across"):
        screen = make()
        screen.apply_settings({
            "score_data": (str(tmp_path / "scores.csv"),),
            "count_data": [str(tmp_path / "counts.csv")],
            "dependent_variable": "fraction",
            "src": str(tmp_path)})
        assert screen.base_settings() == {
            "score_data": [str(tmp_path / "scores.csv")],
            "count_data": [str(tmp_path / "counts.csv")],
            "dependent_variable": "fraction",
            "annotation_source": "", "verbose": False}
        assert screen.destination.text() == os.path.join(str(tmp_path),
                                                          "sweep")

    with subtests.test(msg="empty values change nothing"):
        screen = make()
        screen.apply_settings({"score_data": [], "dependent_variable": "",
                               "src": ""})
        assert screen.base_settings()["score_data"] == []
        assert screen.dependent_variable.text() == "pred"
        assert screen.destination.text() == ""

    with subtests.test(msg="a typed destination is not overwritten"):
        screen = make()
        screen.destination.setText("/somewhere/chosen")
        screen.apply_settings({"src": str(tmp_path)})
        assert screen.destination.text() == "/somewhere/chosen"

    with subtests.test(msg="a file list that refuses does not stop the rest"):
        screen = make()
        seen = []

        def _refuse(value):
            seen.append(value)
            raise ValueError("not a list of paths I recognise")

        screen.score_data.set_value = _refuse
        screen.apply_settings({"score_data": ("a.csv",),
                               "count_data": [str(tmp_path / "c.csv")],
                               "dependent_variable": "fraction"})
        assert seen == [["a.csv"]]
        assert screen.count_data.get_value() == [str(tmp_path / "c.csv")]
        assert screen.dependent_variable.text() == "fraction"

    for case_id, key, start, ticked, value, expected in [
            ("an unticked axis takes the user's value", "regression_type",
             "ols, rlm", False, "glm", "glm"),
            ("a ticked axis gains the user's value first", "regression_type",
             "ols, rlm", True, "quantile", "quantile, ols, rlm"),
            ("a value already in the range is not added twice",
             "regression_type", "ols, rlm", True, "ols", "ols, rlm"),
            ("a None-valued string is written as None", "transform", "log",
             False, "None", "None"),
            ("a number is written as text", "min_cells_per_well", "100",
             False, 250, "250"),
            ("an actual None is ignored", "transform", "log", False, None,
             "log")]:
        with subtests.test(msg=case_id):
            screen = make()
            editor = _set_axis(screen, key, start, ticked)
            screen.apply_settings({key: value})
            assert editor.text() == expected

    with subtests.test(msg="the seeded values reach the space"):
        screen = make()
        _set_axis(screen, "regression_type", "ols", False)
        _set_axis(screen, "inference", "parametric", True)
        screen.apply_settings({"regression_type": "glm",
                               "inference": "nonparametric"})
        space = screen.space()
        assert space.fixed["regression_type"] == "glm"
        assert space.axes["inference"] == ["nonparametric", "parametric"]


def test_the_worker_note_states_its_basis(make, subtests):
    """The note gives the measured memory calculation, says when it could not measure, and names a cut request."""
    for case_id, budget, expected in [
            ("memory could not be measured",
             {"available": None, "workers": 2},
             "Available memory could not be measured, so the worker count "
             "is limited to 2."),
            ("the requested count was cut",
             {"available": 12.0, "per_trial": 1.5, "budget_fraction": 0.6,
              "workers": 4, "requested": 32},
             "12 GiB available; allowing about 1.5 GiB per trial and 60% of "
             "available memory gives a worker count of 4. The requested "
             "count was 32."),
            ("nothing was cut",
             {"available": 64.0, "per_trial": 1.5, "budget_fraction": 0.6,
              "workers": 8, "requested": 8},
             "64 GiB available; allowing about 1.5 GiB per trial and 60% of "
             "available memory gives a worker count of 8."),
            ("nothing was requested",
             {"available": 64.0, "per_trial": 1.5, "budget_fraction": 0.6,
              "workers": 8, "requested": None},
             "64 GiB available; allowing about 1.5 GiB per trial and 60% of "
             "available memory gives a worker count of 8."),
            ("more was granted than requested",
             {"available": 64.0, "per_trial": 1.5, "budget_fraction": 0.6,
              "workers": 8, "requested": 4},
             "64 GiB available; allowing about 1.5 GiB per trial and 60% of "
             "available memory gives a worker count of 8.")]:
        with subtests.test(msg=case_id):
            screen = make()
            screen._set_worker_note(budget)
            assert screen.worker_note.text() == expected

    with subtests.test(msg="the built screen starts from the budget"):
        screen = make()
        assert screen.workers.value() == 3
        assert screen.worker_note.text() == (
            "20 GiB available; allowing about 6.0 GiB per trial and 50% of "
            "available memory gives a worker count of 3.")


def test_the_estimate_counts_trials_and_runtime_without_running(make, monkeypatch):
    """Estimate states raw and valid combinations, the workers and the minutes, and refreshes the note."""
    seen = {}

    def _trials(space, mode, max_trials, seed):
        seen.update(space=space, mode=mode, max_trials=max_trials, seed=seed)
        return [{}] * 7

    requests = []
    monkeypatch.setattr(engine, "build_trials", _trials)
    monkeypatch.setattr(engine, "_recommended_worker_budget",
                        lambda **kw: requests.append(kw) or _budget(**kw))
    screen = make()
    for include, _editor in screen._axis_rows.values():
        include.setChecked(False)
    _set_axis(screen, "regression_type", "ols, rlm", True)
    screen.mode.setCurrentText("grid")
    screen.max_trials.setValue(40)
    screen.seed.setValue(11)
    screen.workers.setValue(5)

    assert screen.estimate() == 7
    assert seen["mode"] == "grid"
    assert seen["max_trials"] == 40
    assert seen["seed"] == 11
    assert seen["space"].axes == {"regression_type": ["ols", "rlm"]}
    assert requests == [{}, {"requested": 5}]
    assert screen.status.text() == (
        "2 raw combinations; 7 valid trials would run on 3 worker(s) — "
        "roughly 2 minutes if each trial takes about one minute. Invalid "
        "combinations are excluded before the sweep starts.")
    assert screen.worker_note.text() == (
        "20 GiB available; allowing about 6.0 GiB per trial and 50% of "
        "available memory gives a worker count of 3. The requested count "
        "was 5.")


def test_the_real_estimate_of_a_pinned_form(make):
    """With every axis unticked there is one raw combination, counted by the real trial builder."""
    screen = make()
    for include, _editor in screen._axis_rows.values():
        include.setChecked(False)
    space = screen.space()
    expected = len(engine.build_trials(space, mode="random", max_trials=500,
                                       seed=20260815))
    assert screen.estimate() == expected
    assert screen.status.text() == (
        f"1 raw combinations; {expected} valid trials would run on 3 "
        f"worker(s) — roughly {expected * 60 / 3 / 60:.0f} minutes if each "
        f"trial takes about one minute. Invalid combinations are excluded "
        f"before the sweep starts.")


def test_the_containment_row_states_whether_limits_are_active(qtbot, monkeypatch, subtests):
    """Active containment names its limits; unavailable containment warns in the error colour, or without it."""
    monkeypatch.setattr(engine, "_recommended_worker_budget", _budget)
    for case_id, available, palette, sheet in [
            ("active", True, lambda: {"error": "#ff0000"}, ""),
            ("unavailable, coloured", False, lambda: {"error": "#ff0000"},
             "color: #ff0000;"),
            ("unavailable, no palette", False, None, "")]:
        with subtests.test(msg=case_id):
            monkeypatch.setattr(engine, "containment_available",
                                lambda available=available: available)
            if palette is None:
                def palette():
                    raise RuntimeError("no palette in this build")
            monkeypatch.setattr(theme, "active_palette", palette)
            widget = _make_screen()
            qtbot.addWidget(widget)
            text = widget.containment.text()
            if available:
                assert text == (
                    "Kernel containment is active for each trial: memory 24G, "
                    "swap disabled, and CPU quota 400%. If a trial exceeds a "
                    "limit, only that trial is stopped and recorded as "
                    "'killed'; the sweep continues.")
                assert widget.containment.objectName() == ""
            else:
                assert text.startswith(
                    "Kernel containment is unavailable because systemd-run "
                    "--user --scope could not be started.")
                assert text.endswith(
                    "from a systemd user session before using a large search "
                    "space.")
                assert widget.containment.objectName() == "DangerLabel"
            assert widget.containment.styleSheet() == sheet


def test_start_refuses_an_incomplete_design_and_hands_the_engine_the_form(make, dialogs, monkeypatch, tmp_path, subtests):
    """No inputs or no folder is refused with a reason; a full form reaches run_sweep_parallel as typed."""
    for case_id, settings, destination, title, text in [
            ("no inputs at all", {}, None, "Nothing to sweep",
             "Choose at least one score CSV and one count CSV."),
            ("scores but no counts", {"score_data": ["s.csv"]}, None,
             "Nothing to sweep",
             "Choose at least one score CSV and one count CSV."),
            ("no output folder",
             {"score_data": ["s.csv"], "count_data": ["c.csv"]}, "  ",
             "No output folder",
             "Choose a folder for the trial folders and the results table.")]:
        with subtests.test(msg=case_id):
            dialogs.clear()
            screen = make()
            screen.apply_settings(settings)
            if destination is not None:
                screen.destination.setText(destination)
            screen.start()
            assert dialogs == [("warning", title, text)]
            assert screen._runner.jobs == []
            assert screen.start_button.isEnabled()
            assert screen.status.text() == "Nothing running."

    with subtests.test(msg="a started sweep passes the form through"):
        dialogs.clear()
        seen = {}

        def _record(base, destination, space, **kwargs):
            seen.update(kwargs, base=base, destination=destination,
                        space=space)
            assert screen.status.text() == "Sweeping…"
            assert not screen.start_button.isEnabled()
            assert screen.progress.maximum() == 0
            return _results_frame(rows=3, status="ok").assign(
                status=["ok", "failed", "ok"])

        monkeypatch.setattr(engine, "run_sweep_parallel", _record)
        screen = make()
        screen.apply_settings({"score_data": [str(tmp_path / "s.csv")],
                               "count_data": [str(tmp_path / "c.csv")],
                               "src": str(tmp_path)})
        for include, _editor in screen._axis_rows.values():
            include.setChecked(False)
        _set_axis(screen, "inference", "parametric, nonparametric", True)
        screen.workers.setValue(2)
        screen.seed.setValue(7)
        screen.max_trials.setValue(9)
        screen.mode.setCurrentText("grid")
        screen.start()
        destination = os.path.join(str(tmp_path), "sweep")
        assert dialogs == []
        assert len(screen._runner.jobs) == 1
        assert seen["destination"] == destination
        assert seen["base"] == {
            "score_data": [str(tmp_path / "s.csv")],
            "count_data": [str(tmp_path / "c.csv")],
            "dependent_variable": "pred", "annotation_source": "",
            "verbose": False}
        assert seen["space"].axes == {
            "inference": ["parametric", "nonparametric"]}
        assert {key: seen[key] for key in (
            "mode", "max_trials", "seed", "n_jobs", "controls")} == {
            "mode": "grid", "max_trials": 9, "seed": 7, "n_jobs": 2,
            "controls": {"positive": "239740"}}
        assert screen.status.text() == (
            f"3 trials, 2 succeeded. Results in {destination}")
        assert screen.start_button.isEnabled()
        assert not screen.progress.isVisible()
        assert screen.table.rowCount() == 3
        assert screen.table.item(1, 1).text() == "failed"

    for case_id, results in [("None", None),
                             ("an empty frame", _results_frame(rows=0))]:
        with subtests.test(msg=f"a sweep that produced {case_id}"):
            screen = make()
            screen.start_button.setEnabled(False)
            screen.progress.setVisible(True)
            screen._sweep_finished(results)
            assert screen.status.text() == "The sweep produced no trials."
            assert screen.table.rowCount() == 0
            assert screen.start_button.isEnabled()
            assert screen.progress.isHidden()
            assert screen._results is results


def test_load_results_reads_the_table_on_disk_in_its_column_order(make, tmp_path, subtests):
    """A missing table says where it looked; a present one is shown with its preferred columns first."""
    with subtests.test(msg="a missing table names the path"):
        screen = make()
        screen.destination.setText(f"  {tmp_path}  ")
        screen.load_results()
        assert screen.status.text() == (
            f"No results table at {os.path.join(str(tmp_path), 'sweep_results.csv')} yet.")
        assert screen._results is None
        assert screen.table.rowCount() == 0

    with subtests.test(msg="a present table is read and shown"):
        screen = make()
        _loaded(screen, tmp_path, _results_frame(rows=4))
        assert screen.status.text() == "4 trials recorded so far."
        assert screen.table.rowCount() == 4
        headers = [screen.table.horizontalHeaderItem(i).text()
                   for i in range(screen.table.columnCount())]
        assert headers == ["trial_id", "status", "regression_type",
                           "inference", "analysis_unit", "n_below_alpha",
                           "positive_rank", "error_type"]
        assert [screen.table.item(3, c).text() for c in range(8)] == [
            "3", "ok", "ols", "parametric", "well", "3", "4", "ValueError"]

    with subtests.test(msg="a table with none of the preferred columns"):
        screen = make()
        pd.DataFrame({f"c{i}": [i] for i in range(14)}).to_csv(
            tmp_path / "sweep_results.csv", index=False)
        screen.destination.setText(str(tmp_path))
        screen.load_results()
        assert [screen.table.horizontalHeaderItem(i).text()
                for i in range(screen.table.columnCount())] == [
            f"c{i}" for i in range(12)]
        assert screen.table.item(0, 11).text() == "11"

    with subtests.test(msg="fit-support counts keep their own columns"):
        screen = make()
        frame = _results_frame()
        values = {
            "n_rows_fitted_grna": [2955, 2800], "n_wells_grna": [620, 610],
            "n_rows_fitted_gene": [2700, 0], "n_wells_gene": [600, 0],
            "n_rows_fitted": [None, None], "n_wells": [None, None],
            "n_rows_prepared": [3000, 3000], "n_wells_prepared": [625, 625],
        }
        for name, counts in values.items():
            frame[name] = counts
        _loaded(screen, tmp_path, frame)
        headers = [screen.table.horizontalHeaderItem(i).text()
                   for i in range(screen.table.columnCount())]
        assert headers == [
            "trial_id", "status", "regression_type", "inference",
            "analysis_unit", "n_rows_fitted_grna", "n_wells_grna",
            "n_rows_fitted_gene", "n_wells_gene", "n_wells", "n_rows_fitted",
            "n_rows_prepared", "n_wells_prepared", "n_below_alpha",
            "positive_rank", "error_type"]
        for name, counts in values.items():
            column = headers.index(name)
            assert not screen.table.isColumnHidden(column)
            assert [screen.table.item(row, column).text()
                    for row in range(2)] == [
                "nan" if count is None else str(count) for count in counts]


def test_activating_a_row_opens_that_trial_or_explains_why_not(make, dialogs, monkeypatch, tmp_path, subtests):
    """Nothing loaded or selected does nothing; a failed trial says why; saved results load; others are refitted."""
    panel_type = None

    with subtests.test(msg="before any results"):
        screen = make()
        panel_type = type(screen.results)
        screen._on_row_activated()
        assert screen._runner.jobs == []
        assert screen.trial_status.text() == ""

    with subtests.test(msg="no row selected"):
        screen = make()
        _loaded(screen, tmp_path, _results_frame())
        screen.table.setCurrentCell(-1, -1)
        screen._on_row_activated()
        assert screen._runner.jobs == []
        assert screen.trial_status.text() == ""

    with subtests.test(msg="a failed trial says why instead of refitting"):
        dialogs.clear()
        screen = make()
        _loaded(screen, tmp_path, _results_frame(rows=1, status="failed"))
        screen.table.setCurrentCell(0, 0)
        screen._on_row_activated()
        assert dialogs == [(
            "information", "That trial failed",
            "This trial did not produce a regression:\n\n"
            "ValueError: no rows survived filtration")]
        assert screen._runner.jobs == []

    loads = []
    monkeypatch.setattr(panel_type, "load",
                        lambda self, folder: loads.append(folder) or True)
    with subtests.test(msg="results on disk are loaded, not refitted"):
        screen = make()
        folder = str(tmp_path / "trial_0")
        _loaded(screen, tmp_path, _results_frame(rows=2, folder=folder))
        screen.table.setCurrentCell(1, 0)
        screen._on_row_activated()
        assert loads == [folder]
        assert screen.trial_status.text() == (
            f"Trial 1 loaded from disk ({folder}). Nothing was re-fitted.")
        assert screen._runner.jobs == []

    with subtests.test(msg="a row the frame does not have is ignored"):
        loads.clear()
        screen = make()
        _loaded(screen, tmp_path, _results_frame(rows=1))
        screen.table.setRowCount(3)
        screen.table.setItem(2, 0, QTableWidgetItem("no-such-trial"))
        screen.table.setCurrentCell(2, 0)
        screen._on_row_activated()
        assert screen._runner.jobs == []
        assert loads == []
        assert screen.trial_status.text() == ""

    with subtests.test(msg="a sorted table finds the row by its trial id"):
        screen = make()
        _loaded(screen, tmp_path, _results_frame(rows=3, folder="f"))
        screen.table.sortItems(0, Qt.DescendingOrder)
        assert screen.table.item(0, 0).text() == "2"
        screen.table.setCurrentCell(0, 0)
        screen._on_row_activated()
        assert screen.trial_status.text() == (
            "Trial 2 loaded from disk (f). Nothing was re-fitted.")

    monkeypatch.setattr(panel_type, "load", lambda self, folder: False)
    with subtests.test(msg="no saved results: that trial is refitted"):
        reruns = []
        frames = []
        real_set_frame = panel_type.set_frame

        def _set_frame(self, frame, source=""):
            frames.append((list(frame.columns), source))
            return real_set_frame(self, frame, source=source)

        monkeypatch.setattr(panel_type, "set_frame", _set_frame)

        def _rerun(base, record):
            reruns.append((base, record))
            assert screen.trial_status.text() == (
                "Trial 0 has no saved results; re-fitting it to draw them…")
            assert not screen.show_button.isEnabled()
            return {"figures": [Figure()],
                    "settings": {"regression_type": "ols",
                                 "analysis_unit": "well", "src": "/data"},
                    "output": {"results": pd.DataFrame(
                        {"gene": ["a"], "coef": [1.0]})}}

        monkeypatch.setattr(engine, "rerun_trial", _rerun)
        screen = make()
        _loaded(screen, tmp_path, _results_frame(rows=1, folder="gone"))
        screen.table.setCurrentCell(0, 0)
        screen._on_row_activated()
        assert len(screen._runner.jobs) == 1
        assert reruns[0][0] == screen.base_settings()
        assert reruns[0][1]["trial_id"] == 0
        assert reruns[0][1]["regression_type"] == "ols"
        assert frames == [(["gene", "coef"], "/data")]
        assert screen.trial_status.text() == (
            "1 figure(s) from regression_type='ols', analysis_unit='well'. "
            "Right-click a figure to restyle it.")
        assert screen.show_button.isEnabled()


def test_a_returned_trial_is_drawn_whatever_part_of_it_fails(make, monkeypatch, subtests):
    """No payload, a figure or frame that will not attach, and an empty figure list each get their own status."""
    with subtests.test(msg="nothing came back"):
        screen = make()
        screen._trial_figures_ready(None)
        assert screen.trial_status.text() == (
            "That trial did not come back. See the console.")
        assert screen.show_button.isEnabled() is False

    with subtests.test(msg="a figure that will not attach"):
        screen = make()
        added = []

        def _refuse(figure):
            added.append(figure)
            if len(added) == 1:
                raise RuntimeError("no canvas for this one")

        monkeypatch.setattr(screen.figures, "add_figure", _refuse)
        screen._trial_figures_ready({"figures": [Figure(), Figure()],
                                     "settings": {}})
        assert len(added) == 2
        assert screen.figures.isVisibleTo(screen)
        assert screen.trial_status.text() == (
            "2 figure(s) from that trial. Right-click a figure to restyle it.")

    with subtests.test(msg="results that will not load keep the figures"):
        screen = make()

        def _refuse(frame, source=""):
            raise RuntimeError("that frame is not a results table")

        monkeypatch.setattr(screen.results, "set_frame", _refuse)
        screen._trial_figures_ready({
            "figures": [Figure()],
            "settings": {"regression_type": "ols", "inference": "parametric",
                         "multiple_testing_method": "fdr_bh",
                         "min_cells_per_well": 100,
                         "fraction_threshold": 0.02, "seed": 3},
            "output": {"results": pd.DataFrame({"gene": ["a"]})}})
        assert screen.trial_status.text() == (
            "1 figure(s) from regression_type='ols', inference='parametric', "
            "multiple_testing_method='fdr_bh', min_cells_per_well=100, "
            "fraction_threshold=0.02. Right-click a figure to restyle it.")

    with subtests.test(msg="an empty results frame is not shown"):
        screen = make()
        frames = []
        monkeypatch.setattr(screen.results, "set_frame",
                            lambda frame, source="": frames.append(frame))
        screen._trial_figures_ready({"figures": [], "settings": None,
                                     "output": {"results": pd.DataFrame()}})
        assert frames == []
        assert screen.trial_status.text() == "That trial produced no figures."
        assert not screen.figures.isVisibleTo(screen)
