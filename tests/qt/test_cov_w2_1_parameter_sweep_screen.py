"""The Parameter Sweep workbench: seeding it, starting it, and reading a row.

Nothing here runs a real sweep -- that is a few hundred regressions -- but
everything up to and after the engine call is driven for real: the axes the
form builds, the settings it seeds from the regression panel beside it, the
status lines it writes, and what double-clicking a row does when the trial
failed, when its results are on disk, and when they are not.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest
from PySide6.QtWidgets import QMessageBox, QTableWidgetItem

from spacr.qt.screens.parameter_sweep import (
    APP_KEY, _make_screen, build_parameter_sweep_card, sweepable,
)


class _Inline:
    """A JobRunner stand-in that runs the job where the test can see it."""

    def __init__(self):
        self.jobs = []

    def submit(self, job, handler):
        self.jobs.append(job)
        handler(job())


@pytest.fixture
def screen(qtbot):
    widget = _make_screen()
    qtbot.addWidget(widget)
    widget._runner = _Inline()
    return widget


@pytest.fixture
def answered_warnings(monkeypatch):
    """Record the modal warnings instead of opening one nobody can dismiss."""
    seen = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args, **kwargs: seen.append(args)))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *args, **kwargs: seen.append(args)))
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


# ---------------------------------------------------------------------------
# The space the form builds


def test_an_unticked_axis_is_pinned_rather_than_dropped(screen):
    """The settings a sweep ran under are always fully recorded."""
    include, editor = screen._axis_rows["regression_type"]
    include.setChecked(False)

    space = screen.space()

    assert "regression_type" not in space.axes
    assert space.fixed["regression_type"] == "ols"


def test_a_ticked_axis_with_one_value_is_also_pinned(screen):
    """One value is not a range, whatever the tick box says."""
    include, editor = screen._axis_rows["regression_type"]
    include.setChecked(True)
    editor.setText("ols")

    space = screen.space()

    assert "regression_type" not in space.axes
    assert space.fixed["regression_type"] == "ols"


def test_an_empty_axis_contributes_nothing_at_all(screen):
    """Clearing a row is how a user says "leave this to the defaults"."""
    include, editor = screen._axis_rows["regression_type"]
    editor.setText("   ")

    space = screen.space()

    assert "regression_type" not in space.axes
    assert "regression_type" not in space.fixed


def test_blank_entries_between_commas_are_skipped(screen):
    """`ols,,rlm` is two values, not three with an empty one."""
    include, editor = screen._axis_rows["regression_type"]
    include.setChecked(True)
    editor.setText("ols, , rlm")

    assert screen.space().axes["regression_type"] == ["ols", "rlm"]


def test_values_that_are_not_literals_stay_strings(screen):
    """`fdr_bh` is a name; `100` is a number. Both have to survive."""
    include, editor = screen._axis_rows["min_cell_count"]
    include.setChecked(True)
    editor.setText("100, 250")
    method_include, method_editor = screen._axis_rows[
        "multiple_testing_method"]
    method_include.setChecked(True)
    method_editor.setText("fdr_bh, bonferroni")

    space = screen.space()

    assert space.axes["min_cell_count"] == [100, 250]
    assert space.axes["multiple_testing_method"] == ["fdr_bh", "bonferroni"]


# ---------------------------------------------------------------------------
# Seeding from the regression panel beside it


def test_settings_that_are_not_a_dict_are_ignored(screen):
    """The host may hand this whatever it has, including nothing."""
    before = screen.base_settings()

    screen.apply_settings(None)

    assert screen.base_settings() == before


def test_the_inputs_already_on_screen_come_straight_across(screen, tmp_path):
    """Opening the sweep should not mean retyping them."""
    screen.apply_settings({
        "score_data": [str(tmp_path / "scores.csv")],
        "count_data": [str(tmp_path / "counts.csv")],
        "dependent_variable": "fraction",
        "src": str(tmp_path)})

    base = screen.base_settings()
    assert base["score_data"] == [str(tmp_path / "scores.csv")]
    assert base["count_data"] == [str(tmp_path / "counts.csv")]
    assert base["dependent_variable"] == "fraction"
    assert screen.destination.text() == os.path.join(str(tmp_path), "sweep")


def test_an_existing_destination_is_not_overwritten(screen, tmp_path):
    """A folder the user typed is a decision, not a placeholder."""
    screen.destination.setText("/somewhere/chosen")

    screen.apply_settings({"src": str(tmp_path)})

    assert screen.destination.text() == "/somewhere/chosen"


def test_a_file_list_that_refuses_a_value_does_not_break_the_seed(screen,
                                                                  tmp_path):
    """One widget rejecting a path must not cost the rest of the seeding."""
    def _refuse(value):
        raise ValueError("not a list of paths I recognise")

    screen.score_data.set_value = _refuse

    screen.apply_settings({"score_data": ["a.csv"],
                           "dependent_variable": "fraction"})

    assert screen.dependent_variable.text() == "fraction"


def test_an_unticked_axis_takes_the_users_current_value(screen):
    """An unticked axis reproduces their run rather than a default."""
    include, editor = screen._axis_rows["regression_type"]
    include.setChecked(False)

    screen.apply_settings({"regression_type": "glm"})

    assert editor.text() == "glm"
    assert screen.space().fixed["regression_type"] == "glm"


def test_a_ticked_axis_gains_the_users_value_as_one_of_its_trials(screen):
    """Their own condition has to be one of the trials that gets run."""
    include, editor = screen._axis_rows["regression_type"]
    include.setChecked(True)
    editor.setText("ols, rlm")

    screen.apply_settings({"regression_type": "quantile"})

    assert editor.text().startswith("quantile, ")
    assert "quantile" in screen.space().axes["regression_type"]


def test_a_value_already_in_the_range_is_not_added_twice(screen):
    """Two identical trials is one measurement, twice charged."""
    include, editor = screen._axis_rows["regression_type"]
    include.setChecked(True)
    editor.setText("ols, rlm")

    screen.apply_settings({"regression_type": "ols"})

    assert editor.text() == "ols, rlm"


def test_a_none_valued_setting_is_written_as_none(screen):
    """`transform: None` is a real choice and has to survive the round trip."""
    include, editor = screen._axis_rows["transform"]
    include.setChecked(False)

    screen.apply_settings({"transform": "None"})

    assert editor.text() == "None"


# ---------------------------------------------------------------------------
# The worker note


def test_the_note_says_so_when_memory_could_not_be_measured(screen):
    """A worker count with no stated basis reads as arbitrary."""
    screen._set_worker_note({"available": None, "workers": 2})

    assert "could not be measured" in screen.worker_note.text()
    assert "2" in screen.worker_note.text()


def test_the_note_names_the_requested_count_when_it_was_cut(screen):
    """A user who asked for 32 and got 4 needs to know why."""
    screen._set_worker_note({"available": 12.0, "per_trial": 1.5,
                             "budget_fraction": 0.6, "workers": 4,
                             "requested": 32})

    text = screen.worker_note.text()
    assert "32" in text and "4" in text


def test_the_note_states_the_calculation_when_nothing_was_cut(screen):
    """The same three numbers, without the "requested" clause."""
    screen._set_worker_note({"available": 64.0, "per_trial": 1.5,
                             "budget_fraction": 0.6, "workers": 8,
                             "requested": 8})

    text = screen.worker_note.text()
    assert "GiB available" in text
    assert "requested" not in text.lower()


# ---------------------------------------------------------------------------
# Starting a sweep


def test_a_sweep_with_no_inputs_is_refused(screen, answered_warnings):
    """There is nothing to fit, and no folder would be written either."""
    screen.start()

    assert answered_warnings
    assert "Nothing to sweep" in answered_warnings[0][1]
    assert screen._runner.jobs == []


def test_a_sweep_with_no_output_folder_is_refused(screen, answered_warnings,
                                                  tmp_path):
    """A sweep writes thousands of folders; it needs to be told where."""
    screen.apply_settings({"score_data": [str(tmp_path / "s.csv")],
                           "count_data": [str(tmp_path / "c.csv")]})
    screen.destination.setText("  ")

    screen.start()

    assert "No output folder" in answered_warnings[0][1]
    assert screen._runner.jobs == []


def test_a_started_sweep_hands_the_engine_what_the_form_says(screen,
                                                             monkeypatch,
                                                             tmp_path):
    """The screen is a workbench over a headless engine, and passes it through."""
    import spacr.parameter_sweep as engine

    seen = {}

    def _record(base, destination, space, **kwargs):
        seen.update(kwargs)
        seen["base"], seen["destination"], seen["space"] = (
            base, destination, space)
        return _results_frame(rows=3)

    monkeypatch.setattr(engine, "run_sweep_parallel", _record)
    screen.apply_settings({"score_data": [str(tmp_path / "s.csv")],
                           "count_data": [str(tmp_path / "c.csv")],
                           "src": str(tmp_path)})
    screen.workers.setValue(2)
    screen.seed.setValue(7)

    screen.start()

    assert seen["destination"] == os.path.join(str(tmp_path), "sweep")
    assert seen["n_jobs"] == 2
    assert seen["seed"] == 7
    assert seen["base"]["score_data"] == [str(tmp_path / "s.csv")]
    assert "3 trials, 3 succeeded" in screen.status.text()
    assert screen.start_button.isEnabled()
    assert not screen.progress.isVisible()
    assert screen.table.rowCount() == 3


def test_a_sweep_that_produced_nothing_says_that(screen):
    """An empty table is not a finished sweep with no hits."""
    screen._sweep_finished(None)

    assert screen.status.text() == "The sweep produced no trials."
    assert screen.table.rowCount() == 0


# ---------------------------------------------------------------------------
# Watching a running sweep from its table


def test_a_missing_results_table_says_where_it_looked(screen, tmp_path):
    """A sweep still starting up has not written one yet."""
    screen.destination.setText(str(tmp_path))

    screen.load_results()

    assert "No results table at" in screen.status.text()
    assert str(tmp_path) in screen.status.text()


def test_the_table_on_disk_is_read_and_shown(screen, tmp_path):
    """This is how a running sweep is watched, so it reads a partial file."""
    _results_frame(rows=4).to_csv(tmp_path / "sweep_results.csv", index=False)
    screen.destination.setText(str(tmp_path))

    screen.load_results()

    assert screen.status.text() == "4 trials recorded so far."
    assert screen.table.rowCount() == 4
    assert screen.table.horizontalHeaderItem(0).text() == "trial_id"


# ---------------------------------------------------------------------------
# Opening one row


def test_activating_a_row_before_any_results_does_nothing(screen):
    """The signal fires on an empty table too."""
    screen._on_row_activated()

    assert screen._runner.jobs == []
    assert screen.trial_status.text() == ""


def test_activating_with_no_row_selected_does_nothing(screen, tmp_path):
    """`currentRow` is -1 until something is clicked."""
    _results_frame().to_csv(tmp_path / "sweep_results.csv", index=False)
    screen.destination.setText(str(tmp_path))
    screen.load_results()
    screen.table.setCurrentCell(-1, -1)

    screen._on_row_activated()

    assert screen._runner.jobs == []


def test_a_failed_trial_says_why_instead_of_refitting_it(screen, tmp_path,
                                                         answered_warnings):
    """Re-fitting a trial that raised would raise again, a minute later."""
    _results_frame(rows=1, status="failed").to_csv(
        tmp_path / "sweep_results.csv", index=False)
    screen.destination.setText(str(tmp_path))
    screen.load_results()
    screen.table.setCurrentCell(0, 0)

    screen._on_row_activated()

    assert answered_warnings
    assert "That trial failed" in answered_warnings[0][1]
    assert "no rows survived filtration" in answered_warnings[0][2]
    assert screen._runner.jobs == []


def test_a_trial_whose_results_are_on_disk_is_not_refitted(screen, tmp_path,
                                                           monkeypatch):
    """A saved run is instant; a re-fit is a minute for an identical answer."""
    _results_frame(rows=1, folder=str(tmp_path / "trial_0")).to_csv(
        tmp_path / "sweep_results.csv", index=False)
    screen.destination.setText(str(tmp_path))
    screen.load_results()
    screen.table.setCurrentCell(0, 0)
    monkeypatch.setattr(type(screen.results), "load",
                        lambda self, folder: True)

    screen._on_row_activated()

    assert "loaded from disk" in screen.trial_status.text()
    assert "Nothing was re-fitted" in screen.trial_status.text()
    assert screen._runner.jobs == []


def test_a_trial_with_no_saved_results_is_refitted_and_drawn(screen, tmp_path,
                                                             monkeypatch):
    """What comes back is that trial, not a fresh one from the controls."""
    from matplotlib.figure import Figure

    import spacr.parameter_sweep as engine

    _results_frame(rows=1).to_csv(tmp_path / "sweep_results.csv", index=False)
    screen.destination.setText(str(tmp_path))
    screen.load_results()
    screen.table.setCurrentCell(0, 0)
    monkeypatch.setattr(type(screen.results), "load",
                        lambda self, folder: False)
    monkeypatch.setattr(engine, "rerun_trial", lambda base, record: {
        "figures": [Figure()],
        "settings": {"regression_type": "ols", "analysis_unit": "well"},
        "output": {"results": pd.DataFrame({"gene": ["a"], "coef": [1.0]})}})

    screen._on_row_activated()

    assert len(screen._runner.jobs) == 1
    assert "1 figure(s) from regression_type='ols'" in screen.trial_status.text()
    assert "analysis_unit='well'" in screen.trial_status.text()


def test_a_row_the_frame_does_not_have_is_ignored(screen, tmp_path):
    """The table can outlive the frame it was drawn from."""
    _results_frame(rows=1).to_csv(tmp_path / "sweep_results.csv", index=False)
    screen.destination.setText(str(tmp_path))
    screen.load_results()
    screen.table.setRowCount(3)
    screen.table.setItem(2, 0, QTableWidgetItem("no-such-trial"))
    screen.table.setCurrentCell(2, 0)

    screen._on_row_activated()

    assert screen._runner.jobs == []


def test_a_trial_id_that_cannot_be_matched_falls_back_to_the_row(screen,
                                                                 tmp_path,
                                                                 monkeypatch):
    """The table may be sorted, so the id is tried first — but not trusted."""
    _results_frame(rows=1).to_csv(tmp_path / "sweep_results.csv", index=False)
    screen.destination.setText(str(tmp_path))
    screen.load_results()
    screen.table.setCurrentCell(0, 0)
    monkeypatch.setattr(type(screen.results), "load",
                        lambda self, folder: True)

    class _Hostile(pd.DataFrame):
        """A results frame whose ``trial_id`` column refuses to be read.

        ``_constructor_from_mgr`` is overridden alongside ``_constructor``
        because pandas rebuilds a subclass through it on every internal
        operation. Overriding ``_constructor`` alone sends pandas down
        ``_Hostile(mgr)``, which DeprecationWarns on the pandas floor
        setup.py declares and errors in the job that installs it -- and the
        hostile ``__getitem__`` this class exists for would go missing from
        any frame pandas rebuilt the other way.
        """

        @property
        def _constructor(self):
            return _Hostile

        def _constructor_from_mgr(self, mgr, axes):
            return _Hostile._from_mgr(mgr, axes=axes)

        def __getitem__(self, key):
            if isinstance(key, str) and key == "trial_id":
                raise RuntimeError("this column is not readable")
            return super().__getitem__(key)

    screen._results = _Hostile(screen._results)

    screen._on_row_activated()

    assert "loaded from disk" in screen.trial_status.text()


def test_a_trial_that_did_not_come_back_says_so(screen):
    """The worker returning nothing is a failure, not an empty result."""
    screen._trial_figures_ready(None)

    assert screen.trial_status.text() == (
        "That trial did not come back. See the console.")


def test_a_figure_that_will_not_attach_does_not_lose_the_others(screen,
                                                                monkeypatch):
    """The figure panel is decoration around the numbers."""
    from matplotlib.figure import Figure

    def _refuse(figure):
        raise RuntimeError("no canvas for this one")

    monkeypatch.setattr(screen.figures, "add_figure", _refuse)

    screen._trial_figures_ready({"figures": [Figure()], "settings": {}})

    assert "1 figure(s)" in screen.trial_status.text()


def test_results_that_will_not_load_do_not_lose_the_figures(screen,
                                                            monkeypatch):
    """Same posture on the other side of the panel."""
    from matplotlib.figure import Figure

    def _refuse(frame, source=""):
        raise RuntimeError("that frame is not a results table")

    monkeypatch.setattr(screen.results, "set_frame", _refuse)

    screen._trial_figures_ready({
        "figures": [Figure()],
        "settings": {"regression_type": "ols"},
        "output": {"results": pd.DataFrame({"gene": ["a"]})}})

    assert "1 figure(s) from regression_type='ols'" in screen.trial_status.text()


def test_a_trial_with_no_figures_says_that(screen):
    """Nothing drawn is a result too, and not an empty status line."""
    screen._trial_figures_ready({"figures": [], "settings": {}})

    assert screen.trial_status.text() == "That trial produced no figures."


# ---------------------------------------------------------------------------
# Where the sweep lives


def test_only_the_regression_module_has_a_sweep():
    """The axes and the row round trip are specific to it."""
    assert sweepable("regression") is True
    assert sweepable(APP_KEY) is False
    assert sweepable("umap") is False


def test_the_card_wraps_the_same_panel_the_screen_is(qtbot):
    """One factory, so the card and the tile cannot drift."""
    panel, card = build_parameter_sweep_card(None)
    qtbot.addWidget(card)

    assert panel.parent() is not None
    assert card.body_layout.indexOf(panel) >= 0
    assert hasattr(panel, "space")


def test_an_uncontainable_host_is_still_told_so_without_a_palette(monkeypatch,
                                                                  qtbot):
    """A theme that will not answer costs the colour, not the warning."""
    import spacr.parameter_sweep as engine
    import spacr.qt.theme as theme

    monkeypatch.setattr(engine, "containment_available", lambda: False)

    def _refuse():
        raise RuntimeError("no palette in this build")

    monkeypatch.setattr(theme, "active_palette", _refuse)

    widget = _make_screen()
    qtbot.addWidget(widget)

    assert "Kernel containment is unavailable" in widget.containment.text()
    assert widget.containment.objectName() == "DangerLabel"


def test_the_estimate_says_how_many_trials_and_how_long(screen):
    """The number a user reads before committing a machine to a sweep."""
    for include, _editor in screen._axis_rows.values():
        include.setChecked(False)

    count = screen.estimate()

    assert count >= 0
    assert "1 raw combinations" in screen.status.text()
    assert f"{count} valid trials would run on" in screen.status.text()
    assert "Invalid combinations are excluded before the sweep starts." in \
        screen.status.text()
    assert "GiB available" in screen.worker_note.text() or \
        "could not be measured" in screen.worker_note.text()
