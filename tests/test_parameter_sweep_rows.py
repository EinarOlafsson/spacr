"""Clicking a sweep row gets that regression back, editable.

A sweep is only useful if a promising row can be opened. These cover the
contract that makes that work: a row carries enough to reproduce its own
trial, and what comes back is live figures rather than saved pages.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


class TestARowReproducesItsTrial:

    def test_settings_round_trip_through_the_csv(self, tmp_path):
        """Values come back as strings from disk and must be parsed back.

        A trial run with min_cell_count=100 must not be reproduced with the
        STRING "100": spaCR compares it numerically and would filter nothing.
        """
        from spacr.parameter_sweep import settings_for_trial

        row = {
            "trial_id": 3, "status": "ok", "seconds": 12.0,
            "folder": str(tmp_path), "regression_type": "ols",
            "multiple_testing_method": "fdr_bh", "min_cell_count": "100",
            "transform": "None", "random_row_column_effects": "False",
            "fdr_alpha": "0.05", "fraction_threshold": "0.02",
        }
        settings = settings_for_trial(
            {"score_data": ["a.csv"], "dependent_variable": "pred"}, row)

        assert settings["min_cell_count"] == 100
        assert settings["fdr_alpha"] == 0.05
        assert settings["transform"] is None
        assert settings["random_row_column_effects"] is False
        assert settings["regression_type"] == "ols"
        # A string that was always a string survives.
        assert settings["multiple_testing_method"] == "fdr_bh"

    def test_bookkeeping_is_not_fed_back_as_a_setting(self, tmp_path):
        """A row records what happened as well as what was asked for."""
        from spacr.parameter_sweep import settings_for_trial

        row = {"trial_id": 3, "status": "ok", "seconds": 12.0,
               "n_below_alpha": 7, "error": "", "preparation_key": "abc",
               "folder": str(tmp_path), "regression_type": "ols"}
        settings = settings_for_trial({}, row)

        for key in ("trial_id", "status", "seconds", "n_below_alpha",
                    "error", "preparation_key"):
            assert key not in settings, f"{key} was fed back as a setting"

    def test_building_settings_creates_nothing(self, tmp_path):
        """It builds a dict. A preview must not leave directories behind."""
        from spacr.parameter_sweep import settings_for_trial

        target = tmp_path / "not_yet"
        settings_for_trial({}, {"folder": str(target),
                                "regression_type": "ols"})
        assert not target.exists()

    def test_the_inputs_the_sweep_ran_on_are_preserved(self, tmp_path):
        """Score/count CSVs are not recorded per trial; they come from base."""
        from spacr.parameter_sweep import settings_for_trial

        settings = settings_for_trial(
            {"score_data": ["s1.csv", "s2.csv"], "count_data": ["c1.csv"],
             "dependent_variable": "pred"},
            {"trial_id": 1, "regression_type": "ridge"})
        assert settings["score_data"] == ["s1.csv", "s2.csv"]
        assert settings["count_data"] == ["c1.csv"]
        assert settings["dependent_variable"] == "pred"
        assert settings["regression_type"] == "ridge"


class TestRerunHandsBackLiveFigures:

    def test_only_figures_this_run_made_come_back(self, tmp_path, monkeypatch):
        """A screen with figures already open must not have them stolen.

        Sweeping up every open figure would attribute whatever happened to be
        on screen to a trial that did not produce it.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from spacr import parameter_sweep

        stale = plt.figure()          # already open, nothing to do with us
        stale.gca().plot([0, 1], [0, 1])

        def fake_regression(settings):
            made = plt.figure()
            made.gca().plot([0, 1], [1, 0])
            return {"results": pd.DataFrame({"p_value": [0.01]})}

        import spacr.ml
        monkeypatch.setattr(spacr.ml, "perform_regression", fake_regression)

        payload = parameter_sweep.rerun_trial(
            {}, {"trial_id": 1, "folder": str(tmp_path),
                 "regression_type": "ols"})

        assert len(payload["figures"]) == 1
        assert stale not in payload["figures"]
        plt.close("all")

    def test_what_comes_back_is_a_figure_not_a_path(self, tmp_path, monkeypatch):
        """The point of reopening a condition is to restyle it."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        from spacr import parameter_sweep
        import spacr.ml

        def fake_regression(settings):
            plt.figure().gca().plot([0, 1], [1, 0])
            return {}

        monkeypatch.setattr(spacr.ml, "perform_regression", fake_regression)
        payload = parameter_sweep.rerun_trial(
            {}, {"trial_id": 1, "folder": str(tmp_path)})

        assert all(isinstance(f, Figure) for f in payload["figures"])
        plt.close("all")

    def test_plots_are_turned_on_for_the_rerun(self, tmp_path, monkeypatch):
        """The sweep itself runs quiet; reopening a row is about the pictures."""
        import matplotlib
        matplotlib.use("Agg")

        from spacr import parameter_sweep
        import spacr.ml

        seen = {}

        def fake_regression(settings):
            seen.update(settings)
            return {}

        monkeypatch.setattr(spacr.ml, "perform_regression", fake_regression)
        parameter_sweep.rerun_trial(
            {"verbose": False}, {"trial_id": 1, "folder": str(tmp_path)})
        assert seen.get("verbose") is True


class TestTheDesignSizeIsRecorded:

    def test_a_hit_count_carries_the_design_it_came_from(self):
        """Two trials differing only by a cutoff can fit different data.

        Without the design size, "3 hits vs 10 hits" cannot be told apart
        from "half the data was thrown away".
        """
        from spacr.parameter_sweep import _design_summary

        model_data = pd.DataFrame({
            "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r1_c2"],
            "grna": ["g1", "g2", "g1"],
        })
        summary = _design_summary({"model_data": model_data})
        assert summary["n_wells"] == 2
        assert summary["n_guides"] == 2
        assert summary["n_rows_fitted"] == 3

    def test_it_says_nothing_rather_than_guessing(self):
        from spacr.parameter_sweep import _design_summary

        assert _design_summary({}) == {}
        assert _design_summary(None) == {}


class TestTheScreenWiresItUp:

    @pytest.fixture()
    def screen(self, qtbot):
        from spacr.qt.screens.parameter_sweep import _make_screen
        widget = _make_screen()
        qtbot.addWidget(widget)
        return widget

    def _results(self):
        return pd.DataFrame([
            {"trial_id": 1, "status": "ok", "regression_type": "ols",
             "multiple_testing_method": "fdr_bh", "min_cell_count": 100,
             "n_wells": 606, "n_guides": 789, "n_below_alpha": 10},
            {"trial_id": 2, "status": "failed", "regression_type": "beta",
             "error_type": "LinAlgError", "error": "singular design"},
        ])

    def test_the_table_shows_what_went_in_next_to_what_came_out(self, screen):
        """"all the relevant values for the final regression and the data
        that went into it"."""
        frame = self._results()
        screen._results = frame
        screen._show(frame)
        headers = [screen.table.horizontalHeaderItem(i).text()
                   for i in range(screen.table.columnCount())]
        for expected in ("regression_type", "multiple_testing_method",
                         "min_cell_count", "n_wells", "n_guides",
                         "n_below_alpha"):
            assert expected in headers, f"{expected} is not shown"

    def test_double_clicking_a_row_runs_that_trial(self, screen, monkeypatch):
        """And the trial it runs is the row's, not the controls' current state."""
        frame = self._results()
        screen._results = frame
        screen._show(frame)

        submitted = {}
        monkeypatch.setattr(screen._runner, "submit",
                            lambda job, done: submitted.update(job=job))
        screen.table.selectRow(0)
        screen._on_row_activated()
        assert "job" in submitted, "selecting a row started nothing"

        captured = {}

        def fake_rerun(base, record, **kwargs):
            captured["record"] = record
            return {"settings": {}, "output": {}, "figures": []}

        import spacr.parameter_sweep as sweep
        monkeypatch.setattr(sweep, "rerun_trial", fake_rerun)
        submitted["job"]()
        assert captured["record"]["trial_id"] == 1
        assert captured["record"]["regression_type"] == "ols"

    def test_a_failed_trial_says_why_instead_of_running(self, screen,
                                                        monkeypatch):
        """Re-running a trial that raised would just raise again."""
        frame = self._results()
        screen._results = frame
        screen._show(frame)

        started = []
        monkeypatch.setattr(screen._runner, "submit",
                            lambda job, done: started.append(job))
        shown = []
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "information",
                            lambda *a, **k: shown.append(a))

        screen.table.selectRow(1)          # the failed one
        screen._on_row_activated()
        assert not started, "a failed trial was re-run anyway"
        assert shown, "a failed trial was silently ignored"

    def test_the_figures_land_in_the_queue_and_are_editable(self, screen):
        """Live figures, so right-click restyling works on them."""
        from matplotlib.figure import Figure

        figure = Figure()
        figure.add_subplot(111).plot([0, 1], [0, 1])
        screen._trial_figures_ready(
            {"settings": {"regression_type": "ols"},
             "output": {}, "figures": [figure]})

        assert screen.figures.count() == 1
        assert screen.figures.has_live_figure(0), \
            "the figure arrived as a picture and cannot be restyled"
        assert "ols" in screen.trial_status.text()

    def test_a_row_that_produced_nothing_says_so(self, screen):
        screen._trial_figures_ready(
            {"settings": {}, "output": {}, "figures": []})
        assert "no figures" in screen.trial_status.text().lower()


class TestARowOpensItsWholeSetOfGraphs:
    """Instruction 116. Re-running was the expensive half and already worked;
    showing one figure at a time meant the user still could not put a run's
    residual plot beside its volcano, which is the comparison that decides
    whether a configuration is any good."""

    @pytest.fixture()
    def screen(self, qtbot):
        from spacr.qt.screens.parameter_sweep import _make_screen

        widget = _make_screen()
        qtbot.addWidget(widget)
        return widget

    def test_a_saved_trial_loads_without_refitting(self, screen, tmp_path,
                                                   monkeypatch):
        """A saved run is instant; a re-fit is a minute for an identical
        answer."""
        import numpy as np

        run = tmp_path / "trial_0001"
        run.mkdir()
        pd.DataFrame({
            "feature": ["gene_fraction:gene[225160]"],
            "coefficient": [1.2], "p_value": [4.6e-08], "q_value": [1.8e-05],
        }).to_csv(run / "results.csv", index=False)

        frame = pd.DataFrame([{"trial_id": 1, "status": "ok",
                               "folder": str(run), "regression_type": "ols"}])
        screen._results = frame
        screen._show(frame)
        screen.table.selectRow(0)

        started = []
        monkeypatch.setattr(screen._runner, "submit",
                            lambda job, done: started.append(job))
        screen._on_row_activated()

        assert not started, "it re-fitted a trial whose results were on disk"
        assert screen.results.table.table.rowCount() == 1
        assert "loaded from disk" in screen.trial_status.text()

    def test_a_trial_with_no_saved_results_is_refitted(self, screen, tmp_path,
                                                       monkeypatch):
        frame = pd.DataFrame([{"trial_id": 2, "status": "ok",
                               "folder": str(tmp_path / "empty"),
                               "regression_type": "ols"}])
        screen._results = frame
        screen._show(frame)
        screen.table.selectRow(0)

        started = []
        monkeypatch.setattr(screen._runner, "submit",
                            lambda job, done: started.append(job))
        screen._on_row_activated()

        assert started, "nothing was re-fitted and nothing was loaded"
        assert "re-fitting" in screen.trial_status.text()

    def test_a_refit_populates_the_same_panel(self, screen):
        """So the two paths end in the same place."""
        from matplotlib.figure import Figure

        figure = Figure()
        figure.add_subplot(111).plot([0, 1], [0, 1])
        screen._trial_figures_ready({
            "settings": {"regression_type": "ols", "src": "/tmp/x"},
            "output": {"results": pd.DataFrame({
                "feature": ["a", "b"], "coefficient": [1.0, 2.0],
                "p_value": [0.01, 0.2]})},
            "figures": [figure]})

        assert screen.results.table.table.rowCount() == 2
