"""Where the regression screen looks for a run's inputs and outputs.

Every one of these answers a question of the form "which folder is this run
in?" or "which table did it fit?", and every one is asked of a panel that may
not be built yet, may have been replaced, or may be holding a path that
vanished between one question and the next.

A wrong answer here is not a crash, which is the problem: the sweep writes its
effects grid into the wrong project, the measurement scan reads the previous
run's frame, or the merged measurements land in a temporary directory the user
never finds. So each helper says "I do not know" rather than guessing, and
that is what is pinned below.
"""
from __future__ import annotations

import os
import types

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


class TestWhichFolderTheRunIsIn:

    def test_the_selected_run_wins_over_the_loaded_table(self, screen,
                                                          monkeypatch):
        """Every view follows the active run, not the last file opened."""
        monkeypatch.setattr(screen, "_sweep_runs", types.SimpleNamespace(
            loaded_run_folder=lambda: "/data/plate1/results/run_3"))
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            run_folder=lambda: "/somewhere/else"))

        assert screen._results_source_path() == "/data/plate1/results/run_3"

    def test_a_runs_tab_that_will_not_answer_defers_to_the_panel(
            self, screen, monkeypatch):
        """A half-built Runs tab must not hide a table loaded off disk."""
        def refusing():
            raise RuntimeError("the runs model is not loaded")

        monkeypatch.setattr(screen, "_sweep_runs", types.SimpleNamespace(
            loaded_run_folder=refusing))
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            run_folder=lambda: "/data/plate1/results"))

        assert screen._results_source_path() == "/data/plate1/results"

    def test_with_neither_there_is_no_folder(self, screen, monkeypatch):
        monkeypatch.setattr(screen, "_sweep_runs", None)
        monkeypatch.setattr(screen, "_results_panel", None)

        assert screen._results_source_path() == ""

    def test_an_older_panel_is_asked_for_the_path_it_kept(self, screen,
                                                           monkeypatch):
        """A panel with no ``run_folder`` still knows the file it loaded."""
        monkeypatch.setattr(screen, "_sweep_runs", None)
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            _path="/data/plate1/results/coefficients.csv"))

        assert screen._results_source_path() == (
            "/data/plate1/results/coefficients.csv")


class TestTheFrameTheMeasurementScanRunsOn:

    def _point_at(self, screen, monkeypatch, folder):
        monkeypatch.setattr(screen, "_sweep_runs", None)
        monkeypatch.setattr(screen, "_results_panel",
                            types.SimpleNamespace(run_folder=lambda: folder))

    def test_the_run_s_own_merged_frame_is_what_is_scanned(
            self, screen, monkeypatch, tmp_path):
        """It already carries the gene assignment beside the measurements."""
        (tmp_path / "regression_data.csv").write_text(
            "gene,plateID,response\nA,p1,0.5\n", encoding="utf-8")
        self._point_at(screen, monkeypatch, str(tmp_path))

        frame = screen._scan_source_frame()

        assert list(frame.columns) == ["gene", "plateID", "response"]

    def test_the_older_merged_name_is_read_too(self, screen, monkeypatch,
                                                tmp_path):
        (tmp_path / "merged_data.csv").write_text(
            "gene,response\nA,0.5\n", encoding="utf-8")
        self._point_at(screen, monkeypatch, str(tmp_path))

        assert len(screen._scan_source_frame()) == 1

    def test_a_frame_that_cannot_be_read_is_no_frame_at_all(
            self, screen, monkeypatch, tmp_path):
        """A truncated CSV must not reach the scan as half a table."""
        (tmp_path / "regression_data.csv").write_text("", encoding="utf-8")
        self._point_at(screen, monkeypatch, str(tmp_path))

        assert screen._scan_source_frame() is None

    def test_a_folder_with_neither_file_has_nothing_to_scan(
            self, screen, monkeypatch, tmp_path):
        self._point_at(screen, monkeypatch, str(tmp_path))

        assert screen._scan_source_frame() is None

    def test_no_run_folder_means_no_frame(self, screen, monkeypatch):
        monkeypatch.setattr(screen, "_sweep_runs", None)
        monkeypatch.setattr(screen, "_results_panel", None)

        assert screen._scan_source_frame() is None


class TestKeepingTheSweepsEffectsGrid:

    def test_the_grid_is_written_beside_the_run(self, screen, monkeypatch,
                                                 tmp_path):
        from spacr import cell_montage

        written = {}
        monkeypatch.setattr(cell_montage, "write_effects_grid",
                            lambda effects, folder: written.update(
                                effects=effects, folder=folder))
        monkeypatch.setattr(screen, "_sweep_runs", None)
        monkeypatch.setattr(screen, "_results_panel",
                            types.SimpleNamespace(
                                run_folder=lambda: str(tmp_path)))

        screen._keep_the_effects_grid(
            types.SimpleNamespace(effects=[{"gene": "A"}]))

        assert written["folder"] == str(tmp_path)
        assert written["effects"] == [{"gene": "A"}]

    def test_a_run_folder_given_as_a_file_writes_next_to_it(
            self, screen, monkeypatch, tmp_path):
        """A table loaded off disk names the CSV, not the directory."""
        from spacr import cell_montage

        table = tmp_path / "coefficients.csv"
        table.write_text("a\n", encoding="utf-8")
        written = {}
        monkeypatch.setattr(cell_montage, "write_effects_grid",
                            lambda effects, folder: written.update(
                                folder=folder))
        monkeypatch.setattr(screen, "_sweep_runs", None)
        monkeypatch.setattr(screen, "_results_panel",
                            types.SimpleNamespace(_path=str(table)))

        screen._keep_the_effects_grid(
            types.SimpleNamespace(effects=[{"gene": "A"}]))

        assert written["folder"] == str(tmp_path)

    def test_a_sweep_with_no_effects_writes_nothing(self, screen,
                                                     monkeypatch):
        from spacr import cell_montage

        monkeypatch.setattr(cell_montage, "write_effects_grid",
                            lambda *a, **k: pytest.fail("nothing to write"))

        screen._keep_the_effects_grid(types.SimpleNamespace(effects=[]))
        screen._keep_the_effects_grid(types.SimpleNamespace(effects=None))
        screen._keep_the_effects_grid(object())

    def test_a_montage_that_refuses_does_not_fail_the_sweep(
            self, screen, monkeypatch, tmp_path):
        """The sweep produced its answer; the grid is a picture of it."""
        from spacr import cell_montage

        attempts = []

        def refusing(effects, folder):
            attempts.append((effects, folder))
            raise OSError("the run folder is read-only")

        monkeypatch.setattr(cell_montage, "write_effects_grid", refusing)
        monkeypatch.setattr(screen, "_sweep_runs", None)
        monkeypatch.setattr(screen, "_results_panel",
                            types.SimpleNamespace(
                                run_folder=lambda: str(tmp_path)))

        screen._keep_the_effects_grid(
            types.SimpleNamespace(effects=[{"gene": "A"}]))

        assert attempts == [([{"gene": "A"}], str(tmp_path))]


class TestTheSweepsInputs:

    def test_the_scores_are_read_from_every_attached_database(
            self, screen, monkeypatch, tmp_path):
        """The merged measurements carry no score column at all."""
        first = tmp_path / "a.csv"
        first.write_text("prcfo,plate,row,col,score\nx,p1,r1,c1,0.9\n",
                         encoding="utf-8")
        second = tmp_path / "b.csv"
        second.write_text("prcfo,plate,row,col,score\ny,p1,r1,c2,0.1\n",
                          encoding="utf-8")
        monkeypatch.setattr(screen, "_attached_database_rows", lambda: [
            {"score": str(first)}, {"score": str(second)}])

        frame = screen._sweep_scores()

        assert len(frame) == 2
        assert "score" in frame.columns

    def test_no_score_file_means_the_column_is_absent_not_zero(
            self, screen, monkeypatch):
        """Reporting zeros would read as "no cell was called positive"."""
        monkeypatch.setattr(screen, "_attached_database_rows", lambda: [
            {"score": ""}, "not a row"])

        assert screen._sweep_scores() is None

    def test_a_score_file_that_cannot_be_parsed_is_no_scores(
            self, screen, monkeypatch, tmp_path):
        broken = tmp_path / "broken.csv"
        broken.write_text("", encoding="utf-8")
        monkeypatch.setattr(screen, "_attached_database_rows",
                            lambda: [{"score": str(broken)}])

        assert screen._sweep_scores() is None

    def test_counts_that_cannot_be_turned_into_fractions_are_no_counts(
            self, screen, monkeypatch):
        from spacr import cell_montage

        def refusing(_paths):
            raise ValueError("the count file has no grna column")

        monkeypatch.setattr(cell_montage, "fractions_from_counts", refusing)
        monkeypatch.setattr(screen, "_attached_database_rows",
                            lambda: [{"count": "/data/counts.csv"}])

        assert screen._sweep_counts() is None

    def test_no_count_file_at_all_is_no_counts(self, screen, monkeypatch):
        monkeypatch.setattr(screen, "_attached_database_rows",
                            lambda: [{"count": ""}])

        assert screen._sweep_counts() is None

    def test_a_sweep_card_with_no_destination_field_names_no_folder(
            self, screen, monkeypatch):
        """Asked of the card, so the two cannot disagree about the folder."""
        monkeypatch.setattr(screen, "_sweep", types.SimpleNamespace())

        assert screen._sweep_destination() == ""


class TestReadingThePanelsOwnSettings:

    def test_attached_databases_that_will_not_be_read_are_none(
            self, screen, monkeypatch):
        """The provider must not assume the Measurements tab was built."""
        model = screen._settings_model

        class Refusing:
            def get_value(self):
                raise RuntimeError("the table model has gone")

        monkeypatch.setitem(model._widgets, "paired_data", Refusing())

        assert screen._attached_database_rows() == []

    def test_a_column_fit_reads_the_panel_live(self, screen):
        """Copied at the moment Run is pressed, so the queue fits one model."""
        settings = screen._column_fit_settings()

        assert isinstance(settings, dict) and settings

    def test_a_column_fit_with_no_panel_carries_no_settings(self, screen,
                                                             monkeypatch):
        monkeypatch.setattr(screen, "_settings_model", None)

        assert screen._column_fit_settings() == {}

    def test_a_panel_that_will_not_be_collected_carries_no_settings(
            self, screen, monkeypatch):
        def refusing():
            raise RuntimeError("a widget was destroyed mid-collect")

        monkeypatch.setattr(screen._settings_model, "collect", refusing)

        assert screen._column_fit_settings() == {}

    def test_the_umap_display_window_offers_only_what_is_set(self, screen,
                                                              monkeypatch):
        """A dialog showing 0 for a setting the user set to 20 is worse."""
        monkeypatch.setattr(screen._settings_model, "collect", lambda: {
            "figuresize": 10, "image_nr": 16, "img_zoom": None,
            "verbose": True})

        assert screen._umap_display_defaults() == {
            "figuresize": 10, "image_nr": 16}

    def test_the_umap_display_window_offers_nothing_without_a_panel(
            self, screen, monkeypatch):
        monkeypatch.setattr(screen, "_settings_model", None)

        assert screen._umap_display_defaults() == {}

    def test_the_umap_display_window_offers_nothing_it_could_not_read(
            self, screen, monkeypatch):
        def refusing():
            raise RuntimeError("a widget was destroyed mid-collect")

        monkeypatch.setattr(screen._settings_model, "collect", refusing)

        assert screen._umap_display_defaults() == {}

    def test_the_merged_measurements_still_get_a_home_when_the_panel_is_mute(
            self, screen, monkeypatch, tmp_path):
        """The frame must land beside the runs, not in a temporary directory.

        With the settings unreadable the rule falls through to the attached
        databases, which is the branch a regression project actually uses --
        that screen has no ``src``.
        """
        def refusing():
            raise RuntimeError("a widget was destroyed mid-collect")

        monkeypatch.setattr(screen._settings_model, "collect", refusing)
        plate = tmp_path / "plate1"
        plate.mkdir()
        database = plate / "measurements" / "measurements.db"
        database.parent.mkdir()
        database.write_text("", encoding="utf-8")
        monkeypatch.setattr(screen, "_attached_database_rows",
                            lambda: [{"database": str(database)}])

        destination = screen._measurements_destination()

        assert destination.endswith("measurements")
        assert str(tmp_path) in destination
