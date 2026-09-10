"""Finding a run's pictures, and what the grid does with the ones it cannot use.

A regression run scatters figures over three places: the run folder and its
subfolders, the screen's ``results/`` folder for the two sequencing panels, and
the screen root for the plate heatmaps. Which of those is walked recursively is
not a detail -- recursing the screen folder pulls every SIBLING run's figures
into the grid under this run's heading, which is worse than missing them.

The other half is refusing to draw. A PDF that will not render and an image
that will not decode both have to disappear from the list rather than become a
null tile, because the caption list is positional: one null and every figure
after it is captioned with its neighbour's name.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage                                 # noqa: E402

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402

pytestmark = pytest.mark.qt


def _png(path, size=8):
    image = QImage(size, size, QImage.Format.Format_RGB32)
    image.fill(0x336699)
    assert image.save(str(path), "PNG")


class TestWhichPicturesBelongToARun:

    def test_a_run_folder_is_walked_into_its_subfolders(self, tmp_path):
        """Most of a run's figures are in one: the QC panels, the summaries."""
        _png(tmp_path / "volcano.png")
        qc = tmp_path / "regression_qc"
        qc.mkdir()
        _png(qc / "residuals.png")

        names = AppScreen._figure_names_under(str(tmp_path), recursive=True)

        assert names == [os.path.join("regression_qc", "residuals.png"),
                         "volcano.png"]

    def test_a_screen_folder_is_not_walked_into_the_other_runs(self, tmp_path):
        """Its subfolders ARE the sibling runs."""
        _png(tmp_path / "fraction_threshold.png")
        sibling = tmp_path / "regression_2"
        sibling.mkdir()
        _png(sibling / "volcano.png")

        names = AppScreen._figure_names_under(str(tmp_path), recursive=False)

        assert names == ["fraction_threshold.png"]

    def test_only_pictures_count(self, tmp_path):
        _png(tmp_path / "volcano.png")
        (tmp_path / "coefficients.csv").write_text("a\n", encoding="utf-8")
        (tmp_path / "settings.json").write_text("{}", encoding="utf-8")

        assert AppScreen._figure_names_under(str(tmp_path)) == ["volcano.png"]

    def test_a_folder_that_cannot_be_listed_has_no_pictures(self, tmp_path):
        """A run folder on a share that went away is not an error here.

        The grid is a view of a run that has already finished; refusing to
        draw anything at all is the right answer, and raising would take the
        whole results tab with it.
        """
        assert AppScreen._figure_names_under(
            str(tmp_path / "never-existed"), recursive=False) == []
        assert AppScreen._figure_names_under(
            str(tmp_path / "never-existed"), recursive=True) == []


class TestTheScreenFoldersAboveARun:

    def test_a_run_under_results_reaches_the_screen_root_as_well(self,
                                                                  tmp_path):
        """Two of the three shared figures are in ``results/``; one is not.

        ``plot_plates`` writes the plate heatmaps a level further up again,
        so the answer is a list rather than a parent.
        """
        run = tmp_path / "screen" / "results" / "regression_1"
        run.mkdir(parents=True)

        folders = AppScreen._screen_folders_above(str(run))

        assert folders == [str(tmp_path / "screen" / "results"),
                           str(tmp_path / "screen")]

    def test_a_folder_that_is_not_in_that_layout_only_gives_its_parent(
            self, tmp_path):
        """A bare directory a user pointed at has no screen above it."""
        run = tmp_path / "somewhere" / "my_run"
        run.mkdir(parents=True)

        assert AppScreen._screen_folders_above(str(run)) == [
            str(tmp_path / "somewhere")]

    def test_the_climb_stops_at_the_top_of_the_filesystem(self):
        """Nothing above the root, and no exception on the way to finding out."""
        assert AppScreen._screen_folders_above("/") == []
        assert AppScreen._screen_folders_above("") == []


class TestTurningNamesIntoPictures:

    def test_a_readable_image_becomes_a_tile_with_its_relative_name(self,
                                                                     tmp_path):
        qc = tmp_path / "regression_qc"
        qc.mkdir()
        _png(qc / "residuals.png")

        pixmaps, titles = AppScreen._pictures_from(
            str(tmp_path), [os.path.join("regression_qc", "residuals.png")])

        assert len(pixmaps) == 1 and not pixmaps[0].isNull()
        assert "regression_qc" in titles[0]

    def test_an_image_that_will_not_decode_is_dropped_not_left_null(self,
                                                                     tmp_path):
        """The caption list is positional; a null shifts every name after it."""
        (tmp_path / "broken.png").write_text("not a png", encoding="utf-8")
        _png(tmp_path / "good.png")

        pixmaps, titles = AppScreen._pictures_from(
            str(tmp_path), ["broken.png", "good.png"])

        assert len(pixmaps) == 1 and len(titles) == 1
        assert "good" in titles[0]

    def test_a_pdf_that_will_not_render_is_dropped_too(self, tmp_path,
                                                        monkeypatch):
        """Rendering a PDF needs a backend that may not be installed."""
        from spacr.qt.widgets import figure_queue

        def refusing(_path):
            raise RuntimeError("no PDF backend is available")

        monkeypatch.setattr(figure_queue, "render_pdf_to_image", refusing)
        (tmp_path / "summary.pdf").write_bytes(b"%PDF-1.4\n")
        _png(tmp_path / "good.png")

        pixmaps, titles = AppScreen._pictures_from(
            str(tmp_path), ["summary.pdf", "good.png"])

        assert len(pixmaps) == 1
        assert "good" in titles[0]

    def test_a_pdf_that_renders_to_nothing_is_dropped(self, tmp_path,
                                                       monkeypatch):
        from spacr.qt.widgets import figure_queue

        monkeypatch.setattr(figure_queue, "render_pdf_to_image",
                            lambda _path: None)
        (tmp_path / "summary.pdf").write_bytes(b"%PDF-1.4\n")

        pixmaps, _titles = AppScreen._pictures_from(str(tmp_path),
                                                    ["summary.pdf"])

        assert pixmaps == []


class TestComparingTwoRunFolders:

    def test_the_same_folder_written_two_ways_is_one_run(self, tmp_path):
        """One end holds a directory, the other a path with ``..`` in it."""
        run = tmp_path / "results" / "regression_1"
        run.mkdir(parents=True)
        awkward = os.path.join(str(run), "..", "regression_1")

        assert AppScreen._same_run_folder(str(run), awkward) is True

    def test_a_row_whose_folder_is_missing_matches_nothing(self):
        """A folder read off a concatenated frame can be NaN, not a string."""
        assert AppScreen._same_run_folder("", "/data/run") is False
        assert AppScreen._same_run_folder(None, "/data/run") is False
        assert AppScreen._same_run_folder("/data/run", None) is False

    def test_a_folder_that_is_not_a_path_at_all_matches_nothing(self):
        """``float('nan')`` reaches here from a concatenated frame."""
        assert AppScreen._same_run_folder(["/data/run"], "/data/run") is False


class TestWhichPageIsOnScreen:

    def test_an_absent_container_is_not_showing_anything(self):
        assert AppScreen._is_the_page(None, object()) is False
        assert AppScreen._is_the_page(object(), None) is False

    def test_an_empty_container_is_not_showing_anything(self):
        """A stack built before its pages have a current widget of None."""
        empty = types.SimpleNamespace(currentWidget=lambda: None)

        assert AppScreen._is_the_page(empty, object()) is False

    def test_a_widget_inside_the_page_counts_as_the_page(self, qtbot):
        """The grid lives inside a scroll area, the results inside a splitter."""
        from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

        stack = QStackedWidget()
        qtbot.addWidget(stack)
        page = QWidget()
        inner = QWidget()
        QVBoxLayout(page).addWidget(inner)
        stack.addWidget(page)
        stack.setCurrentWidget(page)

        assert AppScreen._is_the_page(stack, page) is True
        assert AppScreen._is_the_page(stack, inner) is True
