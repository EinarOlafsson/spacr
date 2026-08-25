"""Volcano Explorer screen — finding the table, trimming it, and the reader.

The screen is a reader with three jobs and this file drives all three
against real files on disk rather than around them:

* :func:`find_results_table` is handed a CSV, a non-CSV file, a path that
  does not exist, a folder holding several candidate names at once, and a
  parent one level above the folder that actually holds the table;
* :func:`load_results` is given a stacked permutation table -- every
  minimum-support family and two fitted responses in one file -- and has to
  come back with one family, one response and a fresh index, because
  plotting the stack draws each guide two to four times and that reads as
  extra hits;
* the widget itself opens a folder, refuses one with nothing in it, and
  accepts a dropped directory.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFileDialog

from spacr.qt.screens import volcano


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _results_frame(n=24, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "guide": [f"g{i}" for i in range(n)],
        "gene": [f"TGGT1_{200000 + i // 3}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(0, 0.08, n),
        "adjusted_p_value": np.clip(rng.beta(0.6, 6, n), 1e-8, 1),
        "wells_with_guide": rng.integers(1, 12, n),
        "significant": False,
        "alpha": 0.05,
    })


@pytest.fixture
def results_folder(tmp_path):
    """A regression output folder holding a plain ``results.csv``."""
    folder = tmp_path / "run_2024"
    folder.mkdir()
    _results_frame().to_csv(folder / "results.csv", index=False)
    return folder


# ---------------------------------------------------------------------------
# find_results_table
# ---------------------------------------------------------------------------

def test_a_csv_handed_in_directly_is_the_table(results_folder):
    csv = results_folder / "results.csv"
    assert volcano.find_results_table(csv) == os.path.abspath(csv)


def test_a_file_that_is_not_a_csv_is_not_a_table(tmp_path):
    """A PDF of the volcano is the thing the user has open, not the data."""
    pdf = tmp_path / "volcano.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert volcano.find_results_table(pdf) is None


def test_a_path_that_does_not_exist_is_not_a_table(tmp_path):
    assert volcano.find_results_table(tmp_path / "never_ran") is None


def test_the_guide_level_table_wins_over_the_gene_level_one(tmp_path):
    """Both names in one folder: the point must stay a guide, not a gene.

    ``results.csv`` is the gene-level table for a simultaneous fit, so
    preferring it would silently change what one dot means.
    """
    folder = tmp_path / "both"
    folder.mkdir()
    _results_frame().to_csv(folder / "results.csv", index=False)
    _results_frame().to_csv(folder / "results_grna.csv", index=False)
    (folder / "guide_permutation_results_long.csv").write_text(
        _results_frame().to_csv(index=False))
    assert volcano.find_results_table(folder) == os.path.abspath(
        folder / "guide_permutation_results_long.csv")


def test_a_folder_of_the_wrong_csvs_holds_no_table(tmp_path):
    folder = tmp_path / "settings_only"
    folder.mkdir()
    (folder / "settings.csv").write_text("key,value\n")
    assert volcano.find_results_table(folder) is None


def test_the_run_folder_finds_the_table_in_its_leaf(tmp_path, results_folder):
    """Pointing at the run folder, not the ``guide_permutation`` leaf."""
    parent = results_folder.parent
    assert volcano.find_results_table(parent) == os.path.abspath(
        results_folder / "results.csv")


def test_a_user_expanded_tilde_reaches_the_same_table(monkeypatch,
                                                      results_folder):
    """``~/run`` and the absolute path must resolve to one file."""
    monkeypatch.setenv("HOME", str(results_folder.parent))
    assert volcano.find_results_table("~/run_2024") == os.path.abspath(
        results_folder / "results.csv")


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------

def test_a_stacked_permutation_table_is_trimmed_to_one_family(tmp_path):
    """Four support families stacked: each guide must survive exactly once."""
    frames = []
    for threshold in (2, 3, 4, 5):
        part = _results_frame(n=12)
        part["minimum_wells_threshold"] = threshold
        frames.append(part)
    path = tmp_path / "guide_permutation_results_long.csv"
    pd.concat(frames).to_csv(path, index=False)

    loaded = volcano.load_results(path)
    assert set(loaded["minimum_wells_threshold"]) == {2}
    assert loaded["guide"].is_unique
    assert len(loaded) == 12


def test_two_fitted_responses_are_not_pooled_into_one_plot(tmp_path):
    """Each response is its own correction family, so only the first plots."""
    first = _results_frame(n=8)
    first["outcome"] = "infected"
    second = _results_frame(n=8, seed=9)
    second["outcome"] = "parasite_count"
    path = tmp_path / "results_grna.csv"
    pd.concat([first, second]).to_csv(path, index=False)

    loaded = volcano.load_results(path)
    assert set(loaded["outcome"]) == {"infected"}
    assert len(loaded) == 8


def test_a_single_response_table_is_not_filtered_away(tmp_path):
    """One ``outcome`` value everywhere must not trigger the split."""
    frame = _results_frame(n=8)
    frame["outcome"] = "infected"
    path = tmp_path / "results.csv"
    frame.to_csv(path, index=False)
    assert len(volcano.load_results(path)) == 8


def test_the_trimmed_table_is_indexed_from_zero(tmp_path):
    """The explorer indexes points positionally; a gapped index misreads."""
    early = _results_frame(n=6)
    early["minimum_wells_threshold"] = 9
    late = _results_frame(n=6, seed=11)
    late["minimum_wells_threshold"] = 2
    path = tmp_path / "results.csv"
    pd.concat([early, late]).to_csv(path, index=False)

    loaded = volcano.load_results(path)
    assert list(loaded.index) == list(range(6))


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

@pytest.fixture
def screen(qtbot):
    widget = volcano._make_screen(app_key=volcano.APP_KEY, host=None)
    qtbot.addWidget(widget)
    return widget


def test_a_new_screen_says_it_has_nothing_loaded(screen):
    assert "No results loaded" in screen._path_label.text()
    assert screen.acceptDrops() is True


def test_loading_a_folder_names_the_table_it_found(screen, results_folder):
    assert screen.load(results_folder) is True
    assert screen._path_label.text().endswith("results.csv")
    assert len(screen.explorer.results()) == 24


def test_loading_a_folder_with_no_table_says_so_and_refuses(screen, tmp_path):
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    assert screen.load(empty) is False
    assert str(empty) in screen._path_label.text()


def test_the_open_button_loads_the_folder_the_dialog_returned(
        screen, results_folder, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(results_folder)))
    screen._open()
    assert screen._path_label.text().endswith("results.csv")


def test_cancelling_the_open_dialog_leaves_the_screen_alone(screen,
                                                            monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    screen._open()
    assert "No results loaded" in screen._path_label.text()


def _mime(paths):
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return data


def test_a_dragged_folder_is_accepted(screen, results_folder):
    # The QMimeData has to outlive the call: the event does not own it, and
    # letting it go early segfaults inside ``mimeData()``.
    data = _mime([results_folder])
    event = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, data,
                            Qt.LeftButton, Qt.NoModifier)
    event.setAccepted(False)
    screen.dragEnterEvent(event)
    assert event.isAccepted()
    del event, data


def test_dragging_something_that_is_not_a_file_is_not_accepted(screen):
    data = QMimeData()
    data.setText("results.csv")
    event = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, data,
                            Qt.LeftButton, Qt.NoModifier)
    event.setAccepted(False)
    screen.dragEnterEvent(event)
    assert not event.isAccepted()
    del event, data


def test_dropping_a_folder_loads_its_table(screen, results_folder):
    data = _mime([results_folder])
    event = QDropEvent(QPointF(4, 4), Qt.CopyAction, data,
                       Qt.LeftButton, Qt.NoModifier)
    screen.dropEvent(event)
    assert screen._path_label.text().endswith("results.csv")
    assert event.isAccepted()
    del event, data


def test_dropping_several_paths_stops_at_the_first_that_loads(
        screen, results_folder, tmp_path):
    """An empty folder first must not stop the good one behind it loading."""
    empty = tmp_path / "empty_drop"
    empty.mkdir()
    data = _mime([empty, results_folder])
    event = QDropEvent(QPointF(4, 4), Qt.CopyAction, data,
                       Qt.LeftButton, Qt.NoModifier)
    screen.dropEvent(event)
    assert screen._path_label.text().endswith("results.csv")
    del event, data


def test_dropping_a_remote_url_is_accepted_but_loads_nothing(screen):
    """``isLocalFile()`` is false for an http URL; the label must not move."""
    data = QMimeData()
    data.setUrls([QUrl("https://example.org/results.csv")])
    event = QDropEvent(QPointF(4, 4), Qt.CopyAction, data,
                       Qt.LeftButton, Qt.NoModifier)
    screen.dropEvent(event)
    assert "No results loaded" in screen._path_label.text()
    assert event.isAccepted()
    del event, data


def test_importing_the_module_puts_no_row_in_the_registry():
    """The explorer is a fold, so importing it must add no tile.

    This asserted the import-time ``register()`` had run and that a
    second call was a no-op. Both the call and the function are gone:
    the explorer is "Publication figure…" on the Regression volcano and
    a button on that masthead, and the failure worth catching now is the
    row coming back.
    """
    from spacr.qt.app import APPS

    assert not any(row[0] == volcano.APP_KEY for row in APPS)
    assert not hasattr(volcano, "register")


def test_the_gui_only_sentence_is_written_where_the_cli_reads_it():
    """The sentence survived the row, in both of its copies.

    It travelled into ``cli.INTERACTIVE_ONLY`` as the row's
    ``cli_note=`` and ``unregister_app`` takes a pushed entry back out
    with the row, so without a hand-written copy ``spacr-run
    volcano_explorer`` would stop naming the renderer to call and start
    guessing at a typo. ``spacr.cli`` answers ``--list`` where PySide6 is
    not installed, so it cannot read the sentence from here; the two
    copies are pinned equal instead.
    """
    from spacr import cli

    assert cli.INTERACTIVE_ONLY[volcano.APP_KEY] == volcano.APP_CLI_NOTE
    assert volcano.APP_KEY not in cli.MODULES
