"""What the PCA screen does when the table, the filter or the disk says no.

Every one of these paths ends in a sentence on the source label rather than
in a dialog or a traceback: the screen is used from a headless-ish offscreen
session as often as from a desktop, and a modal nobody can dismiss is how
such a session hangs. The tests read that label.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog                       # noqa: E402

from spacr.qt.linked_selection import (                         # noqa: E402
    LinkedSelection, register_object_opener, unregister_object_opener)
from spacr.qt.screens import pca as pca_screen                  # noqa: E402
from spacr.selection import CategoryFilter, DataFilter          # noqa: E402


@pytest.fixture
def link():
    """A private link, never the process-wide one every open view listens to."""
    return LinkedSelection()


@pytest.fixture
def screen(qtbot, link):
    """A PCA screen whose reads and fits run inline, so a call settles it."""
    made = pca_screen.PCAScreen(link=link, threaded=False)
    qtbot.addWidget(made)
    try:
        yield made
    finally:
        made.close()


@pytest.fixture
def features():
    """Four correlated numeric features and a group column, 60 rows."""
    rng = np.random.default_rng(7)
    shift = np.repeat([0.0, 1.0], 30)
    return pd.DataFrame({
        "plateID": ["p1"] * 60,
        "rowID": ["r1"] * 60,
        "columnID": [f"c{i % 4 + 1}" for i in range(60)],
        "fieldID": ["f1"] * 60,
        "object_label": list(range(60)),
        "f1": 10.0 + 4.0 * shift + rng.normal(scale=0.4, size=60),
        "f2": 20.0 - 3.0 * shift + rng.normal(scale=0.4, size=60),
        "f3": 5.0 + 2.0 * shift + rng.normal(scale=0.4, size=60),
        "f4": rng.normal(scale=1.0, size=60),
        "gene": ["control"] * 30 + ["treated"] * 30,
    })


# ---------------------------------------------------------------------------
# the filter is upstream of the maths
# ---------------------------------------------------------------------------

def test_with_no_table_there_is_nothing_to_filter(screen):
    """``_filtered`` answers None rather than inventing an empty frame."""
    assert screen._filtered() is None


def test_recomputing_with_no_table_does_nothing(screen):
    """A filter change before a load is not an error."""
    screen._recompute_filtered()               # must not raise

    assert screen.pca.result is None


def test_a_filter_naming_a_missing_column_is_reported_not_swallowed(
        screen, link, features, caplog):
    """The whole frame is decomposed, and the mismatch is logged."""
    screen.set_frame(features)
    link.set_filter(DataFilter([CategoryFilter("no_such_column", ("x",))]))

    with caplog.at_level("INFO", logger="spacr.qt.screens.pca"):
        got = screen._filtered()

    assert len(got) == len(features)
    assert "does not apply to this table" in caplog.text


def test_a_filter_change_is_coalesced_before_it_recomputes(screen, link,
                                                           features):
    """A dragged slider costs one PCA, not one per step."""
    screen.set_frame(features)
    assert not screen._refilter.isActive()

    link.set_filter(DataFilter([CategoryFilter("gene", ("control",))]))

    assert screen._refilter.isActive()
    assert screen._refilter.interval() == pca_screen.REFILTER_MS


def test_a_filter_change_with_no_table_starts_no_timer(screen, link):
    """Nothing to recompute means nothing is scheduled."""
    link.set_filter(DataFilter([CategoryFilter("gene", ("control",))]))

    assert not screen._refilter.isActive()


def test_recomputing_keeps_the_feature_ticks(screen, link, features):
    """Narrowing the population does not silently re-tick the features."""
    screen.set_frame(features)
    chosen = ["f1", "f2", "f3"]
    screen.pca.features.set_selected(chosen)
    link.set_filter(DataFilter([CategoryFilter("gene", ("control",))]))

    screen._recompute_filtered()

    assert set(screen.pca.features.selected()) == set(chosen)
    assert len(screen.pca.result) == 30


def test_recomputing_with_no_ticks_lets_the_defaults_stand(screen, link,
                                                          features):
    """Nothing ticked is not a selection to preserve.

    ``set_frame`` chooses a default feature set for the new population.
    Restoring an EMPTY selection over it would leave the PCA with no features
    at all -- a screen that quietly stops computing after the user clears the
    ticks and then narrows the filter, which is an ordinary thing to do in
    that order.
    """
    screen.set_frame(features)
    screen.pca.features.set_selected([])
    assert not screen.pca.features.selected()
    link.set_filter(DataFilter([CategoryFilter("gene", ("control",))]))

    screen._recompute_filtered()

    assert screen.pca.features.selected(), (
        "an empty selection was restored over the defaults")
    assert screen.pca.result is not None


# ---------------------------------------------------------------------------
# loading a table
# ---------------------------------------------------------------------------

def test_choosing_a_table_loads_the_path_the_dialog_returned(
        screen, monkeypatch, synth_sqlite_db):
    """The file dialog's answer goes straight into the read."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(synth_sqlite_db), "")))

    screen.choose_table()

    assert screen._path == str(synth_sqlite_db)
    assert screen._table_picker.isVisibleTo(screen)
    assert "cell" in [screen._table_picker.itemText(i)
                      for i in range(screen._table_picker.count())]
    assert "measurements.db" in screen._source.text()


def test_cancelling_the_file_dialog_loads_nothing(screen, monkeypatch):
    """An empty path is a cancel, not a read of the empty string."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen.choose_table()

    assert screen._path is None
    assert screen._source.text() == "no table loaded"


def test_a_file_that_is_not_a_database_says_so_on_the_label(screen, tmp_path):
    """Listing the tables fails before any read is dispatched."""
    broken = tmp_path / "not_really.db"
    broken.write_bytes(b"this is not a sqlite file at all")

    screen.load_path(str(broken))

    assert "could not read not_really.db" in screen._source.text()
    assert screen._frame is None


def test_a_missing_table_reports_the_reason_inline(screen, synth_sqlite_db):
    """A read that raises lands on the label, never in a modal."""
    screen.load_path(str(synth_sqlite_db), table="no_such_table")

    text = screen._source.text()
    assert "could not read measurements.db" in text
    assert "no_such_table" in text


def test_picking_a_table_reloads_that_table(screen, synth_sqlite_db):
    """Changing the picker is a load, not just a relabelling."""
    screen.load_path(str(synth_sqlite_db), table="cell")
    assert screen._frame is not None

    screen._table_picker.setCurrentText("png_list")

    assert set(screen._frame.columns) == {"prc", "annotation"}
    assert "png_list" in screen._source.text()


def test_picking_a_table_before_any_path_does_nothing(screen):
    """Populating the picker must not fire a read of nothing."""
    screen._on_table_picked("cell")

    assert screen._frame is None


def test_a_csv_is_read_without_a_table_picker(screen, tmp_path, features):
    """A CSV has one table, so no picker is shown."""
    path = tmp_path / "measurements.csv"
    features.to_csv(path, index=False)

    screen.load_path(str(path))

    assert not screen._table_picker.isVisible()
    assert len(screen._frame) == len(features)
    assert "measurements.csv" in screen._source.text()


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def test_a_computed_decomposition_names_its_share_of_the_variance(
        screen, features):
    """The label states rows, features and PC1's share."""
    screen.set_frame(features)
    screen.pca.features.set_selected(["f1", "f2", "f3", "f4"])
    screen.pca.recompute()

    text = screen._source.text()
    assert "60 objects" in text
    assert "4 features" in text
    assert "PC1 " in text and "%" in text
    assert screen._export.isEnabled()


def test_a_failed_decomposition_disables_the_export(screen, features):
    """The message replaces the summary and the export goes away."""
    screen.set_frame(features)
    screen.pca.features.set_selected(["f1", "f2", "f3"])
    screen.pca.recompute()
    assert screen._export.isEnabled()

    screen.pca.failed.emit("only one usable feature")

    assert not screen._export.isEnabled()
    assert screen._source.text() == "only one usable feature"


# ---------------------------------------------------------------------------
# exporting
# ---------------------------------------------------------------------------

def test_exporting_before_a_pca_says_to_run_one(screen):
    """There is nothing to write, and the label says which thing is missing."""
    screen.export_csv()

    assert screen._source.text() == "Nothing to export — run a PCA first."


def test_exporting_writes_three_files_with_three_row_meanings(
        screen, features, tmp_path, monkeypatch):
    """An object, a feature and a component do not share a sheet."""
    screen.set_frame(features)
    screen.pca.features.set_selected(["f1", "f2", "f3", "f4"])
    screen.pca.recompute()
    stem = tmp_path / "run" / "pca_scores.csv"
    stem.parent.mkdir()
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(stem), "")))

    screen.export_csv()

    scores = pd.read_csv(tmp_path / "run" / "pca_scores_scores.csv")
    loadings = pd.read_csv(tmp_path / "run" / "pca_scores_loadings.csv")
    variance = pd.read_csv(tmp_path / "run" / "pca_scores_variance.csv")
    assert len(scores) == 60
    assert len(loadings) == 4
    assert len(variance) == len(screen.pca.result.explained_variance_ratio)
    assert "wrote pca_scores_scores / _loadings / _variance .csv" \
        in screen._source.text()


def test_cancelling_the_save_dialog_writes_nothing(screen, features, tmp_path,
                                                   monkeypatch):
    """An empty path leaves the label and the directory alone."""
    screen.set_frame(features)
    screen.pca.features.set_selected(["f1", "f2", "f3"])
    screen.pca.recompute()
    before = screen._source.text()
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen.export_csv()

    assert screen._source.text() == before
    assert list(tmp_path.iterdir()) == []


def test_a_path_that_cannot_be_written_is_reported_not_raised(
        screen, features, tmp_path, monkeypatch):
    """An unwritable destination becomes a sentence on the label."""
    screen.set_frame(features)
    screen.pca.features.set_selected(["f1", "f2", "f3"])
    screen.pca.recompute()
    nowhere = tmp_path / "no" / "such" / "dir" / "pca.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(nowhere), "")))

    screen.export_csv()

    assert "could not write those files" in screen._source.text()


# ---------------------------------------------------------------------------
# handing a brushed cluster on
# ---------------------------------------------------------------------------

def test_opening_a_selection_with_nothing_brushed_says_to_brush_first(screen,
                                                                      features):
    """The action explains itself rather than opening every object."""
    screen.set_frame(features)

    screen._open_selection()

    assert "Brush a cluster first" in screen._source.text()


def _brush_everything(screen):
    scores = screen.pca.scores_frame
    screen.pca.canvas.brush(float(scores["PC1"].min()) - 1.0,
                            float(scores["PC2"].min()) - 1.0,
                            float(scores["PC1"].max()) + 1.0,
                            float(scores["PC2"].max()) + 1.0)


def test_with_nothing_able_to_show_crops_the_screen_says_which_screen(
        screen, features):
    """A destination that does not exist yet is named, not guessed at."""
    screen.set_frame(features)
    screen.pca.features.set_selected(["f1", "f2", "f3"])
    screen.pca.recompute()
    _brush_everything(screen)
    assert len(screen.pca.canvas.link.selection) > 0

    screen._open_selection()

    assert "open the Annotate screen once" in screen._source.text()


def test_a_brushed_cluster_reaches_the_registered_opener(screen, link,
                                                          features):
    """The request carries the objects and says where they came from."""
    opened = []
    link.register_object_opener("annotate", lambda request: opened.append(request))
    register_object_opener("annotate", lambda request: opened.append(request))
    try:
        screen.set_frame(features)
        screen.pca.features.set_selected(["f1", "f2", "f3"])
        screen.pca.recompute()
        _brush_everything(screen)

        screen._open_selection()
    finally:
        unregister_object_opener("annotate")

    assert len(opened) == 1
    assert len(opened[0].keys) == len(link.selection)
    assert opened[0].reason.startswith("brushed in PCA")


def test_an_opener_that_refuses_is_reported_on_the_label(screen, features):
    """A routing failure is a sentence, not an exception out of a click."""
    def refuse(request):
        raise RuntimeError("the annotate screen is closing")

    register_object_opener("annotate", refuse)
    try:
        screen.set_frame(features)
        screen.pca.features.set_selected(["f1", "f2", "f3"])
        screen.pca.recompute()
        _brush_everything(screen)

        screen._open_selection()
    finally:
        unregister_object_opener("annotate")

    assert "could not open those objects" in screen._source.text()


def test_a_render_with_no_selection_greys_the_hand_off(qtbot, screen,
                                                       features):
    """The hand-off button follows the brush, once the canvas redraws."""
    screen.set_frame(features)
    screen.pca.features.set_selected(["f1", "f2", "f3"])
    screen.pca.recompute()
    assert not screen._to_annotate.isEnabled()

    _brush_everything(screen)

    qtbot.waitUntil(lambda: screen._to_annotate.isEnabled(), timeout=5000)
    assert screen.pca.canvas.selected_count() == len(screen.pca.canvas.link.selection)


# ---------------------------------------------------------------------------
# closing
# ---------------------------------------------------------------------------

def test_closing_twice_is_not_an_error(qtbot, link, features):
    """The link may already have been let go; closing again is still clean."""
    made = pca_screen.PCAScreen(link=link, threaded=False)
    qtbot.addWidget(made)
    made.set_frame(features)
    link.filter_changed.disconnect(made._on_filter_changed)

    made.close()                                   # must not raise

    assert not made._refilter.isActive()
    assert made.active_jobs() == 0
    assert not made.is_busy()


def test_the_factory_builds_a_screen(qtbot):
    """The registry's factory takes the app key and returns the screen."""
    made = pca_screen.make_pca_screen("pca")
    qtbot.addWidget(made)

    assert isinstance(made, pca_screen.PCAScreen)
    made.close()


def test_the_source_label_survives_a_database_with_no_tables(screen, tmp_path):
    """An empty database gives no picker and an empty read, not a crash."""
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()

    screen.load_path(str(path))

    assert not screen._table_picker.isVisible()
    assert os.path.basename(str(path)) in screen._source.text()


def test_a_link_that_refuses_to_disconnect_does_not_stop_the_close(qtbot,
                                                                   link):
    """Closing must free the workers even if the signal cannot be dropped."""
    import types

    made = pca_screen.PCAScreen(link=link, threaded=False)
    qtbot.addWidget(made)

    def refuse(_slot):
        raise TypeError("that slot is not connected")

    made._link = types.SimpleNamespace(
        filter_changed=types.SimpleNamespace(disconnect=refuse))
    made._refilter.start()

    made.close()                                   # must not raise

    assert not made._refilter.isActive()
    assert made.active_jobs() == 0


def test_the_screen_puts_no_row_in_the_registry():
    """PCA is a fold, so importing this module must add no tile.

    This asserted the module's one row was registered and that a second
    ``register()`` did not duplicate it. There is no row and no
    ``register()`` now: PCA is a button on Image UMAP's masthead, and
    the failure worth catching is the row coming back.
    """
    from spacr.qt.app import APPS

    assert not any(row[0] == pca_screen.APP_KEY for row in APPS)
    assert not hasattr(pca_screen, "register")
