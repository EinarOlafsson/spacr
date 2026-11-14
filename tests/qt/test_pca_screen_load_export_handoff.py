"""The PCA screen: loading, the filter upstream of the maths, export, hand-off.

Every path ends in one exact sentence on the source label rather than a dialog
or a traceback -- the screen runs offscreen as often as on a desktop, and a
modal nobody can dismiss is how such a session hangs.  The decomposition is a
property of the population, so a filter change recomputes it (debounced) and
keeps the user's feature ticks.

Offscreen; reads and fits run inline (``threaded=False``) on a private link.
"""
from __future__ import annotations

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

pytestmark = pytest.mark.qt

NO_OPENER = (
    "could not open those objects: nothing is registered to open 'annotate' "
    "objects (registered: nothing). The screen that opens them may not have "
    "been created yet — ask has_object_opener('annotate') before offering "
    "the action.")


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


def _decomposed(screen, features, ticks=("f1", "f2", "f3")):
    screen.set_frame(features)
    screen.pca.features.set_selected(list(ticks))
    screen.pca.recompute()
    return screen.pca.result


def _brush_everything(screen):
    scores = screen.pca.scores_frame
    screen.pca.canvas.brush(float(scores["PC1"].min()) - 1.0,
                            float(scores["PC2"].min()) - 1.0,
                            float(scores["PC1"].max()) + 1.0,
                            float(scores["PC2"].max()) + 1.0)


def _picker_items(screen):
    picker = screen._table_picker
    return [picker.itemText(i) for i in range(picker.count())]


def test_pca_screen_load_path_reports_every_source_on_the_label(
        subtests, screen, synth_sqlite_db, tmp_path, features):
    """CSV, database, broken file, missing or empty table: one exact label."""
    db = str(synth_sqlite_db)
    broken = tmp_path / "not_really.db"
    broken.write_bytes(b"this is not a sqlite file at all")
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    csv = tmp_path / "measurements.csv"
    features.to_csv(csv, index=False)

    with subtests.test(msg="a file that is not a database"):
        screen.load_path(str(broken))
        assert screen._source.text() == (
            "could not read not_really.db: file is not a database")
        assert screen._frame is None

    with subtests.test(msg="a table the database does not hold"):
        screen.load_path(db, table="no_such_table")
        assert screen._source.text() == (
            "could not read measurements.db: Execution failed on sql "
            "'SELECT * FROM \"no_such_table\"': no such table: no_such_table")
        assert screen._frame is None
        assert _picker_items(screen) == ["cell", "png_list"]

    with subtests.test(msg="the cell table"):
        screen.load_path(db, table="cell")
        rows, columns = screen._frame.shape
        assert screen._source.text() == (
            f"measurements.db · cell · {rows} rows × {columns} columns")
        assert screen._table_picker.isVisibleTo(screen)
        assert screen._table_picker.currentText() == "cell"

    with subtests.test(msg="picking a table reloads that table"):
        screen._table_picker.setCurrentText("png_list")
        assert list(screen._frame.columns) == ["prc", "annotation"]
        assert screen._source.text() == (
            f"measurements.db · png_list · {len(screen._frame)} rows "
            "× 2 columns")

    with subtests.test(msg="a database with no tables"):
        screen.load_path(str(empty))
        assert not screen._table_picker.isVisibleTo(screen)
        assert _picker_items(screen) == []
        assert screen._source.text() == (
            "could not read empty.db: Execution failed on sql "
            "'SELECT * FROM \"object\"': no such table: object")

    with subtests.test(msg="a CSV needs no table picker"):
        screen.load_path(str(csv))
        assert not screen._table_picker.isVisibleTo(screen)
        assert screen._frame.shape == (60, 10)
        assert screen._source.text() == (
            "measurements.csv · 60 rows × 10 columns")


def test_pca_screen_filter_change_recomputes_on_the_narrowed_population(
        subtests, screen, link, features):
    """Debounced, recomputed on the filtered rows, ticks kept or defaults."""
    control = DataFilter([CategoryFilter("gene", ("control",))])

    with subtests.test(msg="the change is coalesced"):
        screen.set_frame(features)
        assert screen._source.text() == "60 rows × 10 columns"
        assert screen.pca.features.selected() == ("f1", "f2", "f3", "f4")
        assert not screen._refilter.isActive()
        link.set_filter(control)
        assert screen._refilter.isActive()
        assert screen._refilter.interval() == pca_screen.REFILTER_MS == 250

    with subtests.test(msg="recomputing keeps the feature ticks"):
        screen.pca.features.set_selected(["f1", "f2", "f3"])
        screen._recompute_filtered()
        assert screen.pca.features.selected() == ("f1", "f2", "f3")
        assert len(screen.pca.result) == 30
        assert screen.pca.result.n_features == 3

    with subtests.test(msg="no ticks lets the defaults stand"):
        screen.pca.features.set_selected([])
        assert screen.pca.features.selected() == ()
        screen._recompute_filtered()
        assert screen.pca.features.selected() == ("f1", "f2", "f3", "f4")
        assert len(screen.pca.result) == 30
        assert screen.pca.result.n_features == 4


def test_pca_screen_result_label_follows_the_decomposition(subtests, screen,
                                                           features):
    """A fit names rows, features and PC1; a failure replaces it and greys export."""
    with subtests.test(msg="computed"):
        result = _decomposed(screen, features, ("f1", "f2", "f3", "f4"))
        assert screen._source.text() == (
            f"60 objects × 4 features · "
            f"PC1 {result.explained_variance_ratio[0]:.1%}")
        assert screen._export.isEnabled()

    with subtests.test(msg="failed"):
        screen.pca.failed.emit("only one usable feature")
        assert not screen._export.isEnabled()
        assert screen._source.text() == "only one usable feature"


def test_pca_screen_export_writes_three_files_or_says_why_not(
        subtests, screen, features, tmp_path, monkeypatch):
    """Scores, loadings and variance as three CSVs; cancel and refusal are clean."""
    answer = {"path": ""}
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (answer["path"], "")))

    with subtests.test(msg="nothing to export yet"):
        screen.export_csv()
        assert screen._source.text() == "Nothing to export — run a PCA first."

    result = _decomposed(screen, features, ("f1", "f2", "f3", "f4"))
    fitted = screen._source.text()

    with subtests.test(msg="a cancelled dialog writes nothing"):
        screen.export_csv()
        assert screen._source.text() == fitted
        assert list(tmp_path.iterdir()) == []

    with subtests.test(msg="three files with three row meanings"):
        (tmp_path / "run").mkdir()
        answer["path"] = str(tmp_path / "run" / "pca_scores.csv")
        screen.export_csv()
        assert sorted(p.name for p in (tmp_path / "run").iterdir()) == [
            "pca_scores_loadings.csv", "pca_scores_scores.csv",
            "pca_scores_variance.csv"]
        scores = pd.read_csv(tmp_path / "run" / "pca_scores_scores.csv")
        loadings = pd.read_csv(tmp_path / "run" / "pca_scores_loadings.csv")
        variance = pd.read_csv(tmp_path / "run" / "pca_scores_variance.csv")
        assert len(scores) == 60
        assert len(loadings) == 4
        assert len(variance) == len(result.explained_variance_ratio)
        assert screen._source.text() == (
            "wrote pca_scores_scores / _loadings / _variance .csv")

    with subtests.test(msg="an unwritable destination"):
        answer["path"] = str(tmp_path / "no" / "such" / "pca.csv")
        screen.export_csv()
        assert screen._source.text() == (
            "could not write those files: Cannot save file into a "
            f"non-existent directory: '{tmp_path / 'no' / 'such'}'")


def test_pca_screen_hands_a_brushed_cluster_to_annotate(subtests, qtbot,
                                                         screen, link,
                                                         features):
    """Nothing brushed, no opener, a refusing one and a real one: exact outcomes."""
    with subtests.test(msg="nothing brushed"):
        screen.set_frame(features)
        screen._open_selection()
        assert screen._source.text() == (
            "Brush a cluster first — nothing is selected.")

    _decomposed(screen, features)
    with subtests.test(msg="the hand-off follows the brush"):
        assert not screen._to_annotate.isEnabled()
        _brush_everything(screen)
        qtbot.waitUntil(screen._to_annotate.isEnabled, timeout=5000)
        assert screen.pca.canvas.selected_count() == len(link.selection) == 60

    with subtests.test(msg="nothing is registered to open crops"):
        screen._open_selection()
        assert screen._source.text() == NO_OPENER

    with subtests.test(msg="a process-wide opener is not this link's"):
        def refuse(request):
            raise RuntimeError("the annotate screen is closing")

        register_object_opener("annotate", refuse)
        try:
            screen._open_selection()
        finally:
            unregister_object_opener("annotate")
        assert screen._source.text() == NO_OPENER

    with subtests.test(msg="an opener on the link that refuses"):
        link.register_object_opener("annotate", refuse)
        try:
            screen._open_selection()
        finally:
            link.unregister_object_opener("annotate")
        assert screen._source.text() == (
            "could not open those objects: the annotate screen is closing")

    with subtests.test(msg="a registered opener gets the cluster"):
        opened = []
        link.register_object_opener("annotate", opened.append)
        try:
            screen._open_selection()
        finally:
            link.unregister_object_opener("annotate")
        assert len(opened) == 1
        assert list(opened[0].keys) == list(link.selection.keys)
        assert len(opened[0].keys) == 60
        assert opened[0].reason == "brushed in PCA · PC1 / PC2"
        assert opened[0].kind == "annotate"
        assert opened[0].context == {}


def test_pca_screen_closes_cleanly_twice(qtbot, link, features):
    """A link already let go does not stop the close, and no work survives it."""
    made = pca_screen.PCAScreen(link=link, threaded=False)
    qtbot.addWidget(made)
    made.set_frame(features)
    link.filter_changed.disconnect(made._on_filter_changed)
    made._refilter.start()

    made.close()                                   # must not raise
    made.close()

    assert not made._refilter.isActive()
    assert made.active_jobs() == 0
    assert made.is_busy() is False
