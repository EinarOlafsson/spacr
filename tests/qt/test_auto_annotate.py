"""Auto-annotation: four ways to pick a population, one way to write it.

Annotating a screen by hand is the slowest step in the pipeline, and every
mechanism for picking a population already existed somewhere in spaCR. What
was missing was the route from those mechanisms into an annotation column.

Two sources are implemented here (metadata, measurement thresholds) and two
are hand-offs (the Gate Editor, the Image UMAP). That split is deliberate:
both of those already select populations AND already write annotations, so
reimplementing either would put a second copy of the gate maths, or of the
clustering, on a path that could drift from the one the user sees.

The property these tests defend hardest is that **nothing is written until a
count has been shown**. A bulk annotation is not undoable through the grid --
the undo stack only holds the slots on the current page -- so the preview is
the safety.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project whose png_list has real metadata and a measurement table."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows, cells = [], []
    for i in range(12):
        p = src / "data" / f"cell_{i:02d}.png"
        Image.fromarray(
            rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)).save(p)
        well = "c1" if i < 6 else "c2"
        prcfo = f"plate1_r1_{well}_f1_o{i}"
        # cell_id, because spacr.io's join anchors png_list onto the cell
        # table through it. A fixture without it is not a spaCR project.
        rows.append((str(p), "plate1", f"r1{well}", "r1", well, "f1",
                     i, None, prcfo, i))
        cells.append((prcfo, "plate1", "r1", well, "f1", i,
                      float(100 + i * 100), float(10 + i)))

    db = src / "measurements" / "measurements.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            'CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, plateID TEXT,'
            ' wellID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT,'
            ' label INTEGER, annotate INTEGER, prcfo TEXT,'
            ' cell_id INTEGER)')
        con.executemany(
            'INSERT INTO "png_list" VALUES (?,?,?,?,?,?,?,?,?,?)', rows)
        con.execute(
            'CREATE TABLE "cell" (prcfo TEXT PRIMARY KEY, plateID TEXT,'
            ' rowID TEXT, columnID TEXT, fieldID TEXT, object_label INTEGER,'
            ' cell_area REAL, nucleus_area REAL)')
        con.executemany('INSERT INTO "cell" VALUES (?,?,?,?,?,?,?,?)', cells)
        con.commit()
    finally:
        con.close()
    return src


def _db(project: Path) -> str:
    return str(project / "measurements" / "measurements.db")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_values_are_read_from_the_database(project):
    """Not guessed from a naming convention. A picker offering rows A-H to
    someone whose plate is numbered is a picker they cannot use."""
    from spacr.qt.annotate_engine import metadata_values

    assert metadata_values(_db(project), "columnID") == ["c1", "c2"]
    # timeID is absent from this (non-timelapse) project, and asking for a
    # column that is not there must be empty rather than an error.
    assert metadata_values(_db(project), "timeID") == []
    assert metadata_values(_db(project), "plateID") == ["plate1"]


def test_an_unknown_metadata_column_is_refused(project):
    """It would otherwise interpolate an arbitrary name straight into SQL."""
    from spacr.qt.annotate_engine import metadata_values, paths_by_metadata

    with pytest.raises(ValueError, match="not a metadata column"):
        metadata_values(_db(project), "png_path; DROP TABLE png_list--")
    with pytest.raises(ValueError, match="not a metadata column"):
        paths_by_metadata(_db(project), "annotate", ["1"])


def test_paths_by_metadata_selects_exactly_that_population(project):
    from spacr.qt.annotate_engine import paths_by_metadata

    c1 = paths_by_metadata(_db(project), "columnID", ["c1"])
    c2 = paths_by_metadata(_db(project), "columnID", ["c2"])
    assert len(c1) == 6 and len(c2) == 6
    assert not set(c1) & set(c2)
    both = paths_by_metadata(_db(project), "columnID", ["c1", "c2"])
    assert len(both) == 12


def test_a_value_that_matches_nothing_returns_nothing(project):
    from spacr.qt.annotate_engine import paths_by_metadata

    assert paths_by_metadata(_db(project), "columnID", ["c9"]) == []


# ---------------------------------------------------------------------------
# Measurements -- several at once, which is the point
# ---------------------------------------------------------------------------

def test_several_measurement_rules_are_anded(project):
    """One threshold is a gate, not a population. Asked for explicitly."""
    from spacr.qt.annotate_engine import paths_by_measurements

    wide = paths_by_measurements(_db(project), "annotate", [
        {"column": "cell_area", "threshold": 500, "direction": "higher"},
    ])
    narrow = paths_by_measurements(_db(project), "annotate", [
        {"column": "cell_area", "threshold": 500, "direction": "higher"},
        {"column": "nucleus_area", "threshold": 19, "direction": "higher"},
    ])
    assert wide, "the fixture must match something or this proves nothing"
    assert set(narrow) < set(wide), (
        "adding a second rule must NARROW the population; if it widened or "
        "left it alone the rules are being ORed")


def test_a_rule_missing_a_field_is_refused(project):
    """Skipping it would widen the population and label objects nobody
    asked for."""
    from spacr.qt.annotate_engine import paths_by_measurements

    with pytest.raises(ValueError, match="column and a threshold"):
        paths_by_measurements(_db(project), "annotate",
                              [{"column": "cell_area"}])
    with pytest.raises(ValueError, match="higher.*lower"):
        paths_by_measurements(_db(project), "annotate", [
            {"column": "cell_area", "threshold": 1, "direction": "above"}])


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

def _dialog(qtbot, project):
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import _AutoAnnotateDialog

    settings = AnnotateSettings(
        src=str(project), db_path=_db(project), annotation_column="annotate")
    dlg = _AutoAnnotateDialog(settings)
    qtbot.addWidget(dlg)
    return dlg


def test_apply_is_disabled_until_a_preview_has_been_taken(qtbot, project,
                                                          qt_theme_applied):
    """The safety. A bulk annotation is not undoable through the grid."""
    dlg = _dialog(qtbot, project)
    assert dlg._apply.isEnabled() is False
    assert dlg.matched_paths() == []

    dlg._column.setCurrentText("columnID")
    dlg._values.setText("c1")
    dlg._on_preview()
    assert dlg._apply.isEnabled() is True
    assert len(dlg.matched_paths()) == 6


def test_changing_the_selection_invalidates_the_preview(qtbot, project,
                                                        qt_theme_applied):
    """Otherwise Apply would write the PREVIOUS population — the one whose
    count the user read and approved — under the new settings."""
    dlg = _dialog(qtbot, project)
    dlg._column.setCurrentText("columnID")
    dlg._values.setText("c1")
    dlg._on_preview()
    assert dlg._apply.isEnabled()

    dlg._source.setCurrentIndex(1)          # -> measurement
    assert dlg._apply.isEnabled() is False
    assert dlg.matched_paths() == []


def test_the_rule_box_refuses_a_line_it_cannot_read(qtbot, project,
                                                    qt_theme_applied):
    dlg = _dialog(qtbot, project)
    dlg._rules.setPlainText("cell_area 500")
    with pytest.raises(ValueError, match="cannot read rule"):
        dlg.parsed_rules()
    dlg._rules.setPlainText("cell_area > lots")
    with pytest.raises(ValueError, match="not a number")  :
        dlg.parsed_rules()


def test_the_rule_box_parses_both_directions(qtbot, project, qt_theme_applied):
    dlg = _dialog(qtbot, project)
    dlg._rules.setPlainText("cell_area > 500\nnucleus_area < 20")
    assert dlg.parsed_rules() == [
        {"column": "cell_area", "threshold": 500.0, "direction": "higher"},
        {"column": "nucleus_area", "threshold": 20.0, "direction": "lower"},
    ]


# ---------------------------------------------------------------------------
# The write path
# ---------------------------------------------------------------------------

def test_the_bulk_write_goes_through_the_existing_save_worker(
        qtbot, project, qt_theme_applied):
    """Not a new sqlite connection.

    A second writer on measurements.db is a known hazard, and routing bulk
    writes through the worker the screen already owns means they land in the
    same place, in the same order, as annotations made by hand.
    """
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.src = str(project)
    screen._settings.db_path = _db(project)
    screen._open_source(str(project))
    qtbot.waitUntil(lambda: screen._worker is not None, timeout=5000)

    submitted = []
    screen._worker.submit = submitted.append

    from spacr.qt.annotate_engine import paths_by_metadata
    paths = paths_by_metadata(_db(project), "columnID", ["c1"])
    n = screen._apply_bulk_annotation(paths, 3)

    assert n == 6
    assert len(submitted) == 1, "the write must be ONE batch, not one per row"
    assert set(submitted[0]) == set(paths)
    assert set(submitted[0].values()) == {3}
    screen._worker.stop(wait=True)


def test_the_grid_agrees_with_the_database_afterwards(qtbot, project,
                                                      qt_theme_applied):
    """A user who auto-annotates then labels by hand would otherwise be
    looking at stale borders while writing on top of them."""
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 2
    screen._settings.grid_cols = 2
    screen._rebuild_grid()
    screen._open_source(str(project))
    qtbot.waitUntil(lambda: len(screen._page_paths) == 4, timeout=5000)
    screen._worker.submit = lambda batch: None

    on_page = [path for path, _ in screen._page_paths]
    screen._apply_bulk_annotation(on_page, 7)
    assert [v for _, v in screen._page_paths] == [7, 7, 7, 7]
    screen._worker.stop(wait=True)


def test_the_column_is_created_if_it_is_missing(qtbot, project,
                                                qt_theme_applied):
    """An auto-annotation into a fresh column must behave like a hand one."""
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.annotation_column = "brand_new"
    screen._open_source(str(project))
    qtbot.waitUntil(lambda: screen._worker is not None, timeout=5000)
    screen._worker.submit = lambda batch: None

    from spacr.qt.annotate_engine import paths_by_metadata
    screen._apply_bulk_annotation(
        paths_by_metadata(_db(project), "columnID", ["c1"]), 1)

    con = sqlite3.connect(_db(project))
    try:
        cols = {r[1] for r in con.execute('PRAGMA table_info("png_list")')}
    finally:
        con.close()
    assert "brand_new" in cols
    screen._worker.stop(wait=True)
