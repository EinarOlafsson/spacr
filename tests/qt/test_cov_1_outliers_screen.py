"""Loading a table into the Outliers screen, and the ways that goes wrong.

Everything here is about the screen's edges rather than its statistics: the
file picker, the table picker inside a measurement database, a scan asked for
with nothing ticked, and an export to a place that cannot be written. Each one
has to end in a sentence in the status line -- the screen has no modal, because
a dialog nobody can dismiss is how a headless run hangs.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.outliers import OutliersScreen, _cell
from spacr.qt.widgets.outlier_model import OBJECT_COLUMNS

pytestmark = pytest.mark.qt


def planted_plate(seed: int = 3, bad_well: int = 3, shift: float = 1.4,
                  n_wells: int = 8, per_well: int = 25) -> pd.DataFrame:
    """Eight wells, one of them shifted; small enough to draw entirely."""
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(n_wells):
        factor = shift if well == bad_well else 1.0
        area = factor * rng.lognormal(0.0, 0.2, per_well)
        perimeter = rng.lognormal(0.0, 0.2, per_well)
        for i in range(per_well):
            rows.append(("p1", "r1", f"c{well + 1}", "f1", i,
                         area[i], perimeter[i]))
    return pd.DataFrame(rows, columns=[
        "plateID", "rowID", "columnID", "fieldID", "object_label",
        "cell_area", "cell_perimeter"])


@pytest.fixture
def screen(qtbot):
    """The screen with its jobs inline, so a scan is finished when it returns."""
    widget = OutliersScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _measurement_db(path, tables):
    with sqlite3.connect(str(path)) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_a_csv_loads_and_scans_itself(screen, tmp_path):
    """A path is enough: read it, then scan what was read."""
    frame = planted_plate()
    csv = tmp_path / "objects.csv"
    frame.to_csv(csv, index=False)

    screen.load_path(str(csv))

    assert screen.result is not None
    assert screen.frame is not None and len(screen.frame) == len(frame)
    assert list(screen.frame.columns) == list(frame.columns)
    assert screen._table_picker.isVisibleTo(screen) is False, (
        "a CSV holds one table, so there is nothing to pick")


def test_a_loaded_table_is_labelled_with_the_file_it_came_from(screen,
                                                               tmp_path,
                                                               monkeypatch):
    """The label names the file and its shape before the scan overwrites it.

    It is the only record on screen of WHICH table produced the flags, and a
    session that has loaded two files is unreadable without it.
    """
    frame = planted_plate()
    csv = tmp_path / "objects.csv"
    frame.to_csv(csv, index=False)
    scans = []
    monkeypatch.setattr(screen, "scan", lambda: scans.append(True))

    screen.load_path(str(csv))

    assert scans == [True], "loading a table asks for a scan of it"
    assert "objects.csv" in screen._source.text()
    assert f"{len(frame):,} rows × {len(frame.columns)} columns" in \
        screen._source.text()


def test_a_database_offers_its_tables_and_reads_the_one_picked(screen,
                                                               tmp_path):
    """A measurement database has several tables; the user picks one.

    Picking a different table is a different question, so it re-reads and
    re-scans rather than re-labelling the flags already on screen.
    """
    frame = planted_plate()
    other = frame.assign(cell_area=frame["cell_area"] * 10.0)
    path = _measurement_db(tmp_path / "measurements.db",
                           {"object": frame, "second_pass": other})

    screen.load_path(path)
    assert set(screen._table_picker.itemText(i)
               for i in range(screen._table_picker.count())) == {
        "object", "second_pass"}
    first = screen.frame["cell_area"].mean()

    screen._on_table_picked("second_pass")

    assert screen._table_picker.currentText() == "second_pass"
    assert screen.frame["cell_area"].mean() == pytest.approx(first * 10.0)
    assert screen.result is not None, "the new table was scanned, not relabelled"


def test_choosing_a_table_from_the_file_dialog_loads_it(screen, tmp_path,
                                                        monkeypatch):
    """The Open button is the same path as ``load_path``, with a dialog."""
    csv = tmp_path / "objects.csv"
    planted_plate().to_csv(csv, index=False)
    monkeypatch.setattr(
        "spacr.qt.screens.outliers.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(csv), "")))

    screen.choose_table()

    assert screen.result is not None
    assert screen.frame is not None and len(screen.frame) == 200


def test_a_cancelled_file_dialog_loads_nothing(screen, monkeypatch):
    """Dismissing the picker leaves the screen exactly as it was."""
    monkeypatch.setattr(
        "spacr.qt.screens.outliers.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: ("", "")))
    before = screen._source.text()

    screen.choose_table()

    assert screen.frame is None
    assert screen.result is None
    assert screen._source.text() == before


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def test_a_method_this_build_does_not_have_is_refused_in_words(screen,
                                                               monkeypatch):
    """A spec the engine refuses is a sentence, never a traceback.

    The spec round-trips through JSON, so a view saved by another build can
    name a method this one has never heard of. It is refused where the spec is
    built -- before a 200,000-row fit starts -- and the engine's own wording
    reaches the status line.
    """
    screen.set_frame(planted_plate(), scan=False)
    monkeypatch.setattr(screen, "current_method", lambda: "isolation_forest")
    failures = []
    screen.failed.connect(failures.append)

    screen.scan()

    assert screen.result is None
    assert failures, "the refusal is published, not only printed"
    assert "isolation_forest" in failures[0]
    assert screen._source.text() == failures[0]
    assert screen.report.toPlainText() == failures[0]
    assert screen._export.isEnabled() is False


def test_the_flagged_table_is_offered_with_its_flag_columns(screen):
    """``objects_frame`` is the whole table plus the engine's columns.

    It is what the export writes, so a host embedding this screen can hand the
    same frame on without re-running anything.
    """
    frame = planted_plate()
    assert screen.objects_frame() is None

    screen.set_frame(frame)
    objects = screen.objects_frame()

    assert len(objects) == len(frame)
    for name in OBJECT_COLUMNS.values():
        assert name in objects.columns
    assert screen.active_jobs() == 0, "an inline scan leaves no worker behind"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_an_export_that_cannot_be_written_says_so_and_stops(screen, tmp_path,
                                                            monkeypatch):
    """One unwritable destination is a message, not four half-written files."""
    screen.set_frame(planted_plate())
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    monkeypatch.setattr(
        "spacr.qt.screens.outliers.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(blocker / "scan.csv"), "")))

    screen.export_csv()

    assert "could not write those files" in screen._source.text()
    assert not (tmp_path / "scan_objects.csv").exists()


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def test_a_missing_number_is_an_empty_cell():
    """NaN is drawn as nothing at all.

    A well with too few objects has no score, and a table full of the string
    "nan" reads as a computed value rather than as a gap.
    """
    assert _cell(float("nan")) == ""
    assert _cell(np.nan) == ""
    assert _cell(1.5) == "1.5"
    assert _cell(True) == "yes"
    assert _cell(None) == ""
