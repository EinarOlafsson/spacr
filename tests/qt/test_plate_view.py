"""Plate Viewer — the heatmap screen and the edge-effect report beside it.

Everything runs offscreen against a *real* temporary ``measurements.db``
holding a 96-well plate with a known +35 % outer ring and two
deliberately thin wells, so every assertion is about a number that was
put there on purpose.

The four properties the screen lives or dies by:

* it **draws the plate it was asked for** — the right grid, the right
  format, the right number of wells;
* a **click lands on the right well** and says what is behind it;
* it is **read-only** — the file on disk is byte-identical after a full
  open / render / export cycle;
* **errors land inline**, never in a modal dialog (a QMessageBox would
  hang a headless run forever).
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr import plate_qc as pqc
from spacr.qt.screens.plate_view import (
    COLOUR_SCALES,
    DEFAULT_CMAP,
    PlateGridWidget,
    PlateViewScreen,
    PREFERRED_TABLES,
)


N_ROWS, N_COLS = 8, 12
N_OBJECTS = 6
EDGE_BOOST = 0.35
#: Wells given a single object each — they must vanish under min_count.
THIN_WELLS = ((2, 3), (5, 9))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    ``MakeMasksScreen._load_current`` once hung the whole headless suite
    on a QMessageBox; this fixture makes that failure mode impossible to
    reintroduce here without a red test.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


def _rows():
    """Deterministic 96-well plate with a +35 % outer ring."""
    rng = np.random.default_rng(2024)
    thin = set(THIN_WELLS)
    out = []
    for r in range(1, N_ROWS + 1):
        for c in range(1, N_COLS + 1):
            ring = min(r - 1, N_ROWS - r, c - 1, N_COLS - c)
            mult = 1.0 + (EDGE_BOOST if ring == 0 else 0.0)
            for _ in range(1 if (r, c) in thin else N_OBJECTS):
                out.append((f"plate1_r{r}_c{c}",
                            100.0 * mult * float(rng.lognormal(0.0, 0.1)),
                            "text"))
    return out


@pytest.fixture
def measdb(tmp_path):
    """``<src>/measurements/measurements.db`` — returns ``(src, db_path)``."""
    src = tmp_path / "plate1"
    meas = src / "measurements"
    meas.mkdir(parents=True)
    db_path = meas / "measurements.db"
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE cell (prc TEXT, value REAL, note TEXT)")
        con.executemany("INSERT INTO cell VALUES (?, ?, ?)", _rows())
        con.execute("CREATE TABLE png_list (png_path TEXT, prc TEXT, "
                    "score REAL)")
        con.execute("INSERT INTO png_list VALUES ('a.png', 'plate1_r1_c1', 1.0)")
        con.commit()
    finally:
        con.close()
    return str(src), str(db_path)


@pytest.fixture
def frame():
    """The same data as a pandas frame, for expectation arithmetic."""
    return pd.DataFrame(_rows(), columns=["prc", "value", "note"])


@pytest.fixture
def screen(qtbot):
    """A synchronous screen — jobs run inline so assertions are immediate."""
    widget = PlateViewScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def rendered(screen, measdb):
    """A screen with the plate already drawn."""
    _src, db = measdb
    assert screen.open_database(db)
    assert screen.render_plate()
    return screen


def _sized_grid(screen, width=760, height=520):
    """Give the grid a real size so ``cell_rect`` means something."""
    grid = screen._grid
    grid.resize(width, height)
    assert grid.width() >= 300 and grid.height() >= 200
    return grid


# ---------------------------------------------------------------------------
# Construction + registration
# ---------------------------------------------------------------------------

def test_the_screen_builds_offscreen_and_asks_for_a_database(screen):
    assert screen.status_text().startswith("Choose a measurements.db")
    assert screen.last_error == ""
    assert screen.report_text() == ""
    assert not screen._grid.has_plate()
    assert not screen._btn_render.isEnabled()
    assert not screen._btn_export.isEnabled()


def test_it_is_registered_under_results_and_qc_as_alpha():
    """Wiring check — asserts the registration once app.py carries it.

    A plate heatmap is reading a result whether or not the screen is
    finished, which is why #16j took it back out of the "Alpha modules"
    section #16i had put it in and left the alpha mark on its own
    axis."""
    from spacr.qt.app import APPS
    entry = next((a for a in APPS if a[0] == "plate_view"), None)
    if entry is None:
        pytest.skip("plate_view not registered in spacr.qt.app.APPS yet")
    key, name, description, section = entry
    assert name == "Plate Viewer"
    from spacr.qt.app import SECTION_RESULTS, app_stage
    assert section == SECTION_RESULTS
    assert app_stage(key) == "alpha"
    assert description
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    assert APP_TITLES.get("plate_view")
    assert APP_INTROS.get("plate_view")


def test_the_preferred_table_list_puts_cell_first():
    assert PREFERRED_TABLES[0] == "cell"
    assert COLOUR_SCALES[0][1] == "allq"      # robust by default
    assert DEFAULT_CMAP == "viridis"


# ---------------------------------------------------------------------------
# Opening a database
# ---------------------------------------------------------------------------

def test_opening_a_database_lists_tables_and_numeric_columns(screen, measdb):
    _src, db = measdb
    with pytest.MonkeyPatch.context():
        assert screen.open_database(db)
    assert screen.current_table() == "cell"           # preferred
    assert [screen._table_combo.itemText(i)
            for i in range(screen._table_combo.count())] == ["cell", "png_list"]
    columns = [screen._value_combo.itemText(i)
               for i in range(screen._value_combo.count())]
    assert columns == ["value"]                        # 'note' is text
    assert screen.current_value_column() == "value"
    assert screen.last_error == ""
    assert screen._btn_render.isEnabled()


def test_a_run_folder_resolves_to_its_measurements_db(screen, measdb):
    src, db = measdb
    assert screen.open_database(src)
    assert screen._path_edit.text() == os.path.abspath(db)


def test_a_bad_path_reports_inline_and_leaves_the_screen_usable(screen, tmp_path):
    assert not screen.open_database(str(tmp_path / "nope.db"))
    assert "nope.db" in screen.last_error
    assert screen.last_error == screen.status_text()
    assert not screen._grid.has_plate()

    assert not screen.open_database("")
    assert "No database selected" in screen.last_error


def test_a_table_with_no_numeric_columns_says_so(screen, measdb, tmp_path):
    _src, db = measdb
    screen.open_database(db)
    con = sqlite3.connect(str(tmp_path / "other.db"))
    con.execute("CREATE TABLE only_text (prc TEXT, label TEXT)")
    con.commit()
    con.close()
    assert screen.open_database(str(tmp_path / "other.db"))
    assert "no numeric columns" in screen.last_error


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_rendering_draws_the_plate_and_the_edge_report(rendered, frame):
    screen = rendered
    layout = screen._layout_df
    assert len(layout) == N_ROWS * N_COLS
    assert layout.attrs["plate_format"] == 96
    assert screen._grid.grid_size() == (N_ROWS, N_COLS)
    assert screen._grid.has_plate()

    report = screen._report
    assert report.edge_detected
    assert report.pct_difference == pytest.approx(35.0, abs=6.0)
    text = screen.report_text()
    assert "Edge effect" in text
    assert "Cliff's delta" in text
    assert "Ring profile" in text
    assert "96-well" in text
    assert screen.last_error == ""
    assert "Edge effect" in screen.status_text()
    assert screen._btn_export.isEnabled()

    # The drawn values are the per-well means of what is in the database.
    expected = frame.groupby("prc")["value"].mean()
    assert screen._grid.well_value(1, 1) == pytest.approx(
        expected["plate1_r1_c1"])
    assert screen._grid.well_count(1, 1) == N_OBJECTS


def test_the_plate_combo_is_filled_from_the_data(rendered):
    plates = [rendered._plate_combo.itemText(i)
              for i in range(rendered._plate_combo.count())]
    assert plates == ["plate1"]
    assert rendered._report.plate == "plate1"


def test_changing_an_option_recomputes_without_touching_the_database(
        rendered, monkeypatch):
    """Dragging the min-objects box must not re-query a huge table."""
    calls = []
    monkeypatch.setattr(pqc, "load_plate_frame",
                        lambda *a, **k: calls.append(a) or pd.DataFrame())
    rendered._min_count_box.setValue(3)
    assert calls == []
    assert rendered._report.n_dropped_min_count == len(THIN_WELLS)
    assert f"{len(THIN_WELLS)} dropped" in rendered.status_text()
    assert len(rendered._layout_df) == N_ROWS * N_COLS - len(THIN_WELLS)


def test_the_grouping_control_switches_the_aggregation(rendered, frame):
    rendered._grouping_combo.setCurrentText("count")
    layout = rendered._layout_df
    assert (layout["value"] == layout["n"]).all()
    rendered._grouping_combo.setCurrentText("median")
    expected = frame.groupby("prc")["value"].median()
    drawn = rendered._grid.well_value(4, 4)
    assert drawn == pytest.approx(expected["plate1_r4_c4"])


def test_the_colour_scale_control_changes_the_limits(rendered):
    robust = pqc.colour_limits(rendered._layout_df, "allq")
    rendered._scale_combo.setCurrentIndex(1)        # full range
    full = pqc.colour_limits(rendered._layout_df, "all")
    assert full[0] < robust[0] and full[1] > robust[1]
    assert f"{full[0]:.4g}" in rendered._scale_label.text()


def test_rendering_before_opening_anything_reports_inline(screen):
    assert not screen.render_plate()
    assert "Open a measurements database first" in screen.last_error


def test_an_invalid_measurement_column_reports_inline(screen, measdb):
    """Exactly what happens when a column choice outlives its table."""
    _src, db = measdb
    screen.open_database(db)
    screen.set_value_column("does_not_exist")
    assert not screen.render_plate()
    assert "does_not_exist" in screen.last_error
    assert screen.last_error == screen.status_text()
    assert not screen._grid.has_plate()

    # And the screen still works afterwards.
    screen.set_value_column("value")
    assert screen.render_plate()
    assert screen.last_error == ""


def test_a_table_with_no_well_identifier_reports_inline(screen, measdb, tmp_path):
    path = str(tmp_path / "noids.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE cell (value REAL)")
    con.executemany("INSERT INTO cell VALUES (?)", [(1.0,), (2.0,)])
    con.commit()
    con.close()
    assert screen.open_database(path)
    assert not screen.render_plate()
    assert "well identifier" in screen.last_error


def test_recompute_before_a_render_reports_inline(screen):
    assert not screen.recompute()
    assert "Nothing loaded yet" in screen.last_error


# ---------------------------------------------------------------------------
# Clicking a well
# ---------------------------------------------------------------------------

def test_clicking_a_well_reports_that_exact_well(rendered, frame, qtbot):
    grid = _sized_grid(rendered)
    with qtbot.waitSignal(grid.well_clicked, timeout=1000) as blocker:
        QTest.mouseClick(
            grid, Qt.LeftButton,
            pos=grid.cell_rect(3, 7).center().toPoint())
    assert blocker.args == [3, 7]

    text = rendered.well_info_text()
    expected = frame.groupby("prc")["value"].mean()["plate1_r3_c7"]
    assert text.startswith("C07 ")
    assert f"{N_OBJECTS} object(s)" in text
    assert f"{expected:.6g}" in text
    assert "interior" in text
    assert grid.selected_well() == (3, 7)


def test_clicking_an_edge_well_says_it_is_the_edge(rendered, qtbot):
    grid = _sized_grid(rendered)
    QTest.mouseClick(
        grid, Qt.LeftButton,
        pos=grid.cell_rect(1, 1).center().toPoint())
    assert rendered.well_info_text().startswith("A01 ")
    assert "outer ring (edge)" in rendered.well_info_text()


def test_clicking_a_dropped_well_says_blank_not_zero(rendered, qtbot):
    rendered._min_count_box.setValue(3)
    grid = _sized_grid(rendered)
    row, col = THIN_WELLS[0]
    QTest.mouseClick(
        grid, Qt.LeftButton,
        pos=grid.cell_rect(row, col).center().toPoint())
    text = rendered.well_info_text()
    assert text.startswith(pqc.well_id(row, col))
    assert "blank" in text
    assert "fewer than the 3 required" in text
    assert grid.well_value(row, col) is None


def test_every_well_of_the_grid_maps_back_to_itself(rendered):
    """A one-well-off mapping would mislabel every readout on the screen."""
    grid = _sized_grid(rendered)
    for r in range(1, N_ROWS + 1):
        for c in range(1, N_COLS + 1):
            assert grid.well_at(grid.cell_rect(r, c).center()) == (r, c)


def test_clicks_in_the_margins_hit_no_well(rendered, qtbot):
    grid = _sized_grid(rendered)
    assert grid.well_at(QPoint(2, 2)) is None                  # corner labels
    assert grid.well_at(QPoint(2, grid.height() // 2)) is None  # row letters
    assert grid.well_at(QPoint(grid.width() // 2, 2)) is None   # column numbers
    assert grid.well_at(QPoint(grid.width() - 1,
                               grid.height() - 1)) is None      # past the grid
    before = grid.selected_well()
    QTest.mouseClick(grid, Qt.LeftButton, pos=QPoint(2, 2))
    assert grid.selected_well() == before


def test_an_empty_grid_answers_nothing_rather_than_guessing(qtbot):
    grid = PlateGridWidget()
    qtbot.addWidget(grid)
    grid.resize(400, 300)
    assert not grid.has_plate()
    assert grid.well_at(QPoint(50, 50)) is None
    assert grid.grid_size() == (0, 0)
    assert grid.well_value(1, 1) is None
    assert grid.well_count(1, 1) == 0


def test_the_grid_paints_a_plate_and_an_empty_state(rendered, qtbot):
    """Exercise paintEvent both ways — a crash here is a blank screen."""
    from PySide6.QtGui import QPixmap
    grid = _sized_grid(rendered)
    grid.render(QPixmap(grid.size()))
    grid.select(2, 2)
    grid.render(QPixmap(grid.size()))
    grid.clear()
    grid.set_placeholder("nothing here")
    grid.render(QPixmap(grid.size()))
    assert not grid.has_plate()


def test_selecting_a_well_before_a_render_still_answers(screen):
    text = screen.select_well(4, 4)
    assert text.startswith("D04")
    assert "blank" in text


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_exporting_writes_the_well_grid(rendered, tmp_path, frame):
    out = str(tmp_path / "wells.csv")
    assert rendered.export_csv(out)
    assert os.path.isfile(out)

    back = pd.read_csv(out)
    assert list(back.columns) == list(pqc.LAYOUT_COLUMNS)
    assert len(back) == N_ROWS * N_COLS
    expected = frame.groupby("prc")["value"].mean()
    first = back[back["well"] == "A01"].iloc[0]
    assert first["value"] == pytest.approx(expected["plate1_r1_c1"])
    assert first["n"] == N_OBJECTS
    assert bool(first["is_edge"])
    assert int(back[back["well"] == "D06"].iloc[0]["ring"]) == 3
    assert f"→ {os.path.abspath(out)}" in rendered.status_text()


def test_exporting_respects_the_current_filters(rendered, tmp_path):
    rendered._min_count_box.setValue(3)
    out = str(tmp_path / "filtered.csv")
    assert rendered.export_csv(out)
    back = pd.read_csv(out)
    assert len(back) == N_ROWS * N_COLS - len(THIN_WELLS)
    assert "B03" not in set(back["well"])


def test_exporting_before_a_render_reports_inline(screen, tmp_path):
    assert not screen.export_csv(str(tmp_path / "nothing.csv"))
    assert "render a plate first" in screen.last_error
    assert not os.path.exists(str(tmp_path / "nothing.csv"))


def test_an_unwritable_export_path_reports_inline(rendered, tmp_path):
    assert not rendered.export_csv("")
    assert "Export failed" in rendered.last_error


# ---------------------------------------------------------------------------
# Read-only
# ---------------------------------------------------------------------------

def test_a_full_cycle_leaves_the_database_byte_identical(rendered, measdb,
                                                         tmp_path):
    _src, db = measdb
    before = hashlib.sha256(open(db, "rb").read()).hexdigest()
    rendered._min_count_box.setValue(3)
    rendered._grouping_combo.setCurrentText("median")
    rendered._scale_combo.setCurrentIndex(1)
    rendered.render_plate()
    rendered.select_well(1, 1)
    rendered.export_csv(str(tmp_path / "out.csv"))
    assert hashlib.sha256(open(db, "rb").read()).hexdigest() == before
    assert sorted(os.listdir(os.path.dirname(db))) == ["measurements.db"]


def test_only_the_needed_columns_are_read(rendered, monkeypatch, measdb):
    """A spaCR feature table is 500 columns wide; never SELECT * one."""
    _src, db = measdb
    seen = {}
    real = pqc.load_plate_frame

    def _spy(db_path, table, value_col, **kw):
        seen["args"] = (table, value_col)
        return real(db_path, table, value_col, **kw)

    monkeypatch.setattr(pqc, "load_plate_frame", _spy)
    rendered.set_table("cell")
    rendered.set_value_column("value")
    assert rendered.render_plate()
    assert seen["args"] == ("cell", "value")


# ---------------------------------------------------------------------------
# Threaded path
# ---------------------------------------------------------------------------

def test_the_threaded_path_renders_the_same_plate(qtbot, measdb):
    """The default path runs off the GUI thread; same result, no freeze."""
    screen = PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    with qtbot.waitSignal(screen.job_finished, timeout=15000):
        screen.open_database(measdb[1])
    with qtbot.waitSignal(screen.plate_rendered, timeout=15000):
        screen.render_plate()
    assert screen._report.edge_detected
    assert len(screen._layout_df) == N_ROWS * N_COLS
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=15000)
    assert not screen.is_busy()


# ---------------------------------------------------------------------------
# No modal dialogs, anywhere
# ---------------------------------------------------------------------------

def test_no_error_path_opens_a_dialog(screen, measdb, tmp_path):
    """Every failure the user can cause, in one pass.

    The autouse fixture turns any modal into an assertion failure, so
    reaching the end of this test is the assertion.
    """
    _src, db = measdb
    screen.open_database(str(tmp_path / "missing.db"))
    screen.render_plate()
    screen.recompute()
    screen.export_csv(str(tmp_path / "x.csv"))
    screen.open_database(db)
    screen.set_value_column("ghost")
    screen.render_plate()
    screen.set_value_column("value")
    screen.render_plate()
    screen._min_count_box.setValue(10_000)
    screen.export_csv(str(tmp_path / "y.csv"))
    assert screen.last_error


def test_dropping_every_well_explains_itself(rendered):
    rendered._min_count_box.setValue(10_000)
    assert len(rendered._layout_df) == 0
    assert not rendered._report.ok
    assert not rendered._grid.has_plate()
    assert rendered._btn_export.isEnabled() is False
    assert not math.isnan(rendered._report.n_wells)
    assert "0 well(s) drawn" in rendered.status_text()
