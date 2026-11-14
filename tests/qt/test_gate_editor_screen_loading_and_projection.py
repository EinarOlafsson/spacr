"""The Gate Editor screen: getting tables on screen, projecting them, and its plot menu.

Every step can fail for a reason the user can act on -- a file that is not a
database, two plates of the same name, a table the first database does not
list, a projection over one column -- and each failure has to arrive as a
sentence naming the thing that went wrong.  The successes are pinned as
exactly: merged row counts, source labels, chip names, component columns on
the axis pickers, and the variance and missingness notes.

Jobs run unthreaded, so a load finishes before it returns.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint                               # noqa: E402

import spacr.merge_tables as merge                              # noqa: E402
import spacr.multi_database as multi                            # noqa: E402
import spacr.qt.screens.gate_editor as module                   # noqa: E402
from spacr.multi_database import SOURCE_COLUMN                  # noqa: E402
from spacr.qt.screens.gate_editor import (                      # noqa: E402
    SIDE_TABS_NAME, GateEditorScreen,
)
from spacr.qt.widgets.table_chip import TableChip               # noqa: E402

pytestmark = pytest.mark.qt


def _objects(plate="p1", n=8):
    return pd.DataFrame({
        "plateID": [plate] * n,
        "rowID": ["A"] * n,
        "columnID": ["1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "area": np.linspace(10.0, 80.0, n),
        "intensity": np.linspace(100.0, 800.0, n),
    })


def _database(path, plate="p1", n=8, nucleus=False):
    with sqlite3.connect(str(path)) as db:
        _objects(plate, n).to_sql("cell", db, index=False)
        if nucleus:
            _objects(plate, n).rename(columns={"area": "nucleus_area"}).to_sql(
                "nucleus", db, index=False)
    return str(path)


def _chip_names(layout):
    return [layout.itemAt(i).widget().name for i in range(layout.count())
            if isinstance(layout.itemAt(i).widget(), TableChip)]


def _wide(n=60, seed=1):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"m{i}": rng.normal(i, 1.0, n) for i in range(6)})
    frame["plateID"] = "p1"
    return frame


def _lopsided(n=200, seed=0):
    """One morphology measurement against twenty-four intensity ones."""
    rng = np.random.default_rng(seed)
    columns = {"cell_area": rng.normal(100.0, 10.0, n)}
    for i in range(1, 25):
        columns[f"cell_channel_{i}_mean_intensity"] = rng.normal(
            50.0 * i, 5.0, n)
    return pd.DataFrame(columns)


def _half_measured(n=240):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"cell_area": rng.normal(0.0, 1.0, n)})
    uninfected = np.zeros(n, dtype=bool)
    uninfected[: n // 2] = True
    for i in range(6):
        values = rng.normal(0.0, 50.0, n)
        values[uninfected] = np.nan
        frame[f"pathogen_channel_{i}_mean_intensity"] = values
    return frame


@pytest.fixture
def make(qtbot):
    """Build a fresh unthreaded screen."""
    def _build():
        widget = GateEditorScreen(threaded=False)
        qtbot.addWidget(widget)
        return widget
    return _build


def test_merging_databases_pools_distinct_plates_and_names_every_refusal(make, tmp_path, monkeypatch, subtests):
    """No files does nothing; unreadable, empty, clashing and failing merges say why; distinct plates pool into one frame."""
    with subtests.test(msg="no files at all"):
        screen = make()
        screen.load_paths([])
        assert screen._frame is None
        assert screen._path is None

    with subtests.test(msg="a file whose tables cannot be listed"):
        screen = make()
        bad = tmp_path / "notes.db"
        bad.write_text("this is not sqlite", encoding="utf-8")
        screen.load_paths([bad])
        assert screen._source.text().startswith(f"could not read {bad}: ")
        assert screen._frame is None

    with subtests.test(msg="a database with no tables"):
        screen = make()
        monkeypatch.setattr(module, "table_names", lambda _p: [])
        screen.load_paths([str(tmp_path / "empty.db")])
        assert screen._source.text() == "no table to merge in the chosen files"
        monkeypatch.undo()

    with subtests.test(msg="two plates of the same name are refused"):
        screen = make()
        recorded = []
        monkeypatch.setattr(multi, "record_decision", recorded.append)
        paths = [_database(tmp_path / "a.db"), _database(tmp_path / "b.db")]
        screen.load_paths(paths)
        monkeypatch.undo()
        assert screen._frame is None
        assert "p1" in screen._source.text()
        assert [d.outcome for d in recorded] == ["refused"]
        assert recorded[0].resolution == screen._source.text()

    with subtests.test(msg="any other merge failure"):
        screen = make()
        monkeypatch.setattr(module, "table_names", lambda _p: ["cell"])
        monkeypatch.setattr(multi, "describe_merge",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("the disk went away")))
        screen.load_paths([str(tmp_path / "a.db"), str(tmp_path / "b.db")])
        monkeypatch.undo()
        assert screen._source.text() == (
            "could not merge 2 files: the disk went away")
        assert screen._frame is None

    run_a = _database(tmp_path / "runA.db", "plate1")
    run_b = _database(tmp_path / "runB.db", "plate2")
    with subtests.test(msg="distinct plates merge into one frame"):
        screen = make()
        recorded = []
        monkeypatch.setattr(multi, "record_decision", recorded.append)
        screen.load_paths([run_a, run_b])
        monkeypatch.undo()
        assert len(screen._frame) == 16
        assert set(screen._frame["plateID"]) == {"plate1", "plate2"}
        assert sorted(screen._frame[SOURCE_COLUMN].unique()) == [
            "runA", "runB"]
        columns = len(screen._frame.columns)
        assert screen._source.text().startswith(
            f"2 databases · cell · 16 rows × {columns} columns · colour by "
            f"{SOURCE_COLUMN}")
        assert screen._merge_plan is not None
        assert screen._paths == [run_a, run_b]
        assert screen._table == "cell" and screen._tables == ["cell"]
        assert _chip_names(screen._chips) == ["cell"]
        assert [d.outcome for d in recorded] == ["merged"]
        assert recorded[0].resolution == (
            "merged 2 databases into the Gate Editor")

        screen._frame = None
        screen._reload_working_set()
        assert len(screen._frame) == 16
        assert set(screen._frame["plateID"]) == {"plate1", "plate2"}
        assert screen._source.text().startswith("2 databases · cell · 16 rows")

    with subtests.test(msg="a table the first database does not list"):
        screen = make()
        screen.load_paths([run_a, run_b], table="cell")
        assert screen._table_picker.currentText() == "cell"
        monkeypatch.setattr(module, "table_names", lambda _path: ["nucleus"])
        screen.load_paths([run_a, run_b], table="cell")
        monkeypatch.undo()
        picker = screen._table_picker
        assert [picker.itemText(i) for i in range(picker.count())] == [
            "nucleus"]
        assert picker.currentText() == "nucleus"
        assert screen._table == "cell"
        assert _chip_names(screen._chips) == ["cell"]
        assert len(screen._frame) == 16


def test_loading_one_file_reads_it_or_names_the_file(make, tmp_path, subtests):
    """A CSV is read whole with no table picker; a database it cannot open is named by its file name."""
    with subtests.test(msg="a csv is read whole"):
        screen = make()
        path = tmp_path / "plate.csv"
        _objects().to_csv(path, index=False)
        screen.load_path(str(path))
        assert len(screen._frame) == 8
        assert list(screen._frame.columns) == list(_objects().columns)
        assert screen._table_picker.isHidden() is True
        assert screen._source.text() == "plate.csv · 8 rows × 7 columns"

    with subtests.test(msg="a database the loader cannot open"):
        screen = make()
        bad = tmp_path / "broken.db"
        bad.write_text("not sqlite either", encoding="utf-8")
        screen.load_path(str(bad))
        assert screen._source.text().startswith("could not read broken.db: ")
        assert str(tmp_path) not in screen._source.text()
        assert screen._frame is None

    with subtests.test(msg="a database table is read and labelled"):
        screen = make()
        path = _database(tmp_path / "one.db", nucleus=True)
        screen.load_path(path, table="nucleus")
        picker = screen._table_picker
        assert [picker.itemText(i) for i in range(picker.count())] == [
            "cell", "nucleus"]
        assert picker.currentText() == "nucleus"
        assert screen._tables == ["nucleus"]
        assert "nucleus_area" in screen._frame.columns
        assert screen._source.text() == "one.db · nucleus · 8 rows × 7 columns"


def test_picking_a_table_adds_it_to_the_working_set_once(make, tmp_path, subtests):
    """A second table is merged in beside the first; one already in the set changes nothing."""
    path = _database(tmp_path / "measurements.db", nucleus=True)

    with subtests.test(msg="a second table is merged in"):
        screen = make()
        screen.load_path(path, table="cell")
        assert screen._tables == ["cell"]
        screen._table_picker.setCurrentText("nucleus")
        screen._on_table_added(0)
        assert screen._tables == ["cell", "nucleus"]
        assert _chip_names(screen._chips) == ["cell", "nucleus"]
        assert "cell_area" in screen._frame.columns
        assert "area" not in screen._frame.columns
        assert len(screen._frame) == 8

    with subtests.test(msg="a table already in the set"):
        screen = make()
        screen.load_path(path, table="cell")
        before = screen._frame
        screen._table_picker.setCurrentText("cell")
        screen._on_table_added(0)
        assert screen._tables == ["cell"]
        assert screen._frame is before


def test_reading_the_working_set_thins_merges_evenly_and_reads_one_table_straight(tmp_path, subtests):
    """A cap or a fraction steps through a merge; one table keeps its own column names."""
    path = _database(tmp_path / "measurements.db", n=9, nucleus=True)
    whole = GateEditorScreen._read_working_set(
        path, ["cell", "nucleus"], 1.0, None, None)
    assert len(whole) == 9

    for case_id, fraction, cap, rows in [
            ("a cap of four steps by two", 1.0, 4, [0, 2, 4, 6]),
            ("a cap above the rows keeps all", 1.0, 50, list(range(9))),
            ("half steps by two", 0.5, None, [0, 2, 4, 6, 8]),
            ("a third steps by three", 0.34, None, [0, 3, 6])]:
        with subtests.test(msg=case_id):
            merged = GateEditorScreen._read_working_set(
                path, ["cell", "nucleus"], fraction, cap, None)
            assert merged.reset_index(drop=True).equals(
                whole.iloc[rows].reset_index(drop=True))
            assert list(merged.index) == list(range(len(rows)))

    with subtests.test(msg="one table is read straight"):
        frame = GateEditorScreen._read_working_set(
            path, ["cell"], 1.0, None, None)
        assert list(frame.columns) == list(_objects().columns)
        assert len(frame) == 9


def test_projecting_puts_components_on_the_axes_or_says_why_not(make, monkeypatch, subtests):
    """A projection needs two measurements; refusals land on the source line; components become axis columns with honest notes."""
    with subtests.test(msg="one measurement selected"):
        screen = make()
        screen.set_frame(_wide())
        screen.apply_settings(screen._settings.replaced(
            reduction_columns=("m0",)))
        message = ("The xD tab selects fewer than two measurements; a "
                   "projection needs two.")
        assert screen.reduce_to_components() == message
        assert screen._source.text() == message

    with subtests.test(msg="a reduction the data refuses"):
        screen = make()

        def _refuse(*_a, **_k):
            raise merge.ReductionError("t-SNE needs more rows than that")

        monkeypatch.setattr(merge, "reduce_dimensions", _refuse)
        screen.set_frame(_wide())
        assert screen.reduce_to_components() == (
            "t-SNE needs more rows than that")
        assert screen._source.text() == "t-SNE needs more rows than that"
        monkeypatch.undo()

    with subtests.test(msg="the components arrive as ordinary columns"):
        screen = make()
        screen.set_frame(_wide())
        assert screen.reduce_to_components() is None
        assert (screen._x.currentText(), screen._y.currentText(),
                screen._z.currentText()) == ("PC1", "PC2", "PC3")
        assert [c for c in screen._frame.columns if c.startswith("PC")] == [
            "PC1", "PC2", "PC3"]
        assert screen._source.text().startswith("projected onto PC1 ")
        assert "of the variance" not in screen._source.text()

    with subtests.test(msg="one component leaves the pickers alone"):
        screen = make()
        screen.set_frame(_wide())
        screen._x.setCurrentText("m0")
        screen._y.setCurrentText("m1")

        def _one_component(frame, columns, **_kwargs):
            out = pd.DataFrame({"PC1": np.arange(len(frame), dtype=float)},
                               index=frame.index)
            out.attrs["explained_variance"] = [1.0]
            return out

        monkeypatch.setattr(merge, "reduce_dimensions", _one_component)
        assert screen.reduce_to_components() is None
        monkeypatch.undo()
        assert "PC1" in screen._frame.columns
        assert (screen._x.currentText(), screen._y.currentText()) == (
            "m0", "m1")
        assert screen._z.currentText() != "PC1"
        assert screen._source.text() == "projected onto PC1 100%"

    with subtests.test(msg="a group carrying almost none of the variance"):
        screen = make()
        screen.set_frame(_lopsided())
        screen.apply_settings(screen._settings.replaced(
            reduction_groups={"family": ["morphology", "intensity"]}))
        assert screen.reduce_to_components() is None
        assert screen._source.text().endswith(
            "  ·  family:morphology carries 4% of the variance")

    with subtests.test(msg="a balanced split is not reported"):
        screen = make()
        frame = _lopsided()
        frame["cell_perimeter"] = frame["cell_area"] * 0.9
        keep = ["cell_area", "cell_perimeter",
                "cell_channel_1_mean_intensity",
                "cell_channel_2_mean_intensity"]
        screen.set_frame(frame[keep])
        screen.apply_settings(screen._settings.replaced(
            reduction_groups={"family": ["morphology", "intensity"]}))
        assert screen.reduce_to_components() is None
        assert "  ·  " not in screen._source.text()

    with subtests.test(msg="diagnostics that fail keep the projection"):
        screen = make()
        monkeypatch.setattr(merge, "group_variance_share",
                            lambda *_a, **_k: (_ for _ in ()).throw(
                                RuntimeError("no variance today")))
        monkeypatch.setattr(merge, "missingness_leak",
                            lambda *_a, **_k: (_ for _ in ()).throw(
                                RuntimeError("nor that")))
        screen.set_frame(_wide())
        screen.apply_settings(screen._settings.replaced(
            reduction_columns=("m0", "m1", "m2")))
        assert screen.reduce_to_components() is None
        monkeypatch.undo()
        assert "PC1" in screen._frame.columns
        assert "  ·  " not in screen._source.text()

    with subtests.test(msg="a split on whether something was measured"):
        screen = make()
        screen.set_frame(_half_measured())
        assert screen.reduce_to_components() is None
        said = screen._source.text()
        assert "  ·  the projection separates objects by whether " \
               "pathogen_channel_" in said
        assert said.endswith(
            " was measured (50% missing) — that is not a phenotype")


def test_the_plot_menu_greys_what_needs_a_graph_and_survives_a_missing_canvas(make, monkeypatch, subtests):
    """Greyed rows carry their reason; a canvas that cannot answer still offers the rest; no canvas is silent."""
    with subtests.test(msg="before anything is drawn"):
        screen = make()
        items = [(label, enabled, why)
                 for label, enabled, _cb, why in screen.graph_menu_items()]
        assert items[:7] == [
            ("Save graph…", False, "Draw a graph first."),
            ("Copy image to clipboard", False, "Draw a graph first."),
            (None, True, ""),
            ("Reset view", True, ""),
            ("Graph settings…", True, ""),
            (None, True, ""),
            ("Export gates to the database…", True, "")]

    with subtests.test(msg="a canvas that cannot say whether it drew"):
        screen = make()
        monkeypatch.setattr(screen.gates.canvas, "figure",
                            lambda: (_ for _ in ()).throw(
                                RuntimeError("no figure here")))
        items = {label: enabled
                 for label, enabled, _cb, _why in screen.graph_menu_items()
                 if label}
        assert items["Save graph…"] is False
        assert items["Copy image to clipboard"] is False
        assert items["Export gates to the database…"] is True
        assert items["Graph settings…"] is True

    with subtests.test(msg="no canvas at all"):
        screen = make()
        canvas, screen.gates.canvas = screen.gates.canvas, None
        try:
            assert screen.graph_menu_items() == []
            screen._install_graph_context_menu()
            screen._show_graph_menu(QPoint(0, 0))
            screen._copy_graph_to_clipboard()
        finally:
            screen.gates.canvas = canvas
        assert screen.console.log.toPlainText() == ""


def test_the_screen_builds_its_side_tabs_even_without_transparency(qtbot, monkeypatch, subtests):
    """The tab strip carries its style name and both tabs, whether or not the theme can make it transparent."""
    from PySide6.QtWidgets import QTabWidget

    from spacr.qt import theme

    real = theme.make_transparent
    refused = []

    def _only_the_tabs(widget):
        if isinstance(widget, QTabWidget):
            refused.append(widget.objectName())
            raise RuntimeError("no compositing here")
        return real(widget)

    for case_id, patch in [("a theme that can", False),
                           ("a theme that cannot", True)]:
        with subtests.test(msg=case_id):
            if patch:
                monkeypatch.setattr(theme, "make_transparent", _only_the_tabs)
            screen = GateEditorScreen(threaded=False)
            qtbot.addWidget(screen)
            assert screen.side_tabs.objectName() == SIDE_TABS_NAME
            assert [screen.side_tabs.tabText(i)
                    for i in range(screen.side_tabs.count())] == [
                "Filter", "Search"]
            assert refused == ([SIDE_TABS_NAME] if patch else [])
