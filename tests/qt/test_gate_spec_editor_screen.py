"""What the Gate Editor screen, its gate list and its panel do with gate_spec gates.

The canvas draws gates; this is everything afterwards -- writing them back to
the database as filter columns, turning the shown ones into an annotation,
saving a strategy or a filter set and loading it against another table,
saving and copying the picture, and the gate list / panel that select, count,
cluster and publish them.  None of these steps may be silent: each failure
arrives as a sentence naming what went wrong, never an empty label or a
traceback.  Jobs run unthreaded, which ``JobRunner`` supports explicitly.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture; every case builds its own screen or panel.
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

import spacr.qt.screens.gate_editor as screen_module  # noqa: E402
from spacr.derived_tables import default_definition  # noqa: E402
from spacr.qt.linked_selection import LinkedSelection  # noqa: E402
from spacr.qt.screens.gate_editor import GateEditorScreen  # noqa: E402
from spacr.qt.widgets.gate_editor import GateEditorPanel, GateTree  # noqa: E402
from spacr.qt.widgets.gate_spec import (  # noqa: E402
    BoxGate, CylinderGate, GateSet, RectGate, ThresholdGate,
)
from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec  # noqa: E402
from tests.qt.test_requested_gate_table_merges import db  # noqa: E402,F401

pytestmark = pytest.mark.qt


def _objects(n=8):
    return pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["A"] * n, "columnID": ["1"] * n,
        "fieldID": ["f1"] * n, "object_label": list(range(1, n + 1)),
        "area": np.linspace(10.0, 80.0, n),
        "intensity": np.linspace(100.0, 800.0, n),
    })


def _big():
    return RectGate(name="big", x_column="area", y_column="intensity",
                    x_low=30.0, x_high=90.0, y_low=0.0, y_high=1000.0)


def _filters_of(path):
    with sqlite3.connect(path) as con:
        return pd.read_sql("SELECT * FROM filters", con)


def _tables_of(path):
    with sqlite3.connect(path) as con:
        return {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}


class _Dialogs:
    """Every modal the screen can raise, recorded instead of shown."""

    def __init__(self, mp):
        self.shown = []
        self.item = (None, False)
        self.text = ("", False)
        self.save_path = ""
        self.open_path = ""
        owner = self

        class _Box:
            @staticmethod
            def information(_parent, title, body):
                owner.shown.append((title, body))

            @staticmethod
            def warning(_parent, title, body):
                owner.shown.append((title, body))

        class _Input:
            @staticmethod
            def getItem(*_a, **_k):
                return owner.item

            @staticmethod
            def getText(*_a, **_k):
                return owner.text

        import PySide6.QtWidgets as qtw
        mp.setattr(qtw, "QMessageBox", _Box)
        mp.setattr(qtw, "QInputDialog", _Input)
        mp.setattr(screen_module.QFileDialog, "getSaveFileName",
                   staticmethod(lambda *a, **k: (owner.save_path, "")))
        mp.setattr(screen_module.QFileDialog, "getOpenFileName",
                   staticmethod(lambda *a, **k: (owner.open_path, "")))

    @property
    def titles(self):
        return [title for title, _body in self.shown]


@pytest.fixture
def make_loaded(qtbot, tmp_path, monkeypatch):
    """A screen as it is after a database has been read and gated, plus dialogs."""
    counter = iter(range(1000))

    def build(mp):
        folder = tmp_path / f"case{next(counter)}"
        folder.mkdir()
        path = str(folder / "measurements.db")
        with sqlite3.connect(path) as con:
            _objects().to_sql("cell", con, index=False)
        screen = GateEditorScreen(threaded=False)
        qtbot.addWidget(screen)
        screen._path = path
        screen._table = "cell"
        screen.set_frame(_objects())
        screen.gates.set_gates(GateSet().add(_big()))
        return screen, _Dialogs(mp), folder
    return build


def _fresh(qtbot, frame=None):
    other = GateEditorScreen(threaded=False)
    qtbot.addWidget(other)
    other.set_frame(_objects() if frame is None else frame)
    return other


def test_gates_are_written_back_as_filter_and_annotation_columns(make_loaded, qtbot, monkeypatch, subtests):
    """export_gates / annotate_from_gates write 1/0 columns or refuse by name."""
    with subtests.test(msg="a_gate_becomes_a_filters_column"):
        with monkeypatch.context() as mp:
            screen, dialogs, _ = make_loaded(mp)
            screen.export_gates()
        filters = _filters_of(screen._path)
        assert filters["big"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
        assert "wrote big (6 objects)" in screen._source.text()
        assert dialogs.shown == []
    with subtests.test(msg="one_unappliable_gate_does_not_cost_the_others"):
        with monkeypatch.context() as mp:
            screen, _dialogs, _ = make_loaded(mp)
            screen.gates.set_gates(
                GateSet().add(ThresholdGate(name="bright", column="intensity", low=400.0))
                .add(ThresholdGate(name="ratio", column="area_over_intensity", low=0.5)))
            screen.export_gates()
        said = screen._source.text()
        assert "bright" in said and "could not export ratio" in said
        columns = set(_filters_of(screen._path).columns)
        assert "bright" in columns and "big" not in columns and "ratio" not in columns
    refusals = [
        ("export_from_a_csv", "export", dict(path="plate.csv"), ["Not a database"]),
        ("export_with_no_table", "export", dict(table=""), ["No table"]),
        ("annotate_with_nothing_shown", "annotate", dict(hide=True), ["No gates shown"]),
        ("annotate_without_a_table", "annotate", dict(fresh=True), ["No table"]),
        ("annotate_mode_refused", "annotate", dict(item=("sideways", True),
                                                   text=("phenotype", True)),
         ["Could not annotate"]),
        ("annotate_mode_cancelled", "annotate", dict(item=("binary", False)), []),
        ("annotate_blank_name", "annotate", dict(item=("binary", True),
                                                 text=("   ", True)), []),
    ]
    for case_id, action, how, titles in refusals:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                screen, dialogs, folder = make_loaded(mp)
                path = screen._path
                if how.get("fresh"):
                    screen = GateEditorScreen(threaded=False)
                    qtbot.addWidget(screen)
                    screen.gates.set_gates(GateSet().add(
                        ThresholdGate(name="bright", column="intensity", low=400.0)))
                if "path" in how:
                    screen._path = str(folder / how["path"])
                if "table" in how:
                    screen._table = how["table"]
                if how.get("hide"):
                    for name in screen.gates.canvas.enabled_gates:
                        screen.gates.canvas.set_gate_enabled(name, False)
                dialogs.item = how.get("item", dialogs.item)
                dialogs.text = how.get("text", dialogs.text)
                (screen.export_gates if action == "export" else screen.annotate_from_gates)()
            assert dialogs.titles == titles
            assert "filters" not in _tables_of(path)
            if case_id == "export_from_a_csv":
                assert "measurement database" in dialogs.shown[0][1]
    with subtests.test(msg="shown_gates_become_one_annotation_column"):
        with monkeypatch.context() as mp:
            screen, dialogs, _ = make_loaded(mp)
            dialogs.item, dialogs.text = ("binary", True), ("phenotype", True)
            screen.annotate_from_gates()
        assert "phenotype" in _filters_of(screen._path).columns
        assert "wrote phenotype" in screen._source.text()
    with subtests.test(msg="a_csv_still_gets_its_counts"):
        with monkeypatch.context() as mp:
            screen, dialogs, folder = make_loaded(mp)
            dialogs.item, dialogs.text = ("binary", True), ("phenotype", True)
            screen._path = str(folder / "plate.csv")
            screen.annotate_from_gates()
        said = screen._source.text()
        assert said.startswith("binary annotation — ")
        assert "not written: this table came from a file" in said


def _saved_merge(path, name="Derived gate data"):
    from spacr.derived_tables import execute, save_definition

    definition = default_definition(path, ["cell", "pathogen"], name=name)
    frame, _report = execute(path, definition)
    save_definition(path, frame.attrs.get("merge_definition", definition))
    return name


def test_gate_export_provenance_locks_and_gated_crop_paths(db, tmp_path, monkeypatch, subtests):  # noqa: F811
    """Provenance and lock bookkeeping refuse exactly; gated crops join like thresholds."""
    import spacr.derived_tables as dt

    with subtests.test(msg="derived_table_without_provenance_writes_no_gate"):
        name = _saved_merge(db)
        real = dt.execute

        def no_provenance(path, definition):
            frame, report = real(path, definition)
            frame.attrs["image_provenance"] = False
            return frame, report

        with monkeypatch.context() as mp:
            mp.setattr(dt, "execute", no_provenance)
            gates = GateSet([ThresholdGate(name="g", column="pathogen_area", low=1)])
            written, failed = GateEditorScreen._write_gates(db, name, gates)
        assert written == []
        assert [reason for _gate, reason in failed] == ["No verified image/object provenance"]
    with subtests.test(msg="provenance_failure_after_writing_is_reported"):
        with monkeypatch.context() as mp:
            mp.setattr(screen_module, "_record_gate_export_provenance",
                       lambda *a, **k: (_ for _ in ()).throw(OSError("sidecar locked")))
            mp.setattr(GateEditorScreen, "_write_gates",
                       staticmethod(lambda path, table, gates: ([("g", 1)], [])))
            written, failed, _verdicts = GateEditorScreen._write_gates_checked(
                db, "cell", GateSet([]), "")
        assert written == [("g", 1)]
        assert failed == [("export provenance", "sidecar locked")]
    with subtests.test(msg="unreadable_and_unrelated_locks_are_skipped"):
        from spacr.run_journal import _locks_root

        root = _locks_root()
        (root / "broken.json").write_text("{not json", encoding="utf-8")
        (root / "other.json").write_text(json.dumps({"gates": {"/x.json": {}}}), encoding="utf-8")
        strategy = tmp_path / "gates.json"
        strategy.write_text(json.dumps({"gates": []}), encoding="utf-8")
        assert screen_module._gate_export_lock_verdicts(str(strategy), GateSet([])) == []
        assert screen_module._gate_export_lock_verdicts("", GateSet([])) == []
    with subtests.test(msg="gated_join_that_already_carries_its_key"):
        import spacr.io as io
        from spacr.qt.annotate_engine import gate_paths

        path = tmp_path / "keyed.db"
        path.write_bytes(b"")
        joined = pd.DataFrame({"prcfo": ["o1", "o2", "o3"],
                               "cell_area": [50.0, 500.0, 900.0]})
        crops = pd.DataFrame({"prcfo": ["o1", "o2", "o3"],
                              "png_path": ["/c/o1.png", "/c/o2.png", "/c/o3.png"]}
                             ).set_index("prcfo")
        with monkeypatch.context() as mp:
            mp.setattr(io, "_read_and_join_tables", lambda *_a, **_k: joined.copy())
            mp.setattr(io, "_read_db", lambda *_a, **_k: [crops.copy()])
            gate = ThresholdGate(name="big cells", column="cell_area", low=100.0)
            assert gate_paths(str(path), [gate]) == ["/c/o2.png", "/c/o3.png"]


def test_strategies_and_filter_sets_round_trip_through_files(make_loaded, qtbot, monkeypatch, subtests):
    """save/load gates and filters, named misses, and dialogs obeyed only with a path."""
    with subtests.test(msg="gating_strategy_round_trip"):
        with monkeypatch.context() as mp:
            screen, _d, folder = make_loaded(mp)
        path = str(folder / "gates.json")
        assert screen.save_gates(path) == path
        assert screen._source.text() == "gates saved to gates.json"
        other = _fresh(qtbot)
        assert other.load_gates(path) is True
        assert [g.name for g in other.gates.gates.gates] == ["big"]
        assert other.gates.gates.get("big") == _big()
        assert other._source.text() == "1 gate(s) from gates.json"
    filter_cases = [
        ("filter_set_round_trip", _objects(), [], "filters loaded from filters.json"),
        ("filter_set_on_another_plate", _objects().drop(columns=["area"]), ["area"],
         "filters.json loaded; this table has no area"),
    ]
    for case_id, frame, missing, said in filter_cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                screen, _d, folder = make_loaded(mp)
            screen.filters.add_column("area")
            path = str(folder / "filters.json")
            assert screen.save_filters(path) == path
            assert screen._source.text() == "filters saved to filters.json"
            other = _fresh(qtbot, frame)
            assert other.load_filters(path) == missing
            assert other._source.text() == said
    with subtests.test(msg="gate_dialogs_obeyed_only_with_a_path"):
        with monkeypatch.context() as mp:
            screen, dialogs, folder = make_loaded(mp)
            screen._source.setText("untouched")
            screen.choose_save_gates()
            assert screen._source.text() == "untouched"
            dialogs.save_path = str(folder / "chosen.json")
            screen.choose_save_gates()
            assert os.path.isfile(dialogs.save_path)
            screen.gates.set_gates(GateSet())
            screen.choose_load_gates()
            assert len(screen.gates.gates) == 0
            dialogs.open_path = dialogs.save_path
            screen.choose_load_gates()
            assert [g.name for g in screen.gates.gates.gates] == ["big"]
    with subtests.test(msg="filter_dialogs_obeyed_only_with_a_path"):
        with monkeypatch.context() as mp:
            screen, dialogs, folder = make_loaded(mp)
            screen._source.setText("untouched")
            screen.choose_save_filters()
            assert screen._source.text() == "untouched"
            dialogs.save_path = str(folder / "f.json")
            screen.filters.add_column("area")
            screen.choose_save_filters()
            assert os.path.isfile(dialogs.save_path)
            screen._source.setText("untouched")
            screen.choose_load_filters()
            assert screen._source.text() == "untouched"
            dialogs.open_path = dialogs.save_path
            screen.choose_load_filters()
            assert screen._source.text() == "filters loaded from f.json"


class _Menu:
    """A QMenu that records instead of popping up (a real one hangs offscreen)."""

    built = []

    def __init__(self, _parent=None):
        self.rows = []
        self.at = None
        _Menu.built.append(self)

    def addSeparator(self):  # noqa: N802 - Qt name
        self.rows.append(None)

    def addAction(self, label):  # noqa: N802 - Qt name
        action = _Action(label)
        self.rows.append(action)
        return action

    def exec(self, point):
        self.at = point


class _Action:
    def __init__(self, label):
        self.label = label
        self.enabled = True
        self.calls = []

    def setEnabled(self, on):  # noqa: N802 - Qt name
        self.enabled = bool(on)

    def setToolTip(self, _text):  # noqa: N802 - Qt name
        pass

    @property
    def triggered(self):
        return self

    def connect(self, slot):
        self.calls.append(slot)


MENU = ["Save graph…", "Copy image to clipboard", None, "Reset view",
        "Graph settings…", None, "Export gates to the database…", "Annotate…",
        "Annotate conditions", "Save annotated table", "Export table",
        "Open selection in Annotate"]


def test_the_graph_is_saved_copied_and_driven_from_one_menu(make_loaded, monkeypatch, subtests):
    """save_graph, the plot menu, the clipboard, settings, sampling and axes."""
    import spacr.qt.preferences as preferences
    import spacr.qt.widgets.figure_queue as queue

    def log_of(screen):
        return screen.console.log.toPlainText()

    with subtests.test(msg="graph_written_through_the_print_restyler"):
        with monkeypatch.context() as mp:
            screen, _d, folder = make_loaded(mp)
            called = {}

            def render(figure, path, **kwargs):
                called.update(kwargs, path=path)
                open(path, "wb").close()
                return True

            mp.setattr(queue, "render_figure_to_png", render)
            target = str(folder / "graph.png")
            assert screen.save_graph(target) == target
        assert (called["path"], called["for_print"], called["write_pdf"]) == (target, True, False)
        assert "Saved the graph to" in log_of(screen)
    with subtests.test(msg="a_bare_path_gets_the_preferred_suffix"):
        with monkeypatch.context() as mp:
            screen, _d, folder = make_loaded(mp)
            mp.setattr(preferences, "get_figure_format", lambda: "pdf")
            written = []

            def render(figure, path, **_kwargs):
                written.append(path)
                open(path, "wb").close()
                open(os.path.splitext(path)[0] + ".pdf", "wb").close()
                return True

            mp.setattr(queue, "render_figure_to_png", render)
            assert screen.save_graph(str(folder / "graph")) == str(folder / "graph.pdf")
        assert written == [str(folder / "graph.png")]
    with subtests.test(msg="a_refused_render_claims_no_file"):
        with monkeypatch.context() as mp:
            screen, _d, folder = make_loaded(mp)
            mp.setattr(queue, "render_figure_to_png", lambda *_a, **_k: False)
            assert screen.save_graph(str(folder / "graph.png")) == ""
        assert "Could not save the graph." in log_of(screen)
    with subtests.test(msg="cancelled_graph_dialog"):
        with monkeypatch.context() as mp:
            screen, dialogs, _ = make_loaded(mp)
            assert screen.save_graph() == ""
    with subtests.test(msg="menu_items_call_the_screen_methods"):
        with monkeypatch.context() as mp:
            screen, _d, _ = make_loaded(mp)
        items = screen.graph_menu_items()
        assert [label for label, _on, _cb, _why in items] == MENU
        by_label = {label: cb for label, _on, cb, _why in items if label}
        assert by_label["Export gates to the database…"] == screen.export_gates
        assert by_label["Graph settings…"] == screen.open_settings
        assert by_label["Annotate…"] == screen._annotate_selection
        assert by_label["Annotate conditions"] == screen.annotate_from_gates
        assert by_label["Save annotated table"] == screen._save_selected_annotation
        assert by_label["Export table"] == screen._export_selected_annotation
        assert by_label["Open selection in Annotate"] == screen._open_selection_in_annotate
    with subtests.test(msg="menu_pops_up_at_the_right_click"):
        from PySide6.QtCore import QPoint
        import PySide6.QtWidgets as qtw
        from spacr.qt.widgets import figure_settings

        with monkeypatch.context() as mp:
            screen, _d, _ = make_loaded(mp)
            _Menu.built = []
            mp.setattr(qtw, "QMenu", _Menu)
            figures = []
            mp.setattr(figure_settings, "_add_figure_tools",
                       lambda _menu, figure, _parent, _on_change: figures.append(figure))
            screen._show_graph_menu(QPoint(4, 5))
        menu = _Menu.built[0]
        assert [row.label if row is not None else None for row in menu.rows] == MENU + [None]
        assert menu.at == screen.gates.canvas.mapToGlobal(QPoint(4, 5))
        assert figures == [screen.gates.canvas.figure()]
        actions = [row for row in menu.rows if row is not None]
        assert [row.label for row in actions if not row.enabled] == ["Save annotated table"]
        assert all(bool(row.calls) is row.enabled for row in actions)
    with subtests.test(msg="clipboard"):
        from PySide6.QtWidgets import QApplication

        with monkeypatch.context() as mp:
            screen, _d, _ = make_loaded(mp)
            QApplication.clipboard().clear()
            screen._copy_graph_to_clipboard()
            assert not QApplication.clipboard().pixmap().isNull()
            assert "Graph copied to the clipboard." in log_of(screen)
            mp.setattr(screen.gates.canvas, "grab", lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("no compositor")))
            screen._copy_graph_to_clipboard()
        assert "Could not copy the graph: no compositor" in log_of(screen)
    with subtests.test(msg="one_settings_window_and_one_mode"):
        with monkeypatch.context() as mp:
            screen, _d, _ = make_loaded(mp)
        screen.open_settings()
        first = screen._settings_dialog
        assert first.isVisible()
        screen.open_settings()
        assert screen._settings_dialog is first
        screen._on_mode_requested("3D")
        assert (screen._settings.gate_mode, first.settings().gate_mode) == ("3D", "3D")
        assert screen._z.isHidden() is False
        assert screen.gates.canvas._mode == "3D"
        first.close()
    with subtests.test(msg="display_sampling_reuses_the_loaded_table"):
        with monkeypatch.context() as mp:
            screen, _d, _ = make_loaded(mp)
            reloaded = []
            mp.setattr(screen, "load_path", lambda p, t=None: reloaded.append((p, t)))
            original = screen._frame.copy(deep=True)
            screen.apply_settings(screen._settings.replaced(colour_map="magma"))
            screen.apply_settings(screen._settings.replaced(sample_fraction=0.25))
            shown = screen.gates.canvas.population()
            assert len(shown) == 2
            pd.testing.assert_frame_equal(shown, original.loc[shown.index])
            first = shown.index.tolist()
            screen.apply_settings(screen._settings.replaced(sample_fraction=1., max_points=4))
            assert screen.gates.canvas.population().index.tolist()[:2] == first
            assert len(screen.gates.canvas.population()) == 4
            pd.testing.assert_frame_equal(screen._frame, original)
            assert reloaded == []
    with subtests.test(msg="merge_rules"):
        with monkeypatch.context() as mp:
            screen, dialogs, _ = make_loaded(mp)
            screen.show_aggregation_rules()
            assert dialogs.shown == []
            screen._rules_dialog.close()
            reloaded = []
            mp.setattr(screen, "_reload_working_set", lambda: reloaded.append(True))
            screen._tables = ["cell"]
            screen._on_aggregation_rules_changed({"area": "median"})
            assert reloaded == []
            assert screen._settings.merge_overrides == {"area": "median"}
            screen._tables = ["cell", "nucleus"]
            screen._on_aggregation_rules_changed({"area": "mean"})
            assert reloaded == [True]
    with subtests.test(msg="axes_requested_through_the_pickers"):
        with monkeypatch.context() as mp:
            screen, _d, _ = make_loaded(mp)
        screen._x.setCurrentText("object_label")
        screen._y.setCurrentText("object_label")
        screen._on_axes_requested("area", "intensity")
        assert (screen._x.currentText(), screen._y.currentText()) == ("area", "intensity")
        screen._on_axes_requested("perimeter", "")
        assert screen._x.currentText() == "area"


def _pair(name="one", parent=None, x="a", y="b", low=0.0, high=2.5):
    return RectGate(name=name, parent=parent, x_column=x, y_column=y,
                    x_low=low, x_high=high, y_low=low, y_high=high)


def test_the_gate_list_counts_and_edits_each_gate_row(qtbot, monkeypatch, subtests):
    """GateTree rows: a count, the unavailable marker or blank; refused bound edits."""
    def rows_of(tree):
        return {tree.tree.topLevelItem(i).text(0): [tree.tree.topLevelItem(i).text(c)
                                                     for c in (1, 2, 3)]
                for i in range(tree.tree.topLevelItemCount())}

    one = GateSet().add(RectGate(name="g", x_column="a", y_column="b",
                                 x_low=0.0, x_high=1.0, y_low=0.0, y_high=1.0))
    stats_cases = [
        ("neither_count_nor_reason_is_blank", ({}, {}), {"g": ["", "", ""]}),
        ("unavailable_says_so", ({}, {"g": "no column"}), {"g": [GateTree.UNAVAILABLE, "", ""]}),
    ]
    for case_id, stats, want in stats_cases:
        with subtests.test(msg=case_id):
            tree = GateTree()
            qtbot.addWidget(tree)
            with monkeypatch.context() as mp:
                mp.setattr(tree, "_gate_stats", lambda: stats)
                tree.set_gates(one, pd.DataFrame({"a": [0.5], "b": [0.5]}))
            assert rows_of(tree) == want
    with subtests.test(msg="working_set_without_the_measurement"):
        gates = (GateSet()
                 .add(RectGate(name="big_cell", x_column="cell_area", y_column="cell_intensity",
                               x_low=0.0, x_high=1e9, y_low=0.0, y_high=1e9))
                 .add(RectGate(name="bright_nucleus", x_column="nucleus_area",
                               y_column="nucleus_intensity", x_low=0.0, x_high=1e9,
                               y_low=0.0, y_high=1e9)))
        tree = GateTree()
        qtbot.addWidget(tree)
        tree.set_gates(gates, pd.DataFrame({"cell_area": [1.0, 2.0, 3.0],
                                            "cell_intensity": [10.0, 20.0, 30.0]}))
        rows = rows_of(tree)
        assert GateTree.UNAVAILABLE != ""
        assert rows["bright_nucleus"][0] == GateTree.UNAVAILABLE
        assert rows["big_cell"][0] == "3"
    with subtests.test(msg="a_bound_the_gate_can_no_longer_take_is_refused"):
        grid = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0],
                             "c": [1.0, 2.0, 3.0, 4.0]})
        tree = GateTree()
        qtbot.addWidget(tree)
        gates = GateSet()
        gates.add(CylinderGate(name="c", u_column="a", v_column="b", axis_column="c",
                               u_radius=1.0, v_radius=1.0))
        tree.set_gates(gates, grid)
        tree._rebuild_thresholds("c")
        assert set(tree._threshold_rows) == {"c"}
        announced = []
        tree.gates_changed.connect(lambda: announced.append(1))
        gates.add(ThresholdGate(name="c", column="a", low=0.0, high=1.0))
        tree._threshold_rows["c"][0].setText("2")
        tree._apply_threshold("c")
        assert gates.get("c") == ThresholdGate(name="c", column="a", low=0.0, high=1.0)
        assert announced == []


def test_the_gate_panel_selects_clusters_and_publishes_gates(qtbot, monkeypatch, subtests):
    """GateEditorPanel: axes follow the selected gate; publish rings or refuses."""
    grid = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0],
                         "c": [1.0, 2.0, 3.0, 4.0]})
    n = 6
    keyed = pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["A"] * n, "columnID": ["1"] * n,
        "fieldID": ["1"] * n, "object_label": [str(i + 1) for i in range(n)],
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    def make_panel(frame=grid, x="a", y="b", gates=None):
        panel = GateEditorPanel(link=LinkedSelection())
        qtbot.addWidget(panel)
        panel.set_frame(frame)
        roles = {c: CONTINUOUS for c in frame.columns
                 if pd.api.types.is_numeric_dtype(frame[c])}
        panel.set_spec(GraphSpec(x=x, y=y, roles=roles))
        if gates:
            gate_set = GateSet()
            for gate in gates:
                gate_set.add(gate)
            panel.set_gates(gate_set)
        return panel

    cluster_cases = [
        ("nothing_to_cluster", dict(frame=pd.DataFrame({"a": [], "b": []})), "Nothing to cluster"),
        ("needs_two_axes", dict(y=None), "Pick two measurements"),
    ]
    for case_id, how, title in cluster_cases:
        with subtests.test(msg=case_id):
            seen = []
            with monkeypatch.context() as mp:
                grab = staticmethod(lambda _p, t, text, *_a, **_k: seen.append(t))
                mp.setattr(QMessageBox, "information", grab)
                mp.setattr(QMessageBox, "warning", grab)
                panel = make_panel(**how)
                panel.run_cluster(ask=False)
            assert seen == [title]
            assert panel.gates.is_empty
    with subtests.test(msg="a_child_of_a_deleted_gate_is_refused"):
        panel = make_panel()
        panel.set_namer(lambda: "orphan")
        panel.canvas.gate_drawn.emit(_pair(name="(unnamed)", parent="deleted"))
        assert panel.gates.is_empty
        assert "deleted" in panel.status()
    with subtests.test(msg="selecting_a_gate_asks_for_its_axes_stale_rows_do_not"):
        gates = GateSet()
        gates.add(_pair(name="one"))
        gates.add(_pair(name="two", x="b", y="c"))
        panel = make_panel()
        panel.set_gates(gates)
        asked = []
        panel.axes_requested.connect(lambda x, y: asked.append((x, y)))
        panel.tree.select("one")
        assert asked == [("a", "b")]
        gates.remove("two")
        panel.tree.select("two")
        assert asked == [("a", "b")]
    publish_cases = [
        ("nothing_selected", grid, _pair(), "", None, "Select a gate"),
        ("no_table", grid, _pair(), "one", "drop", "Load a table first."),
        ("measurement_missing", grid, _pair(name="ghost", x="not_measured"), "ghost", None,
         "not_measured"),
        ("rings_its_objects", keyed, _pair(name="mid", low=1.5, high=4.5), "mid", None,
         "3 object(s) highlighted by mid"),
    ]
    for case_id, frame, gate, select, drop, said in publish_cases:
        with subtests.test(msg=case_id):
            panel = make_panel(frame, gates=[gate])
            panel.tree.select(select)
            if drop:
                panel._frame = None
            assert panel.publish() is None
            assert said in panel.status()
            if case_id == "rings_its_objects":
                link = panel.canvas.link
                assert panel.status() == said
                assert list(link.selection.keys) == ["p1_A_1_1_2", "p1_A_1_1_3", "p1_A_1_1_4"]
                assert link.filter is None or link.filter.is_empty
    with subtests.test(msg="the_framed_volume_becomes_a_box"):
        panel = make_panel()
        panel.canvas._mode = "3D"
        panel.canvas._z_column = "c"
        panel.canvas.render_now()
        panel.set_namer(lambda: "framed")
        panel.gate_from_view()
        gate = panel.gates.get("framed")
        assert isinstance(gate, BoxGate)
        assert gate.columns == ("a", "b", "c")
