"""Redraws in the widgets_a group rewrite their items, never delete them."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QColor, QImage

pytestmark = pytest.mark.qt


@pytest.fixture
def accessible():
    was = QAccessible.isActive()
    QAccessible.setActive(True)
    try:
        yield
    finally:
        QAccessible.setActive(was)


def _walk(view):
    iface = QAccessible.queryAccessibleInterface(view)
    if iface is None:
        return
    for i in range(iface.childCount()):
        iface.child(i).text(QAccessible.Name)


class Seen:
    def __init__(self, view):
        self.view = view
        self.items = []

    def _current(self):
        view = self.view
        if hasattr(view, "topLevelItemCount"):
            out = []

            def visit(item):
                out.append(item)
                for k in range(item.childCount()):
                    visit(item.child(k))

            for i in range(view.topLevelItemCount()):
                visit(view.topLevelItem(i))
            return out
        if hasattr(view, "rowCount"):
            return [view.item(r, c) for r in range(view.rowCount())
                    for c in range(view.columnCount()) if view.item(r, c) is not None]
        return [view.item(i) for i in range(view.count())]

    def take(self):
        _walk(self.view)
        for item in self._current():
            if not any(item is seen for seen in self.items):
                self.items.append(item)

    def valid(self):
        return all(shiboken6.isValid(item) for item in self.items)


def test_calibration_preview_table(qtbot, accessible):
    from spacr.qt.widgets import calibration_preview as cp

    dialog = cp._CalibrationPreview(lambda: {}, threaded=False)
    qtbot.addWidget(dialog)
    seen = Seen(dialog.table)
    for _ in range(3):
        for n in (2, 1, 2):
            cp._retire_table_cells(dialog.table)
            for r in range(n):
                dialog.table.insertRow(r)
                for c in range(7):
                    dialog.table.setItem(r, c, cp._spare_table_item(dialog.table, f"v{r}{c}"))
            seen.take()
            assert seen.valid()
        dialog._clear()
        assert dialog.table.rowCount() == 0
        seen.take()
        assert seen.valid()
    dialog._cancel()
    assert dialog.table.rowCount() == 0 and seen.valid()


def test_class_editor_table(qtbot, accessible):
    from spacr.qt.widgets.class_editor import ClassEditorWidget

    editor = ClassEditorWidget(frame=pd.DataFrame({"c": ["a", "b", "a"]}))
    qtbot.addWidget(editor)
    seen = Seen(editor.table)
    editor.set_value({"a": {"column": "c", "value": "a"},
                      "b": {"column": "c", "value": "b"}})
    seen.take()
    editor.add_random_complement()
    seen.take()
    assert editor.table.topLevelItemCount() == 3
    editor.table.setCurrentItem(editor.table.topLevelItem(1))
    editor.remove_selected()
    seen.take()
    assert [editor.table.topLevelItem(i).text(0)
            for i in range(editor.table.topLevelItemCount())] == ["a", "rest"]
    assert editor.table.topLevelItem(1).text(1) == "the rest, at random"
    assert editor.table.topLevelItem(0).flags() & Qt.ItemIsEditable
    editor.remove_at(0)
    editor.set_value({"b": {"column": "c", "value": "b"}})
    seen.take()
    assert editor.table.topLevelItemCount() == 1
    editor.table.topLevelItem(0).setText(0, "bee")
    assert editor.rules()[0].name == "bee"
    assert seen.valid()


def _db(tmp_path):
    path = tmp_path / "m.db"
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE one (a TEXT, b REAL, c INTEGER)")
    con.execute("CREATE TABLE two (x TEXT, y REAL)")
    con.execute("INSERT INTO one VALUES ('p', 1.0, 1)")
    con.execute("INSERT INTO two VALUES ('p', 1.0)")
    con.commit()
    con.close()
    return str(path)


def test_column_picker_tree(qtbot, accessible, tmp_path):
    from spacr.qt.widgets.column_picker import ColumnPickerDialog

    dialog = ColumnPickerDialog(db_path=_db(tmp_path))
    qtbot.addWidget(dialog)
    tree = dialog._column_tree
    seen = Seen(tree)
    seen.take()
    assert tree.topLevelItemCount() == 3
    tree.setCurrentItem(tree.topLevelItem(1))
    dialog._filter.setText("b")
    assert tree.topLevelItem(0).isHidden()
    for name in ("two", "one", "two", "one"):
        dialog._tables.setCurrentItem(dialog._tables.findItems(name, Qt.MatchExactly)[0])
        seen.take()
        assert seen.valid()
    assert tree.topLevelItemCount() == 3
    assert [tree.topLevelItem(i).text(0) for i in range(3)] == ["a", "b", "c"]
    assert tree.topLevelItem(0).isHidden() and not tree.topLevelItem(1).isHidden()
    assert tree.currentItem() is None
    assert not dialog._count_btn.isEnabled()
    dialog.select_column("c")
    assert tree.currentItem().text(0) == "c"
    assert seen.valid()


def test_condition_dialog_lists(qtbot, accessible):
    from spacr.condition_annotations import source_context
    from spacr.qt.widgets.condition_annotation_dialog import ConditionAnnotationDialog

    frame = pd.DataFrame({"wellID": [f"A{i}" for i in range(260)], "v": range(260)})
    dialog = ConditionAnnotationDialog(frame, source_context(), threaded=False)
    qtbot.addWidget(dialog)
    box = dialog.boxes[0]
    seen = Seen(box.rows)
    tokens = list(dialog.source_model.locations)
    box.manual_rows = tokens[:230]
    box._show_manual_rows()
    seen.take()
    assert box.rows.count() == 201
    assert box.rows.item(200).flags() == Qt.NoItemFlags
    box.manual_rows = tokens[:3]
    box._show_manual_rows()
    seen.take()
    assert box.rows.count() == 3
    assert all(box.rows.item(i).flags() != Qt.NoItemFlags for i in range(3))
    assert [box.rows.item(i).data(Qt.UserRole) for i in range(3)] == tokens[:3]
    box.rows.item(1).setSelected(True)
    box.remove_selected_rows()
    assert box.rows.count() == 2
    box.clear_manual_rows()
    assert box.rows.count() == 0
    box.manual_rows = tokens[:2]
    box._show_manual_rows()
    assert box.rows.count() == 2 and seen.valid()

    views = [Seen(dialog.template_parts), Seen(dialog.template_palette),
             Seen(dialog.combine_inputs)]
    dialog.combine_available.setCurrentIndex(0)
    dialog._append_combine_input()
    dialog.combine_available.setCurrentIndex(1)
    dialog._append_combine_input()
    dialog.template_parts._append_part({"kind": "column", "column": "v"})
    dialog.template_parts._append_part({"kind": "text", "text": "-"})
    for view in views:
        view.take()
    dialog._columns[0]["columns"] = [dialog.combine_inputs.item(i).text() for i in range(2)]
    dialog._columns[0]["parts"] = [{"kind": "column", "column": "v"},
                                   {"kind": "text", "text": "-"}]
    for _ in range(3):
        dialog._load_column()
        for view in views:
            view.take()
            assert view.valid()
    assert dialog.combine_inputs.count() == 2
    assert dialog.template_palette.count() == dialog.combine_available.count()
    assert dialog.template_parts._parts() == [{"kind": "column", "column": "v"},
                                              {"kind": "text", "text": "-"}]
    assert dialog.template_parts.item(1).flags() & Qt.ItemIsEditable
    assert not dialog.template_parts.item(0).flags() & Qt.ItemIsEditable
    dialog.template_parts.setCurrentRow(0)
    dialog._remove_template_part()
    dialog.combine_inputs.setCurrentRow(0)
    dialog._remove_combine_input()
    assert dialog.template_parts.count() == 1 and dialog.combine_inputs.count() == 1
    dialog._load_column()
    for view in views:
        view.take()
        assert view.valid()


def test_figure_queue_list(qtbot, accessible, tmp_path):
    from matplotlib.figure import Figure
    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    queue.set_live_canvas_enabled(False)
    seen = Seen(queue._list)

    def add(name):
        image = QImage(24, 16, QImage.Format.Format_RGB32)
        image.fill(QColor("red"))
        path = tmp_path / f"{name}.png"
        assert image.save(str(path))
        queue.add_figure(Figure(label=name), str(path))

    for round_ in range(3):
        for name in ("a", "b", "c"):
            add(f"{round_}{name}")
            seen.take()
        assert queue._list.count() == 3
        assert queue._list.item(2).text() == "#3"
        assert queue._list.item(0).textAlignment() == Qt.AlignCenter
        queue.clear()
        assert queue._list.count() == 0 and queue.count() == 0
        seen.take()
        assert seen.valid()
    add("x")
    add("y")
    state = queue.workspace_state()
    queue.clear()
    assert queue._list.count() == 0
    assert queue.apply_workspace_state(state)
    seen.take()
    assert queue._list.count() == 2 and queue.count() == 2
    assert [queue._list.item(i).text() for i in range(2)] == ["#1", "#2"]
    assert seen.valid()


def test_file_list_widgets(qtbot, accessible, tmp_path):
    from spacr.qt.widgets import file_list as fl

    files = []
    for i in range(4):
        path = tmp_path / f"p{i}.csv"
        path.write_text("a\n1\n")
        files.append(str(path))
    listing = fl.FilePathListWidget(kind="csv")
    qtbot.addWidget(listing)
    seen = Seen(listing._list)
    for _ in range(3):
        listing.set_value(files)
        seen.take()
        assert listing.paths() == files
        listing._list.item(1).setSelected(True)
        listing.remove_selected()
        seen.take()
        assert listing.paths() == [files[0]] + files[2:]
        listing._list.setCurrentRow(0)
        listing._list.item(0).setSelected(True)
        listing._move_selected(1)
        listing.set_value(files[:2])
        assert listing.paths() == files[:2]
        listing.clear()
        assert listing._list.count() == 0
        seen.take()
        assert seen.valid()

    single = fl.FilePathListWidget(kind="csv", single=True)
    qtbot.addWidget(single)
    sseen = Seen(single._list)
    for path in files:
        single.add_paths([path])
        sseen.take()
        assert single.paths() == [path]
    assert sseen.valid()

    paired = fl.PairedFileTableWidget()
    qtbot.addWidget(paired)
    pseen = Seen(paired.table)
    rows = [{"plate": f"plate {i}", "score": files[i], "count": files[(i + 1) % 4]}
            for i in range(3)]
    for _ in range(3):
        paired.set_value(rows)
        pseen.take()
        assert paired.table.rowCount() == 3
        paired.table.selectRow(0)
        paired._move(1)
        pseen.take()
        paired.table.selectRow(0)
        paired._remove()
        pseen.take()
        assert paired.table.rowCount() == 2
        paired.set_value(rows[:1])
        assert paired.table.rowCount() == 1
        paired.set_value([])
        assert paired.table.rowCount() == 0
        assert pseen.valid()


def test_formula_editor_list(qtbot, accessible):
    from spacr.qt.widgets.formula import ColumnFormula
    from spacr.qt.widgets.formula_editor import FormulaPanel

    panel = FormulaPanel()
    qtbot.addWidget(panel)
    panel.set_frame(pd.DataFrame({"area": [1.0, 2.0, 3.0]}))
    seen = Seen(panel._list)
    for _ in range(3):
        assert panel.add_formula(ColumnFormula("half", "area / 2"))
        assert panel.add_formula(ColumnFormula("dbl", "area * 2"))
        seen.take()
        assert panel._list.count() == 2
        assert panel._list.item(0).data(Qt.UserRole) == "half"
        panel._list.setCurrentRow(0)
        panel.remove_selected()
        seen.take()
        assert panel._list.count() == 1
        assert panel._list.item(0).data(Qt.UserRole) == "dbl"
        panel.clear()
        assert panel._list.count() == 0
        panel.set_formulas([ColumnFormula("half", "area / 2")])
        assert panel._list.count() == 1
        panel.clear()
        seen.take()
        assert seen.valid()


def test_gate_tree(qtbot, accessible):
    from spacr.qt.widgets.gate_editor import GateTree
    from spacr.qt.widgets.gate_spec import GateSet, RectGate

    def gate(name, parent=None):
        return RectGate(name=name, parent=parent, x_column="a", y_column="b",
                        x_low=0.0, x_high=2.5, y_low=0.0, y_high=2.5)

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    tree = GateTree()
    qtbot.addWidget(tree)
    seen = Seen(tree.tree)
    big = GateSet()
    big.add(gate("one"))
    big.add(gate("two", "one"))
    big.add(gate("three", "two"))
    big.add(gate("four"))
    small = GateSet()
    small.add(gate("solo"))
    for _ in range(3):
        tree.set_colour_source(lambda name: "#ff0000")
        tree.set_gates(big, frame)
        seen.take()
        assert tree.tree.topLevelItemCount() == 2
        top = tree.tree.topLevelItem(0)
        assert top.text(0) == "one" and top.child(0).text(0) == "two"
        assert top.child(0).child(0).text(0) == "three"
        assert top.foreground(0).color().name() == "#ff0000"
        assert top.checkState(0) == Qt.Checked
        tree.select("three")
        assert tree.active_gate() == "three"
        tree.set_gates(big, frame)
        assert tree.active_gate() == "three"
        top.setCheckState(0, Qt.Unchecked)
        assert not tree.is_enabled("one")
        top.setCheckState(0, Qt.Checked)
        assert tree.is_enabled("one")
        tree.set_gates(small, frame)
        seen.take()
        assert tree.tree.topLevelItemCount() == 1
        assert tree.tree.topLevelItem(0).childCount() == 0
        assert tree.tree.topLevelItem(0).text(0) == "solo"
        tree.set_gates(small, None)
        assert tree.tree.topLevelItemCount() == 0
        seen.take()
        assert seen.valid()
