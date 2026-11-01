"""Walking a layout from Python leaves no layout-item wrappers behind.

``QLayout.itemAt`` and ``QGridLayout.itemAtPosition`` run PySide's
ownership glue: the asked layout becomes the Python parent of each item
wrapper they return. A Qt-created ``QWidgetItem`` deleted later (its widget
moved or went) then stays alive and registered at the freed address, and the
next object allocated there is handed that wrapper -- an event filter was
given a ``QWidgetItem`` as ``watched``, and a strip's items read "already
deleted" while they lived. spaCR's layout walks read layouts through
``indexOf`` and QObject children instead (``spacr.qt.widgets.flow``).
"""
from __future__ import annotations

import ast
import gc
import pathlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget, QWidgetItem)

from spacr.qt.widgets import flow

pytestmark = pytest.mark.qt

ROOT = pathlib.Path(__file__).resolve().parents[2] / "spacr" / "qt"


def _item_wrappers():
    gc.collect()
    return sum(1 for obj in gc.get_objects() if isinstance(obj, QWidgetItem))


def _form(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)
    first, scroll = QLabel("first"), QScrollArea()
    column.addWidget(first)
    row = QHBoxLayout()
    column.addLayout(row)
    caption, field = QLabel("Name"), QLineEdit()
    row.addWidget(caption)
    row.addSpacing(4)
    row.addWidget(field)
    grid = QGridLayout()
    column.addLayout(grid)
    grid_caption, grid_field = QLabel("Channel"), QLineEdit()
    grid.addWidget(grid_caption, 0, 0)
    grid.addWidget(grid_field, 0, 1, 2, 1)
    column.addWidget(scroll, 1)
    column.addStretch(1)
    return host, column, row, grid, (first, caption, field, grid_caption, grid_field, scroll)


def test_the_shared_layout_helpers_answer_without_keeping_item_wrappers(qtbot):
    host, column, row, grid, (first, caption, field, grid_caption, grid_field, scroll) = _form(qtbot)
    before = _item_wrappers()

    assert [entry for _index, entry in flow._layout_entries(column)] == [first, row, grid, scroll]
    assert flow._layout_widgets(row) == [caption, field]
    assert flow._layout_widget_at(row, 0) is caption and flow._layout_widget_at(row, 1) is None
    assert flow._layout_holding(column, field) is row
    assert flow._plain_item_indexes(column) == [column.count() - 1]
    assert flow._plain_item_indexes(row) == [1]
    assert flow._grid_widget_at(grid, 0, 0) is grid_caption
    assert flow._grid_widget_at(grid, 1, 1) is grid_field

    assert _item_wrappers() == before


def test_the_module_walks_keep_no_item_wrappers(qtbot):
    from spacr.qt.dialogs import _already_scrolls
    from spacr.qt.screens.map_barcodes import _page_body
    from spacr.qt.screens.settings_model import _owning_layout
    from spacr.qt.widgets.column_picker import _find_layout_with
    from spacr.qt.widgets.console_panel import ConsolePanel

    host, column, row, _grid, (_first, _caption, field, _gc, _gf, scroll) = _form(qtbot)
    lone = QWidget()
    qtbot.addWidget(lone)
    QVBoxLayout(lone).addWidget(QScrollArea())
    console = ConsolePanel()
    qtbot.addWidget(console)
    for line in ("one", "two", "three"):
        console.append_warning(line)
    qtbot.waitUntil(lambda: "three" in console.as_text())
    before = _item_wrappers()

    assert _already_scrolls(lone.layout()) and not _already_scrolls(column)
    assert _page_body(host) is scroll
    assert _owning_layout(column, field) == (row, 2)
    assert _find_layout_with(column, field) is row
    text = console.as_text()
    assert "one" in text and "three" in text

    assert _item_wrappers() == before


def _single_argument_item_lookups():
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    classes.setdefault(id(child), node)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            name = node.func.attr
            if name == "itemAt" and len(node.args) == 1 or name == "itemAtPosition":
                argument = ast.unparse(node.args[0]).lower() if node.args else ""
                if "pos" in argument or "point" in argument:
                    continue
                owner = classes.get(id(node))
                bases = [ast.unparse(base) for base in owner.bases] if owner is not None else []
                if (isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                        and any("Layout" in base or "Strip" in base for base in bases)):
                    continue
                found.append(f"{path.relative_to(ROOT.parent.parent)}:{node.lineno}: {ast.unparse(node)}")
    return found


def test_no_layout_is_walked_with_item_at():
    assert _single_argument_item_lookups() == []
