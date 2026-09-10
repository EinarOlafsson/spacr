"""Sweep the source for view classes; every one of them must ask for sorting.

Found by walking the syntax tree for ``QTableWidget``, ``QTableView``,
``QTreeWidget`` and ``QTreeView`` rather than from a list somebody typed --
a list stops covering the table added tomorrow, which is exactly how thirty
of them ended up unable to sort at all.
"""
from __future__ import annotations

import ast
import os

import pytest

import spacr

VIEW_CLASSES = {"QTableWidget", "QTableView", "QTreeWidget", "QTreeView"}

QT_ROOT = os.path.join(os.path.dirname(os.path.abspath(spacr.__file__)), "qt")

#: The one view that must NOT take the shared behaviour, and why.
#:
#: The database browser sorts in SQL over the whole table. Qt's model sort
#: would reorder the rows fetched so far and present that as the table's
#: order, which on a 400k-row measurement table is a different answer
#: wearing the same sort indicator. Its header still follows the contract --
#: see ``test_the_database_browser_sorts_descending_first``.
EXEMPT = {("screens/db_browser.py", "self._view")}


def _relative(path: str) -> str:
    return os.path.relpath(path, QT_ROOT).replace(os.sep, "/")


def _python_files():
    for dirpath, _dirs, files in os.walk(QT_ROOT):
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _text(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - unparse handles every node here
        return ""


def _installed_names(tree) -> set:
    """Every expression handed to ``install_sorting`` in this module."""
    out = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "install_sorting"
                and node.args):
            out.add(_text(node.args[0]))
    return out


def _loop_covered(tree) -> set:
    """Views covered by a ``for view in (a, b, c): install_sorting(view)``."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target,
                                                           ast.Name):
            continue
        variable = node.target.id
        body_installs = {
            _text(call.args[0])
            for call in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "install_sorting" and call.args)}
        if variable not in body_installs:
            continue
        if isinstance(node.iter, (ast.Tuple, ast.List, ast.Set)):
            out.update(_text(element) for element in node.iter.elts)
    return out


def _constructions(tree):
    """``(target, class)`` for every view built by an assignment."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value,
                                                              ast.Call):
            continue
        func = node.value.func
        name = (func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else "")
        if name not in VIEW_CLASSES:
            continue
        for target in node.targets:
            yield _text(target), name


def test_every_view_in_the_source_asks_for_the_sorting_contract():
    """Not one table left building its own behaviour, or none at all."""
    swept = 0
    missing = []
    for path in _python_files():
        relative = _relative(path)
        if relative == "widgets/sortable_table.py":
            continue
        tree = ast.parse(open(path, encoding="utf-8").read())
        covered = _installed_names(tree) | _loop_covered(tree)
        for target, _class in _constructions(tree):
            swept += 1
            if (relative, target) in EXEMPT or target in covered:
                continue
            missing.append(f"{relative}: {target}")
    assert swept >= 45, f"the sweep found only {swept} views; it is broken"
    assert not missing, (
        "these views are built without install_sorting():\n  "
        + "\n  ".join(missing))


def test_every_view_subclass_asks_for_it_too():
    """A view that is subclassed rather than built still has to sort."""
    missing = []
    for path in _python_files():
        relative = _relative(path)
        if relative == "widgets/sortable_table.py":
            continue
        tree = ast.parse(open(path, encoding="utf-8").read())
        covered = _installed_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {_text(base) for base in node.bases}
            if not bases & VIEW_CLASSES:
                continue
            if "self" not in covered:
                missing.append(f"{relative}: class {node.name}")
    assert not missing, (
        "these view subclasses never call install_sorting(self):\n  "
        + "\n  ".join(missing))


def test_nothing_builds_a_bare_table_cell_any_more():
    """``QTableWidgetItem(...)`` sorts "10" before "9". Nothing may build one.

    The shared item is the only way a cell reaches a table, so a column of
    numbers cannot quietly go back to sorting as words.
    """
    offenders = []
    for path in _python_files():
        relative = _relative(path)
        if relative == "widgets/sortable_table.py":
            continue
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else "")
            if name in ("QTableWidgetItem", "QTreeWidgetItem"):
                offenders.append(f"{relative}:{node.lineno}")
    assert not offenders, (
        "these build a plain Qt cell instead of the shared one:\n  "
        + "\n  ".join(offenders))
