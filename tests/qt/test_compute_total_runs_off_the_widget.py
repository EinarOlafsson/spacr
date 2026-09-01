"""``_compute_total`` runs on a worker thread and must not touch a widget.

Reported 2026-09-01::

    File "spacr/qt/screens/annotate.py", line 1141, in _compute_total
      "total": count_rows(s.db_path, s.image_type, table=self._settings.png_table)
    NameError: name 'self' is not defined

Its own docstring says why that could never work: it "is module-level rather
than a method so that it *cannot* reach a widget: everything it needs arrives
in ``s``". Two lines had reached for ``self._settings`` anyway, and because the
function runs inside a pipeline worker the failure surfaced as a dead job
rather than as an obvious crash.
"""
from __future__ import annotations

import ast
import pathlib
import sqlite3

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _own_scope_names(node):
    """Names in this function's OWN scope.

    Nested functions and classes are skipped deliberately: a factory that
    builds a class whose methods use ``self`` is correct, and counting those
    would make this check cry wolf on fourteen healthy call sites.
    """
    out = []

    def walk(n):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                continue
            if isinstance(child, ast.Name):
                out.append(child)
            walk(child)

    walk(node)
    return out


def test_no_module_level_function_in_the_package_reads_self():
    """The class of bug, swept over the whole package.

    A module-level function has no ``self`` to read, so every such reference is
    a NameError waiting for its branch to be taken -- which is exactly how this
    one survived: the failing line was in the branch with no filter set.
    """
    offenders = []
    for path in sorted((ROOT / "spacr").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {a.arg for a in node.args.args}
            names |= {a.arg for a in node.args.kwonlyargs}
            if "self" in names:
                continue
            for name in _own_scope_names(node):
                if name.id == "self" and isinstance(name.ctx, ast.Load):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{name.lineno} "
                        f"in {node.name}()")
                    break
    assert not offenders, (
        "module-level functions cannot read 'self':\n  "
        + "\n  ".join(offenders))


def test_compute_total_counts_without_a_widget(tmp_path):
    """The reported call path, driven directly.

    No filter is set, so this takes the branch the traceback came from.
    """
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import _compute_total

    db = tmp_path / "measurements.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE png_list (png_path TEXT)")
    conn.executemany("INSERT INTO png_list VALUES (?)",
                     [("plate1/data/cell/a.png",), ("plate1/data/cell/b.png",)])
    conn.commit()
    conn.close()

    settings = AnnotateSettings(db_path=str(db))
    result = _compute_total(settings, False)

    assert result["total"] == 2, result
    assert result["filtered_rows"] is None


def test_compute_total_honours_the_table_from_its_settings(tmp_path):
    """Not vacuous: the table name really comes from ``s``.

    If the argument were ignored, the count would come from whatever table
    happened to be named ``png_list`` and this would pass for the wrong reason.
    """
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import _compute_total

    db = tmp_path / "measurements.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE png_list (png_path TEXT)")
    conn.execute("INSERT INTO png_list VALUES ('a.png')")
    conn.execute("CREATE TABLE png_list_2 (png_path TEXT)")
    conn.executemany("INSERT INTO png_list_2 VALUES (?)",
                     [("x.png",), ("y.png",), ("z.png",)])
    conn.commit()
    conn.close()

    settings = AnnotateSettings(db_path=str(db), png_table="png_list_2")
    assert _compute_total(settings, False)["total"] == 3
