"""Every ``i18n_catalogs`` import survives the package being absent.

A lightweight source install omits ``spacr/qt/i18n_catalogs`` -- 33 MB of
extended translations that the compact core catalog stands in for. The
exclusion rests on a contract stated in ``spacr/qt/i18n.py`` itself:

    External catalogs add coverage; their absence must not make the
    compact core catalog unavailable.

One import did not honour it. ``retranslate_widget_tree`` imported
``setting_label`` unguarded, and its enclosing ``except`` catches
``AttributeError``, ``RuntimeError`` and ``TypeError`` -- none of which
an ``ImportError`` is. So on a light install every screen change logged a
traceback and abandoned translating that screen, once more for each late
settings panel:

    ERROR spacr.qt.app: Could not translate the mask screen
    ModuleNotFoundError: No module named 'spacr.qt.i18n_catalogs'

Checking that SOME of the imports were guarded is what let that through,
so this walks the source and checks every one.
"""
from __future__ import annotations

import ast
import builtins
import pathlib
import sys

import pytest

pytest.importorskip("PySide6")

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "spacr"


def _catalog_imports():
    """Every ``import ... i18n_catalogs`` in the package, with its file."""
    found = []
    for path in PACKAGE.rglob("*.py"):
        if "i18n_catalogs" in path.parts:
            continue                       # the catalogs' own modules
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                # pragma: no cover - unparseable
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module \
                    and "i18n_catalogs" in node.module:
                found.append((path, node))
            elif isinstance(node, ast.Import):
                if any("i18n_catalogs" in a.name for a in node.names):
                    found.append((path, node))
    return found


def _guarding_handlers(tree, target):
    """The except clauses that would catch ``target`` raising."""
    handlers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for stmt in node.body:
                for inner in ast.walk(stmt):
                    if inner is target:
                        handlers.extend(node.handlers)
    return handlers


def _catches_import_error(handler) -> bool:
    names = []
    kind = handler.type
    if kind is None:
        return True                        # bare except
    parts = kind.elts if isinstance(kind, ast.Tuple) else [kind]
    for part in parts:
        if isinstance(part, ast.Name):
            names.append(part.id)
        elif isinstance(part, ast.Attribute):
            names.append(part.attr)
    return any(n in ("ImportError", "ModuleNotFoundError", "Exception",
                     "BaseException") for n in names)


def test_the_package_still_has_catalog_imports_to_check():
    """Or the sweep below would pass by finding nothing."""
    assert _catalog_imports(), "no i18n_catalogs imports found to check"


def test_every_catalog_import_is_guarded_against_the_package_being_absent():
    """THE REGRESSION, checked across the whole package rather than
    at the two or three sites somebody happens to look at."""
    unguarded = []
    for path, node in _catalog_imports():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # re-find the node in the freshly parsed tree by position
        target = next(
            (n for n in ast.walk(tree)
             if isinstance(n, (ast.Import, ast.ImportFrom))
             and n.lineno == node.lineno), None)
        if target is None:                 # pragma: no cover - defensive
            continue
        handlers = _guarding_handlers(tree, target)
        if not any(_catches_import_error(h) for h in handlers):
            unguarded.append(
                f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not unguarded, (
        "these i18n_catalogs imports are not guarded against the package "
        "being absent, which is exactly what a lightweight source install "
        "produces: " + ", ".join(unguarded))


def test_retranslating_a_tree_survives_the_catalogs_being_gone():
    """The behaviour, not just the shape of the source.

    Simulates the light install and drives the call that failed on it.
    """
    from PySide6.QtWidgets import QLabel, QWidget

    from spacr.qt.i18n import retranslate_widget_tree

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if "i18n_catalogs" in name:
            raise ModuleNotFoundError(
                "No module named 'spacr.qt.i18n_catalogs'")
        return real_import(name, *args, **kwargs)

    hidden = {k: v for k, v in sys.modules.items() if "i18n_catalogs" in k}
    for key in hidden:
        del sys.modules[key]

    # A LABEL THE COMPACT CATALOG DOES NOT KNOW. "Settings" is in it, so
    # the lookup takes the `if` branch and never reaches the import at
    # all -- which is how the first draft of this test passed against the
    # unguarded version.
    from spacr.qt.i18n import _ROWS, _TERM_ROWS

    source = "Minimum object solidity"
    assert source not in _ROWS and source not in _TERM_ROWS, (
        "this label reached the compact catalog, so the external one is "
        "never consulted and this test proves nothing")

    root = QWidget()
    label = QLabel(source, root)
    # Qt property names, not the private ones: the branch is gated on
    # `settingKey` AND `settingsAppKey` both being set.
    label.setProperty("_spacr_i18n_setting_text", source)
    label.setProperty("settingKey", "min_solidity")
    label.setProperty("settingsAppKey", "mask")

    builtins.__import__ = blocked
    try:
        retranslate_widget_tree(root, "fr")     # must not raise
    finally:
        builtins.__import__ = real_import
        sys.modules.update(hidden)
        root.deleteLater()


def test_the_compact_catalog_still_answers_without_the_external_one():
    """So the test above is not passing because nothing was translated."""
    from spacr.qt.i18n import tr

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if "i18n_catalogs" in name:
            raise ModuleNotFoundError("simulated light install")
        return real_import(name, *args, **kwargs)

    hidden = {k: v for k, v in sys.modules.items() if "i18n_catalogs" in k}
    for key in hidden:
        del sys.modules[key]
    builtins.__import__ = blocked
    try:
        assert tr("Settings", "fr") == "Paramètres"
    finally:
        builtins.__import__ = real_import
        sys.modules.update(hidden)
