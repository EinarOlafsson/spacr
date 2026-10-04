"""No caption or tooltip a user can read says "aspect ratio".

The graph menu calls the canvas "Graph shape" and the data lock "lock axis
scales"; a stray "aspect ratio" anywhere else names neither and reads as a
third control. Swept over the source of every call that puts text on screen,
so a new widget cannot bring the phrase back unseen.
"""
from __future__ import annotations

import ast
from pathlib import Path

import spacr

_SHOWS_TEXT = {"setToolTip", "tr", "setText", "addRow", "QLabel",
               "setWhatsThis", "setPlaceholderText", "addItem", "addAction",
               "setStatusTip", "QCheckBox", "QPushButton", "QAction",
               "setTitle", "setWindowTitle", "QGroupBox"}


def _name(func):
    return getattr(func, "attr", None) or getattr(func, "id", None)


def _phrases(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _name(node.func) in _SHOWS_TEXT:
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and \
                            isinstance(sub.value, str) and \
                            "aspect ratio" in sub.value.lower():
                        yield f"{path.name}:{sub.lineno}"


def test_no_on_screen_text_says_aspect_ratio():
    root = Path(spacr.__file__).parent / "qt"
    found = [hit for path in sorted(root.rglob("*.py"))
             if "i18n_catalogs" not in path.parts
             for hit in _phrases(path)]
    assert found == []


def test_the_sweep_would_see_one():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.py"
        probe.write_text('w.setToolTip("Keeps the " "aspect ratio.")\n')
        assert list(_phrases(probe)) == ["probe.py:1"]
