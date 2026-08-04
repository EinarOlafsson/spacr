"""Widgets that used to freeze the dark palette now follow Preferences."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

#: Only the frozen-palette deprecation is fatal in the child. Making every
#: DeprecationWarning fatal would hand this test a failure the day numpy or
#: PySide6 deprecates something, which is not what its name promises.
_ARM = (
    "import sys, warnings\n"
    "assert 'spacr.qt.app' not in sys.modules\n"
    "warnings.filterwarnings('error', message=r'.*theme\\.PALETTE.*',\n"
    "                        category=DeprecationWarning)\n"
)


def _child(code: str):
    """Run ``code`` in a fresh interpreter rooted at *this* checkout."""
    import spacr

    root = str(Path(spacr.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                          capture_output=True, text=True, timeout=300)


def test_qt_app_import_has_no_frozen_palette_warning():
    """Importing the app must not resolve ``theme.PALETTE``.

    This cannot be measured in-process: ``spacr.qt.app`` is already in
    ``sys.modules`` before the first test runs, so ``import spacr.qt.app``
    under a ``warnings.simplefilter("error")`` is a dict lookup that
    executes none of the module body — the version of this test that did
    that could not fail. A subprocess is the only place the body actually
    runs, and the child asserts the module was absent beforehand and
    prints the file it executed, so "it ran" is measured rather than
    assumed.
    """
    import spacr.qt.app

    done = _child(_ARM + "import spacr.qt.app\n"
                         "print('RAN', spacr.qt.app.__file__)\n")
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [
        "RAN", str(Path(spacr.qt.app.__file__).resolve())], done.stdout
    assert "DeprecationWarning" not in done.stderr

    # The control: one deliberate `theme.PALETTE` read, under the same
    # arming, is a non-zero exit. Without it the assertions above would
    # also pass for a child that could never have warned at all.
    control = _child(_ARM + "import spacr.qt.theme as theme\n"
                            "theme.PALETTE\n"
                            "print('RAN')\n")
    assert control.returncode != 0
    assert "DeprecationWarning" in control.stderr
    assert "theme.PALETTE" in control.stderr


def test_no_qt_module_imports_the_frozen_dark_palette():
    import spacr.qt

    root = Path(spacr.qt.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "theme.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                    alias.name == "PALETTE" for alias in node.names):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_toggle_paints_with_current_palette(qtbot, monkeypatch):
    from PySide6.QtGui import QImage
    from spacr.qt.widgets import toggle

    colors = {
        "accent": "#123456",
        "surface_alt": "#654321",
        "border": "#abcdef",
        "fg": "#fedcba",
    }
    monkeypatch.setattr(toggle, "active_palette", lambda: colors)
    widget = toggle.Toggle("test")
    qtbot.addWidget(widget)
    widget.resize(100, 30)

    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image)

    assert image.pixelColor(20, 15).name() == colors["surface_alt"]


def test_hover_tooltip_refreshes_after_theme_change(qtbot, monkeypatch):
    from spacr.qt.widgets import hover_tooltip

    current = {
        "surface_alt": "#112233",
        "border": "#223344",
        "fg": "#eeeeee",
    }
    monkeypatch.setattr(
        hover_tooltip, "active_palette", lambda: current.copy())
    tip = hover_tooltip.HoverTooltip()
    qtbot.addWidget(tip)
    assert "#112233" in tip.styleSheet()

    current["surface_alt"] = "#445566"
    anchor = hover_tooltip.QWidget()
    qtbot.addWidget(anchor)
    tip.show_for(anchor, "Help")
    assert "#445566" in tip.styleSheet()
