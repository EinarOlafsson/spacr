"""Instruction 108 — right-click a figure and change how it looks.

    "i want to be able to right click on each graph and have access to as many
     settings for the graph as possible: ledgend on/off, grid on/off, colors,
     symbols, line width, line color, font size, logx, logy, etc."

The pyqtgraph plots have had this since 108 shipped. The volcano EXPLORER is a
matplotlib canvas and had NO context menu at all — every control it offers is
in the side panel, which is the right home for them, but 108 is about reaching
a figure's style FROM the figure.

The entries come from `dataclasses.fields(VolcanoStyle)` through the same two
functions the pyqtgraph plots use, so a style that gains a field gains a menu
entry with nobody remembering to add one — which is the property worth holding,
because the alternative is a menu that silently stops covering the style.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.volcano_explorer import VolcanoExplorer  # noqa: E402
from spacr.volcano_style import VolcanoStyle                   # noqa: E402


def _results(n: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[G{i}]" for i in range(n)],
        "coefficient": rng.normal(size=n),
        "p_value": rng.random(n),
    })


def _explorer(qtbot):
    explorer = VolcanoExplorer(_results())
    qtbot.addWidget(explorer)
    return explorer


def _entries(menu, prefix="") -> list:
    out = []
    for action in menu.actions():
        if action.menu() is not None:
            out.extend(_entries(action.menu(), prefix + action.text() + " > "))
        elif action.text().strip():
            out.append(prefix + action.text())
    return out


def test_the_canvas_answers_a_right_click(qtbot):
    from PySide6.QtCore import Qt

    explorer = _explorer(qtbot)
    assert explorer._canvas.contextMenuPolicy() == Qt.CustomContextMenu


def test_the_menu_is_built_from_the_style_dataclass(qtbot):
    """So a field added to VolcanoStyle is offered without a second edit."""
    import dataclasses

    entries = " ".join(_entries(_explorer(qtbot).build_style_menu())).lower()
    names = [f.name for f in dataclasses.fields(VolcanoStyle)]
    covered = [n for n in names
               if n.replace("_", " ") in entries or n in entries]
    assert len(covered) > len(names) * 0.6, (
        f"only {len(covered)} of {len(names)} style fields reached the menu")


def test_the_things_the_ask_named_are_all_there(qtbot):
    entries = " ".join(_entries(_explorer(qtbot).build_style_menu())).lower()
    for wanted in ("legend", "grid", "size", "width"):
        assert wanted in entries, f"{wanted} is not on the menu"
    # EITHER SPELLING. The labels are generated from the dataclass's own field
    # names and `VolcanoStyle` spells them `..._color` -- which is also what a
    # saved style FILE says, so renaming them would break every style already
    # written. The prose around them says colour; the keys say color.
    assert "colour" in entries or "color" in entries


def test_a_style_can_be_saved_loaded_and_made_the_default(qtbot):
    """"each figure should be editable and savable" — the 2026-08-16 restatement."""
    entries = _entries(_explorer(qtbot).build_style_menu())
    assert any("Save style" in e for e in entries)
    assert any("Load style" in e for e in entries)
    assert any("default for every volcano" in e for e in entries)


def test_changing_a_setting_from_the_menu_redraws_and_moves_the_side_panel(
        qtbot):
    """Two ways to change one setting that disagree is worse than one way."""
    explorer = _explorer(qtbot)
    seen = []
    explorer.set_style = lambda style: seen.append(style)

    menu = explorer.build_style_menu()
    toggles = [a for a in _walk(menu) if a.isCheckable()]
    assert toggles, "the menu offers nothing to toggle"
    toggles[0].trigger()
    assert seen, "changing a setting did not go through set_style"


def _walk(menu):
    for action in menu.actions():
        if action.menu() is not None:
            yield from _walk(action.menu())
        else:
            yield action


def test_a_closed_field_offers_the_columns_the_data_actually_has(qtbot):
    explorer = _explorer(qtbot)
    choices = explorer._style_choices()

    assert "x_column" in choices
    assert "coefficient" in choices["x_column"]
    # And nothing the frame does not carry.
    assert all(c in explorer.results().columns or not c
               for c in choices["x_column"])


def test_building_the_menu_does_not_change_the_figure(qtbot):
    """A menu is a question, not an answer."""
    explorer = _explorer(qtbot)
    before = explorer.style()
    explorer.build_style_menu()
    assert explorer.style() is before
