"""Restyling a figure renders once, and cheaply.

Instruction 119 section D: "currently changing settings on most graphs laggs
to much and esspecially on the regression plot, that is super slow".

Two wasted full-quality renders, both measured on an 823-point volcano --
the shape of the tsg101 screen -- at ~263 ms each. A full render rewrites the
raster AND the vector page, so these are not cheap.

  1. CLOSING THE SETTINGS DIALOG RENDERED TWICE. The dialog's closeEvent
     already lands a full-quality redraw before exec() returns, and the
     caller then called refresh_current_figure() again. Half a second of
     dead GUI for the second one to overwrite the first with an identical
     picture.

  2. EVERY CONTEXT-MENU TOGGLE FORCED FULL QUALITY. Legend on, grid off --
     each one a full render including the vector page, while the user is
     mid-gesture. The settings dialog learned to preview long ago; the menu
     never did.

The debounce and the anti-stacking dirty flag were already there and are not
what this file is about -- they are why the app survived at all.
"""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def volcano():
    """823 points, like the screen the complaint came from."""
    rng = np.random.default_rng(0)
    figure = plt.figure(figsize=(6.2, 4.8))
    axis = figure.add_subplot(111)
    axis.scatter(rng.normal(0, 1, 823),
                 -np.log10(rng.uniform(1e-8, 1, 823)), s=26)
    axis.legend(["guides"])
    yield figure
    plt.close(figure)


# --------------------------------------------------------------------------- #
#  The context menu previews rather than forcing full quality
# --------------------------------------------------------------------------- #

def test_a_menu_toggle_asks_for_a_preview(qtbot, volcano):
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    host = pytest.importorskip("PySide6.QtWidgets").QWidget()
    qtbot.addWidget(host)
    seen = []

    def on_change(preview=False):
        seen.append(preview)

    menu = build_figure_context_menu(host, volcano, on_change=on_change)
    toggled = False
    for action in menu.actions():
        if action.isCheckable():
            action.toggle()
            toggled = True
            break

    assert toggled, "the menu offered nothing to toggle -- test is not looking"
    assert seen, "toggling did not ask for a redraw at all"
    assert all(preview is True for preview in seen), (
        f"a menu toggle forced a full-quality render ({seen}); it should "
        "preview and let a later full render catch up")


def test_a_callback_that_predates_preview_still_works(qtbot, volcano):
    """`on_change` may be an older callable taking no arguments. The keyword
    is offered and withdrawn, not assumed."""
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    host = pytest.importorskip("PySide6.QtWidgets").QWidget()
    qtbot.addWidget(host)
    calls = []

    def legacy_on_change():
        calls.append(True)

    menu = build_figure_context_menu(host, volcano, on_change=legacy_on_change)
    for action in menu.actions():
        if action.isCheckable():
            action.toggle()
            break

    assert calls, "the fallback path did not fire"


def test_a_menu_with_no_callback_does_not_raise(qtbot, volcano):
    """The toggle still has to reach the figure. A menu that swallows the
    edit along with the redraw is not 'safe', it is broken quietly."""
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    host = pytest.importorskip("PySide6.QtWidgets").QWidget()
    qtbot.addWidget(host)
    menu = build_figure_context_menu(host, volcano, on_change=None)

    grid = [a for a in menu.actions() if a.text() == "Grid"]
    assert grid, [a.text() for a in menu.actions()]
    before = any(line.get_visible() for line in volcano.axes[0].get_xgridlines())
    grid[0].toggle()

    after = any(line.get_visible() for line in volcano.axes[0].get_xgridlines())
    assert after != before, "the toggle did nothing without a redraw callback"


# --------------------------------------------------------------------------- #
#  Closing the settings dialog renders once
# --------------------------------------------------------------------------- #

def test_closing_the_settings_dialog_renders_once(qtbot, volcano, monkeypatch):
    """The dialog's own closeEvent is the full-quality render. A second one
    from the caller is pure waste."""
    from spacr.qt.widgets import figure_queue as fq

    queue = fq.FigureQueue()
    qtbot.addWidget(queue)
    queue.add_figure(volcano)

    renders = []
    monkeypatch.setattr(queue, "refresh_current_figure",
                        lambda *a, **k: renders.append(1))

    class _Dialog:
        def __init__(self, *args, **kwargs):
            self._on_change = kwargs.get("on_change")

        def exec(self):
            # What closeEvent does: one full-quality redraw.
            if self._on_change:
                self._on_change()
            return 0

    monkeypatch.setattr(fq_settings_module(), "FigureSettingsDialog", _Dialog)
    queue._open_figure_settings()

    assert len(renders) == 1, (
        f"closing the settings dialog rendered {len(renders)} times; the "
        "dialog already renders on close, so any extra is ~263 ms of dead "
        "GUI drawing the same picture")


def fq_settings_module():
    from spacr.qt.widgets import figure_settings

    return figure_settings
