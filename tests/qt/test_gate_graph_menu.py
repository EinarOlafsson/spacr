"""Right-clicking the gate editor's plot offers what you can do to it.

The menu's CONTENTS are tested, not a real popup: an offscreen Qt cannot
grab for a menu, so building one hangs the run. That is why
`graph_menu_items` returns data and `_show_graph_menu` renders it.
"""

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import Qt

from spacr.qt.screens.gate_editor import GateEditorScreen


@pytest.fixture
def screen(qt_theme_applied, qtbot):
    widget = GateEditorScreen()
    qtbot.addWidget(widget)
    return widget


def _labels(screen):
    return [label for label, _e, _c, _w in screen.graph_menu_items()
            if label is not None]


class TestTheMenuExists:

    def test_the_canvas_asks_for_a_custom_menu(self, screen):
        assert (screen.gates.canvas.contextMenuPolicy()
                == Qt.CustomContextMenu)

    def test_it_offers_the_actions_the_request_named(self, screen):
        labels = " | ".join(_labels(screen))
        assert "Save graph" in labels
        assert "Copy image" in labels
        assert "Reset view" in labels
        assert "Export gates" in labels

    def test_every_callback_is_an_existing_method(self, screen):
        """The menu is a second ROUTE to the same code, never a second
        implementation -- two routes that drift apart are worse than one."""
        for label, _enabled, callback, _why in screen.graph_menu_items():
            if label is None:
                continue
            assert callable(callback), f"{label} has no callback"


class TestItSaysWhyThingsAreOff:
    """A greyed row with no reason is a dead end that looks like a bug."""

    def test_save_is_disabled_with_no_figure_and_explains(self, screen):
        for label, enabled, _c, why in screen.graph_menu_items():
            if label and label.startswith("Save graph"):
                assert not enabled
                assert why, "disabled with no reason given"

    def test_save_becomes_available_once_something_is_drawn(self, screen,
                                                            qt_theme_applied):
        frame = pd.DataFrame({"area": np.linspace(1, 50, 40),
                              "intensity": np.linspace(0, 5, 40)})
        screen.gates.set_frame(frame)
        qt_theme_applied.processEvents()
        figure = screen.gates.canvas.figure()
        if not figure.get_axes():
            figure.add_subplot(111).scatter(frame["area"], frame["intensity"])
        enabled = {label: on for label, on, _c, _w in screen.graph_menu_items()
                   if label}
        assert enabled["Save graph…"], "still disabled with a figure drawn"


def test_an_empty_menu_is_not_a_crash(qt_theme_applied, qtbot):
    """A screen whose canvas is missing returns no items rather than
    raising -- the menu is decoration and must never break the screen.

    The stub carries `close` because the screen's own closeEvent calls it
    at teardown; a stub that only satisfies the method under test breaks
    the fixture instead of the code.
    """
    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    screen.gates = type("Stub", (), {"canvas": None,
                                     "close": lambda self: None})()
    assert screen.graph_menu_items() == []
