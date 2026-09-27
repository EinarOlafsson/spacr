"""Pathway walkthroughs at their edges, and the tour card's placement.

* A pathway's steps are the same whether asked for by ``build_steps`` or
  shown; starting a second pathway replaces the first even when the first
  cannot be finished; a Home that will not open leaves module tours
  switched back on; a step with no screen to open navigates nowhere; Next
  on the last step ends the walkthrough.
* While a pathway runs, or when the stack cannot be read, no module
  walkthrough is offered over it.
* The tour card is not moved when the overlay has no visible area, and a
  highlight is not drawn for a menu or widget the user cannot see.
"""
from __future__ import annotations

import copy
import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QMainWindow,
    QMenu,
    QPushButton,
    QStackedWidget,
    QWidget,
)

from spacr.qt import first_run as F  # noqa: E402
from spacr.qt import walkthrough as W  # noqa: E402

pytestmark = pytest.mark.qt

PATHWAY = "pooled_screen"


class Window(QMainWindow):
    def __init__(self, refuse_home=False):
        super().__init__()
        self.navigated = []
        self._refuse_home = refuse_home
        self._screens = {}
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self.menuBar().addMenu("Help")
        self.resize(1000, 800)

    def _on_nav_selected(self, key):
        if self._refuse_home:
            raise RuntimeError("Home could not be built")
        self.navigated.append(key)
        if key not in self._screens:
            screen = QWidget()
            screen.app_key = key
            screen._settings_model = object()
            self._screens[key] = screen
            self._stack.addWidget(screen)
        self._stack.setCurrentWidget(self._screens[key])


@pytest.fixture
def window(qtbot):
    window = Window()
    qtbot.addWidget(window)
    window.show()
    return window


def test_a_pathway_asked_for_by_key_has_the_steps_it_shows():
    steps = W.build_steps("pathway:" + PATHWAY)

    assert [s.title for s in steps] == [
        s.title for s in W._pathway_steps(PATHWAY)]
    assert steps[0].title == "Home"


def test_a_second_pathway_replaces_one_that_cannot_be_finished(
        window, monkeypatch):
    first = W.show_walkthrough(window, "pathway:" + PATHWAY)
    stuck = []

    def _refuses():
        stuck.append(1)
        raise RuntimeError("Internal C++ object already deleted.")

    first._finish = _refuses

    second = W.show_walkthrough(window, "pathway:" + PATHWAY)

    assert stuck == [1]
    assert second is not first
    assert window._pathway_overlay is second
    assert window._pathway_walkthrough_active is True


def test_a_home_that_will_not_open_leaves_module_tours_on(qtbot):
    window = Window(refuse_home=True)
    qtbot.addWidget(window)

    with pytest.raises(RuntimeError, match="Home could not be built"):
        W.show_walkthrough(window, "pathway:" + PATHWAY)

    assert window._pathway_walkthrough_active is False


def test_a_step_with_no_screen_navigates_nowhere_and_next_ends_it(
        window, qtbot, monkeypatch):
    data = copy.deepcopy(W._workflow_map())
    route = data["pathways"][PATHWAY]
    for step in route["steps"]:
        module = data["modules"][step["module"]]
        module["parent"] = None
        module["home"] = ""
    monkeypatch.setattr(W, "_workflow_map", lambda: data)
    overlay = W.show_walkthrough(window, "pathway:" + PATHWAY)
    total = len(overlay._steps)

    for _ in range(total - 1):
        qtbot.mouseClick(overlay._next_btn, Qt.LeftButton)
    assert window.navigated == ["__home__"]
    assert overlay._idx == total - 1

    qtbot.mouseClick(overlay._next_btn, Qt.LeftButton)
    assert window._pathway_walkthrough_active is False
    assert window.navigated == ["__home__"]


def test_no_module_walkthrough_is_offered_over_a_pathway(window,
                                                         monkeypatch):
    offered = []
    monkeypatch.setattr(W, "maybe_show", lambda *a: offered.append(a))
    monkeypatch.setattr(W, "was_seen", lambda _key: False)
    handler = W._WalkthroughHandler(window)
    window._on_nav_selected("mask")

    window._pathway_walkthrough_active = True
    handler._offer("mask")
    assert offered == []

    window._pathway_walkthrough_active = False
    broken = W._WalkthroughHandler(window)
    broken._window = types.SimpleNamespace(_stack=None)
    broken._offer("mask")
    assert offered == []

    handler._offer("mask")
    assert offered == [(window, "mask")]


# ---------------------------------------------------------------------------
# The tour card and its highlight
# ---------------------------------------------------------------------------

def test_the_card_stays_put_when_the_overlay_has_no_visible_area(
        window, monkeypatch):
    from spacr.qt import hidpi

    overlay = F._TourOverlay(window, [F.TourStep("One", "First step")],
                             on_finish=lambda: None)
    before = overlay._card.geometry()
    far_away = types.SimpleNamespace(
        availableGeometry=lambda: QRect(100000, 100000, 10, 10))
    monkeypatch.setattr(hidpi, "screen_for_widget", lambda _w=None: far_away)

    overlay._update_card_position()

    assert overlay._card.geometry() == before
    overlay._finish()


def test_the_card_is_not_placed_before_it_is_built():
    bare = types.SimpleNamespace()

    assert F._TourOverlay._update_card_position(bare) is None
    assert not hasattr(bare, "_card")


def test_nothing_the_user_cannot_see_is_highlighted(window, qtbot):
    listed = window.menuBar().addMenu("Tools")
    unlisted = QMenu("Not on the bar", window)
    hidden = QPushButton("hidden", window)
    hidden.hide()
    elsewhere = QWidget()
    qtbot.addWidget(elsewhere)
    elsewhere.show()

    assert F._widget_rect_in_window(unlisted, window) is None
    assert F._widget_rect_in_window(hidden, window) is None
    assert F._widget_rect_in_window(elsewhere, window) is None
    shown = F._widget_rect_in_window(listed, window)
    assert shown is not None and not shown.isEmpty()

    window.menuBar().hide()
    assert F._widget_rect_in_window(listed, window) is None
