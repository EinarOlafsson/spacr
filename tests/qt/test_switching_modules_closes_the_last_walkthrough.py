"""Switching modules closes the walkthrough of the module left behind.

Found on a Mac 2026-10-19: every first visit opened its module's walkthrough over the whole window and
nothing closed it on the next switch, so Map Barcodes' tour sat over Regression and External Masks' over
the next twenty modules. A switch now closes any module walkthrough that is not the current module's,
without marking it seen, and at most one module walkthrough is ever open.
"""
import pytest
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget

from spacr.qt import walkthrough
from spacr.qt.first_run import _TourOverlay

pytestmark = pytest.mark.qt


class _Screen(QWidget):
    def __init__(self, app_key):
        super().__init__()
        self.app_key = app_key
        self._settings_model = object()


@pytest.fixture
def window(qtbot, monkeypatch):
    seen = set()
    monkeypatch.setattr(walkthrough, "build_steps",
                        lambda app_key: [walkthrough.WalkStep(app_key, "first step")])
    monkeypatch.setattr(walkthrough, "was_seen", lambda app_key: app_key in seen)
    monkeypatch.setattr(walkthrough, "mark_seen", seen.add)
    w = QMainWindow()
    stack = QStackedWidget()
    w.setCentralWidget(stack)
    w._stack = stack
    w._screens = {}
    for key in ("map_barcodes", "regression", "mask"):
        screen = _Screen(key)
        stack.addWidget(screen)
        w._screens[key] = screen
    w._on_nav_selected = lambda key: stack.setCurrentWidget(w._screens[key])
    handler = walkthrough._WalkthroughHandler(w)
    stack.currentChanged.connect(handler.on_current_changed)
    w.resize(900, 700)
    qtbot.addWidget(w)
    w.show()
    w._seen = seen
    return w


def _open(window):
    return [(overlay._walkthrough_app_key if hasattr(overlay, "_walkthrough_app_key") else None)
            for overlay in window.findChildren(_TourOverlay) if overlay.isVisible()]


def test_switching_modules_leaves_only_the_current_modules_walkthrough(window, qtbot):
    walkthrough.show_walkthrough(window, "map_barcodes")
    assert _open(window) == ["map_barcodes"]
    window._on_nav_selected("regression")
    assert _open(window) == []
    walkthrough.show_walkthrough(window, "regression")
    assert _open(window) == ["regression"]
    window._on_nav_selected("mask")
    assert _open(window) == []
    assert "map_barcodes" not in window._seen and "regression" not in window._seen


def test_reopening_a_walkthrough_never_stacks_a_second_one(window):
    walkthrough.show_walkthrough(window, "map_barcodes")
    walkthrough.show_walkthrough(window, "map_barcodes", force=True)
    assert _open(window) == ["map_barcodes"]
    walkthrough.show_walkthrough(window, "regression", force=True)
    assert _open(window) == ["regression"]


def test_a_pathway_walkthrough_is_not_closed_by_the_screens_it_opens(window):
    pathway = _TourOverlay(window, [walkthrough.TourStep("Route", "step")])
    pathway.show()
    window._on_nav_selected("regression")
    window._on_nav_selected("mask")
    assert pathway.isVisible()
    pathway.close()
