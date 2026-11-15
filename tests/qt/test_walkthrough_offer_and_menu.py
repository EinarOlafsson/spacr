"""When a module offers its walkthrough, what it points the Run step at, and the Help submenu.

* the stack handler offers a module's walkthrough only on the first visit to
  a settings-form module, a moment after the switch, and only if that module
  is still on show and still unseen by then;
* the Run step points at the button the screen names, or one labelled Run,
  or nothing;
* the Help menu gets one Walkthroughs submenu, before its first separator or
  at the end, with a reset entry even when the module registry cannot be read.

The stand-in window carries only the seams the walkthrough reaches for.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction  # noqa: E402
from PySide6.QtWidgets import (QMainWindow, QPushButton,  # noqa: E402
                               QStackedWidget, QWidget)

from spacr.qt import walkthrough as W  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _clean_seen():
    W.reset()
    yield
    W.reset()


class _Window(QMainWindow):
    """A host with only the seams the walkthrough reaches for."""

    def __init__(self, *, stack=None):
        super().__init__()
        self._screens = {}
        if stack is not None:
            self._stack = stack


def _module_screen(app_key=None, *, settings=True):
    screen = QWidget()
    if app_key is not None:
        screen.app_key = app_key
    if settings:
        screen._settings_model = object()
    return screen


def test_walkthrough_handler_offers_only_a_first_visit_that_is_still_on_show(
        subtests, qtbot, monkeypatch):
    """Offered once the switch settles; never for a nameless, seen, bespoke or left module."""
    offered = []
    monkeypatch.setattr(W, "maybe_show",
                        lambda window, app_key: offered.append(app_key))

    def _host(*screens):
        stack = QStackedWidget()
        qtbot.addWidget(stack)
        for screen in screens:
            stack.addWidget(screen)
        window = _Window(stack=stack)
        qtbot.addWidget(window)
        return window, stack

    def _settle():
        qtbot.wait(W._OFFER_AFTER_MS * 4)

    with subtests.test(msg="the first visit is offered after the switch"):
        offered.clear()
        window, _ = _host(_module_screen("mask"))
        W._WalkthroughHandler(window).on_current_changed(0)
        assert offered == []
        qtbot.waitUntil(lambda: offered == ["mask"], timeout=2000)
        _settle()
        assert offered == ["mask"]

    with subtests.test(msg="a stack that will not say what is on it offers nothing"):
        offered.clear()

        class _Stack:
            def currentWidget(self):              # noqa: N802 - Qt spelling
                raise RuntimeError("mid-teardown")

        window = _Window(stack=_Stack())
        qtbot.addWidget(window)
        W._WalkthroughHandler(window).on_current_changed(0)
        _settle()
        assert offered == []

    for case_id, screen, prepare in [
            ("a screen that is not a module", _module_screen(None), None),
            ("a module already seen", _module_screen("mask"),
             lambda window: W.mark_seen("mask")),
            ("a bespoke screen with no settings form",
             _module_screen("mask", settings=False), None),
            ("a pathway walkthrough is running", _module_screen("mask"),
             lambda window: setattr(window, "_pathway_walkthrough_active",
                                    True))]:
        with subtests.test(msg=case_id):
            W.reset()
            offered.clear()
            window, _ = _host(screen)
            if prepare is not None:
                prepare(window)
            W._WalkthroughHandler(window).on_current_changed(0)
            _settle()
            assert offered == []
            assert window.findChildren(W._TourOverlay) == []

    with subtests.test(msg="a module left before the offer is not walked through"):
        W.reset()
        offered.clear()
        mask, other = _module_screen("mask"), _module_screen("measure")
        window, stack = _host(mask, other)
        W._WalkthroughHandler(window).on_current_changed(0)
        stack.setCurrentWidget(other)
        _settle()
        assert offered == []

    with subtests.test(msg="a module seen before the offer is not offered again"):
        W.reset()
        offered.clear()
        window, _ = _host(_module_screen("mask"))
        W._WalkthroughHandler(window).on_current_changed(0)
        W.mark_seen("mask")
        _settle()
        assert offered == []


def test_run_button_is_the_named_one_or_one_labelled_run(subtests, qtbot):
    """Known attribute names win, then a button labelled Run, else None."""
    def _screen(*labels):
        screen = QWidget()
        qtbot.addWidget(screen)
        return screen, [QPushButton(label, screen) for label in labels]

    for attr in ("_btn_run", "_run_btn", "_btn_start"):
        with subtests.test(msg=f"named {attr}"):
            screen, [labelled, named] = _screen("Run", "Go")
            setattr(screen, attr, named)
            assert W._run_button(screen) is named

    with subtests.test(msg="the first known name wins"):
        screen, [first, second] = _screen("Go", "Start")
        screen._btn_start = second
        screen._btn_run = first
        assert W._run_button(screen) is first

    for label in ("Run", " run ", "RUN"):
        with subtests.test(msg=f"labelled {label!r}"):
            screen, [cancel, run] = _screen("Cancel", label)
            assert W._run_button(screen) is run

    for case_id, labels in [("nothing to run", ("Cancel", "Run all")),
                            ("no buttons at all", ())]:
        with subtests.test(msg=case_id):
            screen, _ = _screen(*labels)
            assert W._run_button(screen) is None


def test_install_help_menu_adds_one_walkthroughs_submenu(
        subtests, qtbot, monkeypatch):
    """No Help menu gets nothing; the submenu goes before the first separator or at the end."""
    def _module_texts(submenu):
        return [a.text() for a in submenu.actions()
                if not a.property("workflowPathway") and not a.isSeparator()]

    reset_text = "Show all walkthroughs again"
    mask_text = W.tr(W.MENU_TITLE) + ": " + W.tr("Mask")

    with subtests.test(msg="a window with no Help menu gets no submenu"):
        window = _Window()
        qtbot.addWidget(window)
        assert W.install_help_menu(window) is None
        assert getattr(window, "_walkthrough_menu", None) is None

    with subtests.test(msg="no separator puts the submenu at the end"), \
            monkeypatch.context() as patch:
        window = _Window()
        qtbot.addWidget(window)
        help_menu = window.menuBar().addMenu("Help")
        about = QAction("About", help_menu)
        help_menu.addAction(about)
        patch.setattr("spacr.qt.app.APPS",
                      [("mask", "Mask", "Segment", "d")])
        submenu = W.install_help_menu(window)
        assert submenu.title() == W.MENU_TITLE
        assert help_menu.actions() == [about, submenu.menuAction()]
        assert window._walkthrough_menu is submenu
        assert isinstance(window._walkthrough_handler, W._WalkthroughHandler)
        assert _module_texts(submenu) == [mask_text, reset_text]
        [module] = [a for a in submenu.actions() if a.property("moduleAppKey")]
        assert module.property("moduleAppKey") == "mask"
        assert module.statusTip() == "Segment"
        shown = []
        patch.setattr(W, "show_walkthrough",
                      lambda win, key, **kw: shown.append((win, key, kw)))
        module.trigger()
        assert shown == [(window, "mask", {"force": True})]

    with subtests.test(msg="a second install adds nothing"):
        assert W.install_help_menu(window) is None
        assert help_menu.actions() == [about, submenu.menuAction()]

    with subtests.test(msg="the reset entry clears every seen flag"):
        W.mark_seen("mask")
        [reset] = [a for a in submenu.actions() if a.text() == reset_text]
        reset.trigger()
        assert W.was_seen("mask") is False

    with subtests.test(msg="a separator puts the submenu before it"), \
            monkeypatch.context() as patch:
        window = _Window()
        qtbot.addWidget(window)
        help_menu = window.menuBar().addMenu("Help")
        about = help_menu.addAction("About")
        separator = help_menu.addSeparator()
        quit_action = help_menu.addAction("Report a problem")
        patch.setattr("spacr.qt.app.APPS", [])
        submenu = W.install_help_menu(window)
        assert help_menu.actions() == [about, submenu.menuAction(), separator,
                                       quit_action]
        assert _module_texts(submenu) == [reset_text]

    with subtests.test(msg="an unreadable registry still leaves a reset entry"), \
            monkeypatch.context() as patch:
        window = _Window()
        qtbot.addWidget(window)
        window.menuBar().addMenu("Help")
        patch.setitem(sys.modules, "spacr.qt.app", None)
        submenu = W.install_help_menu(window)
        assert _module_texts(submenu) == [reset_text]
