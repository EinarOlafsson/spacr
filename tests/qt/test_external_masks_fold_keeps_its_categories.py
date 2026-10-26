"""External Masks opened from Import's fold strip shows every category.

Lesson 31 on the workstation: the Object Crops category was hidden on the
External Masks page. Since the settings search strip went on while a screen
is built (1ee022bf1b), a module built for a fold button got one too; it
opened at the module's remembered disclosure, Essentials by default, and
hid every category without an essential setting. No strip is installed on
a page that never joins the stack.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setenv("SPACR_PREWARM", "0")
    from spacr.qt import preferences as prefs
    from spacr.qt.app import MainWindow
    from spacr.qt.settings_search import forget_disclosure

    shown = prefs.get_show_alpha()
    prefs.set_show_alpha(True)
    forget_disclosure()
    win = MainWindow(initial_app="__home__")
    win.resize(1400, 900)
    win.show()
    for _ in range(3):
        qapp.processEvents()
    yield win
    win.close()
    win.deleteLater()
    qapp.processEvents()
    prefs.set_show_alpha(shown)


def _settled(qapp, seconds=2.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.02)


def test_the_folded_external_masks_page_shows_object_crops(window, qapp):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    window.open_module("foreign")
    _settled(qapp, 1.0)
    buttons = [button for button in window.findChildren(FoldButton)
               if button.app_key == "external_masks" and button.isVisible()]
    assert len(buttons) == 1
    buttons[0].click()
    _settled(qapp)
    screens = [child for child in window.findChildren(AppScreen)
               if child.app_key == "external_masks" and child.isVisible()]
    assert len(screens) == 1
    screen = screens[0]

    titles = {section.property("settingsCategorySource"): section
              for section in screen.rendered_settings_sections()}
    for title in ("Object Crops", "Measurements", "Advanced"):
        assert title in titles, sorted(titles)
        assert not titles[title].isHidden(), (
            f"{title} is hidden on the folded External Masks page")
    assert getattr(screen, "_settings_search", None) is None, (
        "a folded page gets no settings search strip")


def test_a_stack_page_still_gets_the_strip(window):
    screen = window._build_screen("measure")
    assert getattr(screen, "_settings_search", None) is not None
