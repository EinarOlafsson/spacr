"""Item 408, the chat-box half: a hovered setting left the AI chat box black.

THE REPORT: "a black box apears as the background for the settings container
... and for the AI chat box. then when i go out of the modual to home or
another moduale and back again the black box is gone." The settings half was
a table mounted after the sweeps (4bd07819d). The maintainer confirmed that
one gone and this one still there.

WHAT HAPPENS, measured on a real MainWindow on Mask. The first hover writes
the bottom hint strip, and `_hold_the_hint` connects a timer to a method of
the screen. PySide grows the screen's metaobject for that connection, so
Qt's next `ensurePolished` polishes the screen again, and the stylesheet
polish raises `PaletteChange` on it. `AppScreen.changeEvent` answers that
with the container sweep -- from INSIDE Qt's polish. `QStyleSheetStyle`
refuses a polish nested in another sheet style's call and does not refuse
the unpolish, so `make_transparent` stripped the stylesheet from every
re-tagged widget that carries a sheet of its own: the console splitter, the
chat row and the chat input, whose viewport went back to
`autoFillBackground` and painted an opaque `QPalette.Base`. Leaving the
module and coming back swept again outside a polish, which is why it healed.

Both tests assert the STATE that decides the paint, not pixels: a grab
re-renders the tree and cannot show what the display is holding.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _settle(qapp, rounds: int = 60) -> None:
    for _ in range(rounds):
        qapp.processEvents()


@pytest.fixture
def mask_window(qapp, qtbot, monkeypatch):
    """A real MainWindow on Mask, over the window backdrop the sweep serves.

    The backdrop is pinned on rather than assumed. Without it the palette
    handler never sweeps, and both tests would pass for nothing -- another
    file in this directory has been measured leaving it off.
    """
    from spacr.qt import preferences as prefs
    from spacr.qt.app import MainWindow

    for name in ("SPACR_NO_BACKDROP", "SPACR_NO_GL"):
        monkeypatch.delenv(name, raising=False)
    was = (prefs.get_ambient_enabled(), prefs.get_tooltips_bottom_enabled())
    prefs.set_ambient_enabled(True)
    prefs.set_tooltips_bottom_enabled(True)
    prefs.apply_preferences_to_app(qapp)
    window = MainWindow()
    qtbot.addWidget(window)
    try:
        window.resize(1400, 900)
        window.show()
        window._on_nav_selected("mask")
        _settle(qapp)
        assert window.window_backdrop() is not None, (
            "no window backdrop, so the sweep this file is about never runs")
        yield window
    finally:
        prefs.set_ambient_enabled(was[0])
        prefs.set_tooltips_bottom_enabled(was[1])
        window.close()


def _count_sweeps(screen, monkeypatch) -> list:
    """Record each container sweep, so a pass cannot mean none happened."""
    sweeps = []
    real = screen._clear_page_surfaces

    def counted():
        sweeps.append(1)
        real()

    monkeypatch.setattr(screen, "_clear_page_surfaces", counted)
    return sweeps


def _what_lost_its_sheet(screen) -> list:
    """The chat widgets the stylesheet no longer reaches, by name."""
    from PySide6.QtCore import Qt

    console = screen._console
    chat = console._input
    lost = [name for name, widget in (("chat input", chat),
                                      ("chat row", console._chat_row),
                                      ("console splitter", console._split))
            if not widget.testAttribute(Qt.WA_StyleSheetTarget)]
    if chat.viewport().autoFillBackground():
        lost.append("the chat input's viewport fills QPalette.Base")
    return lost


def test_hovering_a_setting_leaves_the_chat_box_styled(mask_window, qapp,
                                                       monkeypatch):
    """The maintainer's gesture: the pointer enters a setting on Mask."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication, QLabel

    screen = mask_window._stack.currentWidget()
    assert _what_lost_its_sheet(screen) == [], "already broken before a hover"
    sweeps = _count_sweeps(screen, monkeypatch)
    label = next(w for w in screen.findChildren(QLabel)
                 if w.property("settingKey") and w.isVisible())

    QApplication.sendEvent(label, QEvent(QEvent.Enter))
    _settle(qapp)

    assert sweeps, "the hover never reached the sweep; this proves nothing"
    assert _what_lost_its_sheet(screen) == []


def test_a_stylesheet_polish_of_the_screen_leaves_the_chat_box_styled(
        mask_window, qapp, monkeypatch):
    """The mechanism on its own, so the test above cannot pass by accident.

    The hover reaches the polish only because PySide grows the screen's
    metaobject, which a ``@Slot`` on the hint timer's target would quietly
    take away. Polishing the screen through its own style is the same nested
    `PaletteChange` with nothing in between.
    """
    screen = mask_window._stack.currentWidget()
    assert _what_lost_its_sheet(screen) == [], "already broken before a polish"
    sweeps = _count_sweeps(screen, monkeypatch)

    screen.style().polish(screen)
    _settle(qapp)

    assert sweeps, "the polish never reached the sweep; this proves nothing"
    assert _what_lost_its_sheet(screen) == []
