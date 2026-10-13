"""Item 515: the Console and Actions headings line up with System's.

The maintainer, 2026-09-25: "the live preview dropdown text button in the
opened widget, and the same in system look great! but the same text for the
console and the actions are outside of a container so they look miss aligned
when collapsed. so please change this so the the console text is in the
console box and if collapsed behaives like the spstem one does and has the
same font size and osssitioning style as the system, also the actions should
have the text slightly to the right so it is alligned with the other text."

MEASURED, in pixels, on a themed Mask screen after the geometry settles: the
left edge of each heading's text, and its font's pixel size, open and
collapsed. Before the fix the Console and Actions text sat 13 px left of
System's (554 against 567 at 1600 px wide).

The second half is the black background the console text sometimes painted on
until the module was left and reopened: an output block whose own stylesheet a
nested polish stripped fell back to painting its viewport in the window
colour. The block now never fills its viewport, whatever its sheet.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint                          # noqa: E402
from PySide6.QtGui import QFontInfo                        # noqa: E402
from PySide6.QtWidgets import QApplication                 # noqa: E402

_PANES = ("Console", "System", "Actions", "Settings")


def _pump(n: int = 12) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _text_left(screen, label) -> int:
    """The x, in ``screen``, where ``label``'s text starts."""
    return (label.mapTo(screen, QPoint(0, 0)).x()
            + label.contentsRect().left() + label.margin())


def _pixel_size(label) -> int:
    return QFontInfo(label.font()).pixelSize()


@pytest.fixture
def mask_screen(qtbot, qt_theme_applied):
    """A themed Mask screen, every fold it may remember opened first."""
    from spacr.qt.preferences import set_folded_panel
    from spacr.qt.screens.app_screen import AppScreen

    for name in _PANES:
        set_folded_panel(f"mask/{name}", False)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.resize(1600, 1000)
    screen.show()
    _pump()
    yield screen
    for name in _PANES:
        set_folded_panel(f"mask/{name}", False)


def _headings(screen) -> dict:
    return {
        "Console": screen._console_header,
        "System": screen._usage_card.title_label,
        "Actions": screen._actions_heading,
    }


def _assert_aligned(screen, reference, headings: dict) -> None:
    edge = _text_left(screen, reference)
    size = _pixel_size(reference)
    for name, label in headings.items():
        assert label.isVisibleTo(screen), name
        assert abs(_text_left(screen, label) - edge) <= 1, (
            f"{name}'s text starts at {_text_left(screen, label)} px, "
            f"{reference.text()!r} at {edge} px")
        assert _pixel_size(label) == size, name


class TestTheHeadingsLineUp:

    def test_open(self, mask_screen):
        screen = mask_screen
        headings = _headings(screen)
        _assert_aligned(screen, headings["System"], headings)

    def test_collapsed(self, mask_screen):
        screen = mask_screen
        for folder in (screen._console_folder, screen._usage_card.folder,
                       screen._actions_folder):
            folder.set_shut(True)
        _pump()
        headings = _headings(screen)
        _assert_aligned(screen, headings["System"], headings)

    def test_with_the_live_preview_shown(self, mask_screen):
        """The preview takes the height; the three below it fold."""
        screen = mask_screen
        screen._preview_switch.setChecked(True)
        _pump()
        split = screen._runtime_splitter
        for name in ("Console", "System", "Actions"):
            assert split.is_collapsed(name), name
        live = screen._live_preview_card.title_label
        _assert_aligned(screen, live, _headings(screen))
        screen._preview_switch.setChecked(False)
        _pump()


class TestTheConsoleFoldsLikeSystem:

    def test_the_heading_is_inside_a_card_as_system_is(self, mask_screen):
        from spacr.qt.widgets import Card

        screen = mask_screen
        card = screen._console_card
        assert isinstance(card, Card)
        assert card.title_label is screen._console_header
        assert card.isAncestorOf(screen._console)
        assert screen._console_wrap.isAncestorOf(card)
        assert (screen._console_header.objectName()
                == screen._usage_card.title_label.objectName())

    def test_collapse_and_expand(self, mask_screen):
        screen = mask_screen
        split = screen._runtime_splitter
        for name, folder in (("Console", screen._console_folder),
                             ("System", screen._usage_card.folder)):
            folder.toggle()
            _pump()
            assert split.is_collapsed(name), name
            folder.toggle()
            _pump()
            assert not split.is_collapsed(name), name
        screen._console_folder.toggle()
        _pump()
        assert not screen._console.isVisibleTo(screen)
        assert screen._console_header.isVisibleTo(screen)
        screen._usage_card.folder.toggle()
        _pump()
        assert abs(screen._console_card.height()
                   - screen._usage_card.height()) <= 2, (
            "collapsed, the console is its heading strip, as System is")
        screen._usage_card.folder.toggle()
        screen._console_folder.toggle()
        _pump()
        assert screen._console.isVisibleTo(screen)


class TestTheConsoleTextNeverSitsOnBlack:

    def test_a_block_that_lost_its_sheet_paints_nothing_of_its_own(
            self, qtbot, qt_theme_applied):
        """The 408 mechanism, staged on one output block.

        A polish nested in another stylesheet style's call lets the unpolish
        through, and a widget so treated falls back to its palette: an
        output block's viewport then filled with the window colour, black in
        the dark themes, behind the console text.
        """
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        from spacr.qt.widgets.console_panel import _StdoutBlock

        host = QWidget()
        host.setObjectName("Probe515")
        host.setStyleSheet("QWidget#Probe515 { background: #ff0000; }")
        layout = QVBoxLayout(host)
        block = _StdoutBlock("hello", parent=host)
        layout.addWidget(block)
        qtbot.addWidget(host)
        host.resize(300, 120)
        host.show()
        _pump()
        viewport = block.viewport()
        block.style().unpolish(block)
        block.style().unpolish(viewport)
        _pump()
        corner = viewport.mapTo(host, viewport.rect().bottomRight())
        pixel = host.grab().toImage().pixelColor(corner.x() - 2,
                                                 corner.y() - 2)
        assert pixel.name() == "#ff0000", (
            f"the block painted {pixel.name()} over its host")

    def test_the_console_scroll_viewport_survives_the_same(
            self, qtbot, qt_theme_applied):
        """The viewport every output block sits in, staged the same way."""
        from spacr.qt.widgets import ConsolePanel

        panel = ConsolePanel()
        qtbot.addWidget(panel)
        panel.resize(400, 300)
        panel.show()
        _pump()
        viewport = panel._scroll.viewport()
        viewport.style().unpolish(viewport)
        _pump()
        assert not viewport.autoFillBackground()
