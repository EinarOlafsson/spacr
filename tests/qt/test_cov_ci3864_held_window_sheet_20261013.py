"""The held window sheet: what a decorator holds back, and what it owes.

``spacr.qt.theme`` lets a decorator (the glass treatment of a dialog) hold
its window's sheet until it is done, so the window is restyled once
instead of three times. The happy path is measured elsewhere; these pin
the edges, each as what the user's window ends up with:

* a held window is not sheeted underneath its decorator, and a nested hold
  does not release the outer one early;
* a widget whose state changes while its window is held is remembered as
  owed a polish, and only the tagged transparent containers are polished
  at the release when no window sheet is in force;
* a decorator predicate that throws is skipped, never fatal, and a window
  that has gone (``RuntimeError``) or is not a widget at all is passed
  through untouched;
* the font scale read once for a whole restyle falls back to 1.0 when the
  preference cannot be read, and is forgotten when the block ends.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                        # noqa: E402
from PySide6.QtWidgets import QWidget                                # noqa: E402

from spacr.qt import theme                                           # noqa: E402

pytestmark = pytest.mark.qt


def test_an_unreadable_font_scale_pins_one_for_the_block(monkeypatch):
    from spacr.qt import preferences

    def unreadable():
        raise OSError("the preference store is locked")

    monkeypatch.setattr(preferences, "get_font_scale", unreadable)
    assert theme._PINNED_FONT_SCALE[0] is None
    with theme._font_scale_pinned() as scale:
        assert scale == 1.0
        assert theme._PINNED_FONT_SCALE[0] == 1.0
        with theme._font_scale_pinned() as inner:
            assert inner == 1.0
        assert theme._PINNED_FONT_SCALE[0] == 1.0, (
            "the inner block forgot the outer block's scale")
    assert theme._PINNED_FONT_SCALE[0] is None


def test_a_decorator_that_throws_is_skipped_not_fatal(qtbot, monkeypatch):
    window = QWidget()
    qtbot.addWidget(window)
    other = QWidget()
    qtbot.addWidget(other)

    def broken(_window):
        raise RuntimeError("the decorator's own window is gone")

    monkeypatch.setattr(theme, "_DECORATORS_THAT_SHEET",
                        [broken, lambda candidate: candidate is window])
    assert theme._a_decorator_sheets_it(window) is True
    assert theme._a_decorator_sheets_it(other) is False

    other.setProperty(theme._SHEET_HELD, True)
    assert theme._a_decorator_sheets_it(other) is True


class _GoneWidget:
    """A widget whose C++ half has been deleted."""

    def window(self):
        raise RuntimeError("Internal C++ object already deleted.")


def test_a_change_inside_a_held_window_is_owed_a_polish(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    child = QWidget(window)

    assert theme._owe_the_polish_to_a_held_sheet(child) is False
    assert not window.property(theme._SHEET_OWES_A_POLISH)

    window.setProperty(theme._SHEET_HELD, True)
    assert theme._owe_the_polish_to_a_held_sheet(child) is True
    assert window.property(theme._SHEET_OWES_A_POLISH) is True

    assert theme._owe_the_polish_to_a_held_sheet(_GoneWidget()) is False


def test_a_nested_hold_releases_the_window_only_at_the_outer_exit(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    with theme._one_sheet_for(window) as held:
        assert held is window
        assert window.property(theme._SHEET_HELD) is True
        with theme._one_sheet_for(window) as inner:
            assert inner is window
        assert window.property(theme._SHEET_HELD) is True
    assert not window.property(theme._SHEET_HELD)


def test_something_that_is_not_a_widget_passes_through_a_hold():
    thing = object()
    with theme._one_sheet_for(thing) as held:
        assert held is thing


class _DiesDuringTheHold:
    """A window deleted while its decorator held its sheet."""

    def __init__(self):
        self.properties = {}

    def property(self, name):
        return self.properties.get(name)

    def setProperty(self, name, value):
        if value is None:
            raise RuntimeError("Internal C++ object already deleted.")
        self.properties[name] = value


def test_a_window_deleted_during_its_hold_does_not_raise_at_the_release():
    window = _DiesDuringTheHold()
    with theme._one_sheet_for(window):
        assert window.properties[theme._SHEET_HELD] is True
    assert window.properties[theme._SHEET_HELD] is True


class _Style:
    """Records what it was asked to unpolish and polish."""

    def __init__(self):
        self.calls = []

    def unpolish(self, widget):
        self.calls.append(("unpolish", widget.objectName()))

    def polish(self, widget):
        self.calls.append(("polish", widget.objectName()))


class _Styled(QWidget):
    """A widget whose style this test chooses."""

    def __init__(self, parent, name, chosen_style):
        super().__init__(parent)
        self.setObjectName(name)
        self._chosen_style = chosen_style

    def style(self):
        return self._chosen_style


def test_only_the_tagged_containers_are_polished_at_the_release(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    recorder = _Style()
    tagged = _Styled(window, "tagged", recorder)
    untagged = _Styled(window, "untagged", recorder)
    styleless = _Styled(window, "styleless", None)
    for widget in (tagged, untagged, styleless):
        widget.setAttribute(Qt.WA_StyleSheet, True)
    tagged.setProperty(theme.TRANSPARENT_PROPERTY, True)
    styleless.setProperty(theme.TRANSPARENT_PROPERTY, True)

    theme._repolish_the_tagged(window)

    assert recorder.calls == [("unpolish", "tagged"), ("polish", "tagged")]


def test_a_held_window_is_not_sheeted_underneath_its_decorator(
        qtbot, qapp):
    window = QWidget()
    qtbot.addWidget(window)
    window.setProperty(theme._SHEET_HELD, True)
    name = theme._WINDOW_SHEET_ATTRIBUTE
    before = getattr(qapp, name, None)
    setattr(qapp, name, "QWidget { color: red; }")
    try:
        assert theme._sheet_one_window(window) is False
    finally:
        setattr(qapp, name, before)
    assert window.styleSheet() == ""
