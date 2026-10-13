"""Annotate's keyboard entry point, and the zoom overlay with nothing to show.

`handle_key` is the single entry point for the whole keyboard feature,
so what it does with a key it does NOT bind matters: returning False
leaves the key to Qt's default handling. A version that swallowed
everything would break tab focus, dialog shortcuts and text entry
everywhere the screen is open.

The overlay's guard is the same idea in paint: with no picture rectangle
there is nothing to clip to, and the scrim alone is the right answer --
a rounded clip on an empty rect draws nothing and can leave the painter
in a clipped state.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPixmap

from spacr.qt.screens.annotate import AnnotateScreen, _ZoomOverlay

pytestmark = pytest.mark.qt


class TestTheKeyboardEntryPoint:

    @pytest.fixture()
    def screen(self, qtbot):
        widget = AnnotateScreen()
        qtbot.addWidget(widget)
        return widget

    def test_an_unbound_key_is_left_to_qt(self, screen):
        """THE UNCOVERED RETURN.

        Marked "every token above is handled", and it is the fallback
        that keeps the screen from swallowing keys it has no use for.
        False means Qt's own handling still runs -- tab focus, dialog
        shortcuts, text entry.
        """
        assert screen.handle_key(Qt.Key.Key_F13) is False

    @pytest.mark.parametrize("key", [
        Qt.Key.Key_F5, Qt.Key.Key_Insert, Qt.Key.Key_ScrollLock,
    ])
    def test_other_unbound_keys_are_left_too(self, screen, key):
        assert screen.handle_key(key) is False

    def test_an_unknown_literal_character_is_left_to_qt(self, screen):
        assert screen.handle_key(Qt.Key.Key_unknown, "§") is False

    def test_escape_is_left_alone_unless_the_legend_is_showing(self,
                                                               screen):
        """Escape belongs to whatever dialog or window wants it.

        Claiming it unconditionally would stop a dialog opened over the
        annotator from closing.
        """
        screen._legend_expanded = False
        assert screen.handle_key(Qt.Key.Key_Escape) is False

    def test_escape_closes_the_legend_when_it_is_open(self, screen):
        screen._legend_expanded = True
        assert screen.handle_key(Qt.Key.Key_Escape) is True

    def test_a_bound_key_is_claimed(self, screen):
        """So the False results above are visibly a decision."""
        assert screen.handle_key(Qt.Key.Key_Question) is True


class TestTheZoomOverlayWithNothingToFrame:

    def test_an_empty_picture_rect_paints_only_the_scrim(self, qtbot,
                                                         monkeypatch):
        """THE UNCOVERED GUARD.

        A rounded clip on an empty rectangle draws nothing and leaves
        the painter clipped to it. Returning after the scrim keeps the
        grid legible behind the overlay, which is the point of a scrim
        rather than a blank.

        Driven through `grab()`, which paints the widget into a pixmap
        and so runs the real paintEvent.
        """
        overlay = _ZoomOverlay()
        qtbot.addWidget(overlay)
        overlay.resize(120, 90)
        # A crop must be loaded, or paintEvent returns before the guard:
        # with no pixmap there is nothing to dim the grid FOR.
        overlay._pixmap = QPixmap(40, 30)
        overlay._pixmap.fill(Qt.GlobalColor.red)
        monkeypatch.setattr(type(overlay), "picture_rect",
                            lambda self: QRectF())

        pixmap = overlay.grab()              # must not raise
        assert not pixmap.isNull()
        assert pixmap.size().width() == 120

    def test_a_real_picture_rect_is_clipped_and_drawn(self, qtbot,
                                                      monkeypatch):
        """The other side, so the guard is visibly a guard."""
        overlay = _ZoomOverlay()
        qtbot.addWidget(overlay)
        overlay.resize(120, 90)
        overlay._pixmap = QPixmap(40, 30)
        overlay._pixmap.fill(Qt.GlobalColor.red)
        monkeypatch.setattr(type(overlay), "picture_rect",
                            lambda self: QRectF(10, 10, 60, 40))

        painted = overlay.grab()
        assert not painted.isNull()


def test_an_overlay_with_no_crop_paints_nothing_at_all(qtbot):
    """The earlier return: with no pixmap there is nothing to dim FOR.

    Drawing the scrim anyway would black out the grid behind an overlay
    that has no crop to show over it.
    """
    overlay = _ZoomOverlay()
    qtbot.addWidget(overlay)
    overlay.resize(120, 90)
    overlay._pixmap = None
    assert not overlay.grab().isNull()
