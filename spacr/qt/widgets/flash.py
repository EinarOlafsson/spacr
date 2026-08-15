"""A brief accent flash, for controls whose action leaves no other trace.

WHY THIS IS SHARED. Two controls need the same feedback for the same reason,
and the reason is not cosmetic: a click that produces no visible change is
indistinguishable from a click that was not received. The console's copy glyph
writes to the clipboard, which is invisible; the figure queue's clear control
empties a list that may already look empty. Both need to say "that happened".

It lives here rather than being written twice because a second copy is a
second duration, and two controls that flash for different lengths read as a
bug in whichever one you notice second.

The colour is deliberately NOT decided here. A flash is drawn differently by a
painted glyph and by a text label, so each control paints itself and asks
:meth:`Flash.active` whether it is flashing. What is shared is the timing and
the state machine, which is the part that must agree.
"""
from __future__ import annotations

from PySide6.QtCore import QTimer

__all__ = ["Flash", "FLASH_MS"]

#: How long the mark stays up, in milliseconds.
#:
#: 650 ms is the value the console's copy glyph has always used. Long enough
#: to register as deliberate feedback rather than a rendering glitch, short
#: enough that it is gone before the next click. Change it here and both
#: controls move together, which is the point of this module.
FLASH_MS = 650


class Flash:
    """Tracks whether a widget is currently flashing, and repaints it.

    Owned by the widget, not inherited, so a control can flash without
    committing to a base class -- the copy glyph is a ``QAbstractButton`` and
    the clear control is a ``QLabel``, and neither should have to change what
    it is in order to blink.

    :param widget: the widget to repaint when the flash starts and ends.
    :param duration_ms: how long to stay lit; defaults to :data:`FLASH_MS`.
    """

    def __init__(self, widget, duration_ms: int = FLASH_MS):
        self._widget = widget
        self._duration = int(duration_ms)
        self._active = False

    @property
    def active(self) -> bool:
        """Whether the mark should be drawn right now."""
        return self._active

    def trigger(self) -> None:
        """Light the mark and schedule it to go out.

        Re-triggering while lit restarts the clock rather than stacking
        timers, so holding down a repeat-firing control does not leave the
        mark on for as long as the last timer happens to run.
        """
        self._active = True
        self._widget.update()
        QTimer.singleShot(self._duration, self._end)

    def _end(self) -> None:
        # The widget may have been destroyed between the trigger and the
        # timeout -- a screen torn down mid-flash is ordinary, not an error.
        # shiboken raises RuntimeError on a deleted C++ object, and there is
        # nothing to repaint by then either way.
        self._active = False
        try:
            self._widget.update()
        except RuntimeError:
            pass
