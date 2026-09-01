"""Module descriptions go to the bottom of the window, not over the grid.

Asked for on 2026-09-01: "the tooltips for modules should only be shown
at the bottom of the screen".

Home already worked this way and the reason is written on ``AppTile``:
these blurbs run to several hundred characters, which is fine in a fixed
line the eye can skip and wrong in a box drawn on top of the very grid
you are reading to choose between. The sidebar and the fold strip kept
popping them anyway.

The hook is ``QEvent.ToolTip`` rather than hover. It fires exactly when
Qt has decided to show a tooltip -- after the same delay, at the same
moment -- so the description appears when the user expects it, and
returning ``True`` from the filter is what stops the popup being drawn.
Reimplementing the timing on ``HoverEnter`` would be a second, slightly
different delay for the same gesture.

Accessible names and descriptions are untouched: they are set separately
on these widgets and are what a screen reader reads, so moving the
visible text costs no assistive text.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QWidget

LOG = logging.getLogger(__name__)

#: Widgets carrying this property describe a module, and their
#: description belongs in the status bar.
SUMMARY_PROPERTY = "moduleSummarySource"
NAME_PROPERTY = "moduleNameSource"

#: How long the description stays after the pointer leaves. Long enough
#: to finish reading a sentence the pointer has already moved off,
#: short enough not to describe a module nobody is pointing at.
LINGER_MS = 4000

#: Longest hint put on the status bar.
#:
#: THE BAR MUST NOT RESIZE. Module descriptions run to 154 characters,
#: and a status bar whose text demands more width than the window has
#: raises the window's own minimum width -- so every hover relaid the
#: main window out and the dock flickered. Reported 2026-09-01: "the
#: dock on linux is acting up, flickering when mouse is hovered".
#:
#: Eliding here rather than relying on the label to do it, because the
#: reflow happens when the bar ASKS for the width, which is before any
#: painting-time elision could help.
MAX_HINT_CHARS = 96


def module_hint_text(widget: QWidget) -> str:
    """The line a module widget contributes, or ``""``.

    Name and summary are read from PROPERTIES rather than from the
    tooltip, so the text is the canonical English source and a language
    switch retranslates it rather than translating a translation.
    """
    if widget is None:
        return ""
    try:
        summary = widget.property(SUMMARY_PROPERTY)
        name = widget.property(NAME_PROPERTY)
    except RuntimeError:                    # the C++ half has gone
        return ""
    summary = str(summary or "").strip()
    name = str(name or "").strip()
    if not summary:
        return name
    line = f"{name} — {summary}" if name else summary
    if len(line) <= MAX_HINT_CHARS:
        return line
    # Cut on a word so the tail is not half a word, and mark it so the
    # reader knows there is more rather than thinking the sentence ends
    # oddly.
    cut = line[:MAX_HINT_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:—-")
    return f"{cut}…"


class _ModuleHints(QObject):
    """Diverts module tooltips into a window's status bar."""

    def __init__(self, window):
        super().__init__(window)
        self._window = window

    def _show(self, text: str) -> bool:
        try:
            bar = self._window.statusBar()
        except (AttributeError, RuntimeError):
            return False
        if bar is None:
            return False
        bar.showMessage(text, LINGER_MS)
        return True

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        if event.type() != QEvent.Type.ToolTip:
            return False
        if not isinstance(watched, QWidget):
            return False
        text = module_hint_text(watched)
        if not text:
            return False
        # SUPPRESSED ONLY IF IT LANDED SOMEWHERE. A window with no status
        # bar would otherwise lose the description entirely, which is
        # worse than the popup this replaces.
        return self._show(text)


def install_module_hints(window) -> Optional[_ModuleHints]:
    """Send module descriptions to ``window``'s status bar.

    Installed on the APPLICATION rather than on each button: the fold
    strip builds its buttons lazily, per host masthead, and a filter
    installed per widget would miss every one made after this ran.

    :returns: the filter, so a caller can remove it; ``None`` when there
        is no application to install it on.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return None
    hints = _ModuleHints(window)
    app.installEventFilter(hints)
    return hints
