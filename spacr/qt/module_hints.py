"""Module descriptions go to the hint strip, not over the grid and not
into the status bar.

A module description is shown in ONE place -- the hint strip along the
bottom of the page -- and never as a popup over the grid.

IT USED TO GO TO THE STATUS BAR, the line in the bottom LEFT of the
window, with a four-second linger, and it flickered: text that looked as
though it were going to the hovered module and then to something else and
back again. Two writers, not one -- this filter put the description there on
every hover and Qt put the permanent message back four seconds later, so the
corner alternated. The strip does not alternate -- it holds
the last module for thirty seconds and is replaced only by the next.

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

#: Which module a widget describes. The strip resolves the sentence and
#: both links from this, so no description is composed or truncated here.
KEY_PROPERTY = "moduleAppKey"

#: Longest hint `module_hint_text` will compose. Still public, so a caller
#: that puts the result somewhere narrow gets the same protection.
#:
#: 96 BECAUSE OF A RESIZE. Module descriptions run to 154 characters, and
#: a status bar whose text demands more width than the window has raises
#: the window's own minimum width -- so every hover relaid the main window
#: out and the dock flickered. Reported 2026-09-01: "the dock on linux is
#: acting up, flickering when mouse is hovered". Nothing writes the status
#: bar on hover any more, so that constraint is historical.
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
    """Diverts module tooltips into the window's hint strip.

    :param window: the main window that owns the hint strip, and the QObject
        parent. The strip is written THROUGH the window rather than reached
        directly, so a window without one drops the hint instead of raising.
    """

    def __init__(self, window):
        super().__init__(window)
        self._window = window

    def _show(self, widget) -> bool:
        """Write ``widget``'s module into the strip. True if it landed.

        Routed through the window rather than written here, because the
        strip is whichever one the page in front owns -- Home's, or a
        module screen's -- and `MainWindow._show_module_hint` is what knows
        that, and what resolves the sentence and the two links.
        """
        try:
            key = str(widget.property(KEY_PROPERTY) or "")
        except RuntimeError:                    # the C++ half has gone
            return False
        if not key or key == "__home__":
            return False
        router = getattr(self._window, "_show_module_hint", None)
        if not callable(router):
            return False
        try:
            router(key)
        except (AttributeError, RuntimeError):
            return False
        return True

    @staticmethod
    def _shows_its_own_name(widget) -> bool:
        """Whether the widget already says what it is on screen.

        A SIDEBAR ROW HAS ITS NAME WRITTEN ON IT; a fold-strip button is
        an icon and nothing else. For the first, a description at the
        bottom of the window is extra detail about something already
        identified. For the second it is the ONLY way to learn what the
        button is, and putting it in the far corner is why the Mask masthead
        button read as having no tooltip at all.
        """
        try:
            return bool(str(widget.text() or "").strip())
        except (AttributeError, RuntimeError):
            return False

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        if event.type() != QEvent.Type.ToolTip:
            return False
        if not isinstance(watched, QWidget):
            return False
        text = module_hint_text(watched)
        if not text:
            return False
        landed = self._show(watched)
        # AN ICON-ONLY BUTTON KEEPS ITS POPUP. The description still goes
        # to the status bar -- that is what was asked for -- but the
        # popup is not suppressed, because a button with no label has
        # nothing else to identify it with.
        if not self._shows_its_own_name(watched):
            return False
        # SUPPRESSED ONLY IF IT LANDED SOMEWHERE. A window with no status
        # bar would otherwise lose the description entirely, which is
        # worse than the popup this replaces.
        return landed


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
