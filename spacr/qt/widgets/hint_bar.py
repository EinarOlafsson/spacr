"""A line at the foot of a window that says what the thing under the mouse does.

THE HOME SCREEN ALREADY WORKS THIS WAY and it is the better answer. A module
tile carries no tooltip: hovering it writes its description into a bar across
the bottom of the page. A popup would be a second copy of the same sentence
drawn ON TOP of the grid the user is reading to choose between, and these
descriptions run to several hundred characters -- fine in a fixed line the eye
can skip, wrong in a box covering the thing it describes.

The same is true of a button. A tooltip appears over the button, which is
where the pointer already is and where the user is about to click; a bar at
the foot is out of the way, holds a long sentence without covering anything,
and does not flicker in and out as the pointer crosses the row.

    bar = HintBar("Hover a control to see what it does.")
    bar.explain(clear_ram_button, "Frees cached memory. Asks first.")
    layout.addWidget(bar)

``explain`` takes the sentence off the widget's tooltip when none is given,
so a control that already had one hands it over rather than being rewritten:

    bar.explain(button)          # uses button.toolTip(), then clears it

The registration is held by the bar and the widget is watched through an
event filter, so nothing has to be unhooked when the window closes -- the
filter dies with the bar.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLabel, QWidget

#: What the bar says when nothing is under the pointer.
DEFAULT_HINT = "Hover a control to see what it does."

#: The objectName the stylesheet selects on, shared with the Home screen's
#: bar so the two are one thing wearing one style rather than two that
#: happen to look alike today.
BAR_NAME = "HintBar"


class HintBar(QLabel):
    """The line itself, plus the register of what each widget should say."""

    def __init__(self, default: str = DEFAULT_HINT,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(default, parent)
        self._default = default
        self._hints: Dict[QWidget, str] = {}
        self.setObjectName(BAR_NAME)
        self.setAlignment(Qt.AlignHCenter)
        self.setWordWrap(True)
        # Tall enough for the sentence it will hold, so the window does not
        # resize the moment the pointer touches a control.
        self.setMinimumHeight(max(28, self.sizeHint().height()))

    # -- registration ----------------------------------------------------

    def explain(self, widget: QWidget, text: str = "") -> str:
        """Have ``widget`` write ``text`` here while the pointer is on it.

        :param widget: the control to watch.
        :param text: what to say. Empty takes the widget's own tooltip,
            which is then cleared -- the sentence moves rather than being
            said twice in two places.
        :returns: the sentence registered, or ``""`` if there was none, in
            which case nothing is watched: a control with nothing to say
            should not blank the bar when the pointer crosses it.
        """
        sentence = (text or widget.toolTip() or "").strip()
        if not sentence:
            return ""
        widget.setToolTip("")
        # A screen reader reads neither the bar nor a tooltip that is gone,
        # so the sentence is put where assistive technology looks for it.
        if not widget.accessibleDescription():
            widget.setAccessibleDescription(sentence)
        self._hints[widget] = sentence
        widget.installEventFilter(self)
        return sentence

    def explains(self, widget: QWidget) -> str:
        """What ``widget`` will write here, or ``""`` if it writes nothing."""
        return self._hints.get(widget, "")

    def count(self) -> int:
        """How many controls report to this bar."""
        return len(self._hints)

    # -- behaviour -------------------------------------------------------

    def reset(self) -> None:
        """Say the default again."""
        self.setText(self._translated(self._default))

    def _translated(self, text: str) -> str:
        try:
            from ..i18n import tr
        except Exception:                               # pragma: no cover
            return text
        return tr(text)

    def eventFilter(self, obj, event):                  # noqa: N802
        kind = event.type()
        if kind == QEvent.Enter:
            sentence = self._hints.get(obj)
            if sentence:
                self.setText(self._translated(sentence))
        elif kind in (QEvent.Leave, QEvent.HoverLeave):
            # ONLY IF THIS WIDGET IS THE ONE BEING SHOWN. Two controls side
            # by side send Leave-then-Enter in that order often enough that
            # blanking unconditionally makes the bar flicker to the default
            # between neighbours.
            if self._hints.get(obj) and \
                    self.text() == self._translated(self._hints[obj]):
                self.reset()
        return super().eventFilter(obj, event)


def hint_bar_of(widget: QWidget) -> Optional[HintBar]:
    """The :class:`HintBar` belonging to ``widget``'s window, if it has one.

    Lets a helper deep in a form hand a sentence to the bar without the
    caller having to thread it down through every layer.
    """
    window = widget.window() if widget is not None else None
    if window is None:
        return None
    found = window.findChild(HintBar)
    return found


def explain_through_the_bar(widget: QWidget, text: str = "") -> bool:
    """Register ``widget`` with its window's bar. False when there is none.

    The caller decides what to do without one -- usually leave the tooltip
    where it is, which is better than a control that explains itself
    nowhere.
    """
    bar = hint_bar_of(widget)
    if bar is None:
        return False
    return bool(bar.explain(widget, text))
