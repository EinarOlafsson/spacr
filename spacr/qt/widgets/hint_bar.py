"""Window-level, non-modal hover help for Qt controls.

:class:`HintBar` is a centered, word-wrapped label that displays registered
help text while the pointer is over a widget. On pointer leave, it restores
the default message only when that widget's text is still displayed, avoiding
an intermediate reset while the pointer moves between adjacent controls.

:meth:`HintBar.explain` registers either explicit text or the widget's current
tooltip. Successful registration clears the tooltip, supplies the same text
as the accessible description only when no accessible description is already
present, and installs the bar as an event filter. Source strings are
translated when displayed. Widgets without available text are not registered.

Use :func:`hint_bar_of` to locate the :class:`HintBar` in a widget's top-level
window, or :func:`explain_through_the_bar` to register a widget only when such
a bar exists. The bar retains the registration mapping and event filter for
its lifetime.
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
    """The line itself, plus the register of what each widget should say.

    :param default: what the line says when nothing is hovered. It is
        restored whenever a widget with no registered hint takes the pointer,
        so it should read as a prompt rather than as a blank.
    :param parent: parent widget.
    """

    def __init__(self, default: str = DEFAULT_HINT,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(default, parent)
        self._default = default
        self._hints: Dict[QWidget, str] = {}
        self.setObjectName(BAR_NAME)
        # JUSTIFIED, like the tooltips it replaces. Asked for 2026-08-28.
        # Centring is right for one short line and wrong for the three a
        # paragraph takes: a centred block has two ragged edges instead of
        # one, and reads as a caption rather than as prose.
        self.setAlignment(Qt.AlignJustify | Qt.AlignVCenter)
        self.setWordWrap(True)
        # Tall enough for the sentence it will hold, so the window does not
        # resize the moment the pointer touches a control.
        self.setMinimumHeight(max(28, self.sizeHint().height()))
        # AND NO TALLER THAN THREE LINES. The strip took whatever height the
        # longest help needed, so moving the pointer between two controls
        # whose help differs in length made the whole dialog jump. A bounded
        # strip elides instead, and the full text is still in the register.
        line = max(1, self.fontMetrics().lineSpacing())
        self.setMaximumHeight(line * 3 + 12)

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
        except Exception:
            return text
        return tr(text)

    def eventFilter(self, obj, event):                  # noqa: N802
        """Watch the widgets whose hints this bar shows.

        :param obj: the object the event is for.
        :param event: the event.
        :returns: True to stop the event going further.
        """
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
