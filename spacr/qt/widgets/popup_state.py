"""Whether a menu or a tooltip is currently on screen.

WHY THIS EXISTS. Opening the spaCR menu cost a burst of ~105 widget
repaints over the live GL backdrop -- 68 QLabel, 25 QWidget, 12 AppTile
-- and a tooltip did the same. That burst is the compositor re-composing
a popup over a native GL child, and it is what the user sees as the dock
and the header flickering.

The backdrop cannot stop the popup from compositing, but it can stop
being a moving target while it happens. Holding the animation still for
the moment a popup is up takes the menu from 238 painted widgets to 5,
and a tooltip from 90 to 0.

WHY IT IS A POLL AND NOT AN EVENT FILTER. The obvious implementation is
an application-wide event filter watching for Show and Hide on popup
windows. That was written, measured, and thrown away: it ran Python for
every event in the application -- 13,646 calls per second while a module
opens -- and it cost 130 ms on the very GUI-thread block that opening a
module had just been fixed to shorten (1380-1424 ms became 1473-1606).

Qt already knows the answer and will give it for two C++ calls, so the
animation asks once per frame instead. At sixty frames a second that is
120 calls, against thirteen thousand.
"""
from __future__ import annotations


def a_popup_is_on_screen() -> bool:
    """True while a menu or a tooltip is showing, anywhere in the app.

    Both are asked for, because they are different things to Qt: a menu
    is an ``activePopupWidget`` (window type ``Qt::Popup``), and a
    tooltip is not -- it is its own ``Qt::ToolTip`` window that
    :class:`~PySide6.QtWidgets.QToolTip` tracks separately. Watching only
    the first would have left the tooltip flicker exactly as it was, and
    that was reported alongside the menu.

    Never raises. This is called from an animation tick, so a failure
    here must cost a frame's decision at worst, not the backdrop: an
    unanswerable question is answered "no popup", which is the behaviour
    the application had before this existed.
    """
    try:
        from PySide6.QtWidgets import QApplication, QToolTip

        if QApplication.activePopupWidget() is not None:
            return True
        return bool(QToolTip.isVisible())
    except Exception:                                        # noqa: BLE001
        return False
