"""Show the backdrop alone, full screen, until any input is received.

A SEPARATE WINDOW, not the main one made full screen. Hiding spaCR's own
widgets and restoring them afterwards means remembering what was visible,
what had focus, which docks were open and what the splitters were set to --
and getting any of it wrong leaves the user's layout rearranged by something
that was meant to be a screensaver. A window of its own has nothing to
restore: it is closed and the application is exactly as it was.

IT BUILDS ITS OWN BACKDROP for the same reason. Reparenting the running one
would take it off the screen behind it, so leaving the screensaver would
have to put it back -- and a reparented GL canvas is a context that may not
survive the move.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QVBoxLayout, QWidget

LOG = logging.getLogger("spacr.qt.screensaver")


class Screensaver(QWidget):
    """A full-screen backdrop that closes on any key or click.

    :param parent: the window it was opened from, used only so Qt gives the
        screensaver the same screen.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the screensaver as its own full-screen window.

        A ``Qt.Window`` and deliberately not parented into the layout: a child
        widget cannot go full screen on its own, and a tool window loses focus
        to the main window the moment it appears -- which would send "any key"
        to the wrong place.

        :param parent: the window it was opened from, remembered so it can be
            returned to; the screensaver is not parented into it.
        """
        super().__init__(None, Qt.WindowType.Window)
        self._opened_from = parent
        self.setWindowTitle("spaCR")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._backdrop = self._build_the_backdrop()
        if self._backdrop is not None:
            layout.addWidget(self._backdrop)

    def _build_the_backdrop(self) -> Optional[QWidget]:
        """One backdrop of its own, at the user's settings.

        :returns: the widget, or None when none can be built -- in which
            case the screensaver is a black screen, which is still a
            screensaver and still closes on a key.
        """
        try:
            from .preferences import get_fractal_settings
            from .widgets.fractal_travel import (RuntimeControls, Settings,
                                                 create_fractal_widget)

            values = get_fractal_settings()
            return create_fractal_widget(
                Settings(pattern=values["pattern"],
                         backend=values["backend"],
                         quality=values["quality"], scale=values["scale"]),
                RuntimeControls(
                    speed=values["speed"], dream=values["dream"],
                    variable_speed=values["variable_speed"],
                    speed_min=values["speed_min"],
                    speed_max=values["speed_max"],
                    speed_period=values["speed_period"],
                    follow_pointer=bool(values["pointer_gravity"]),
                    pointer_size=values["pointer_size"],
                    pointer_strength=values["pointer_strength"],
                    zoom_rate=values["zoom_rate"]))
        except Exception:                                    # noqa: BLE001
            LOG.exception("could not build the screensaver backdrop")
            return None

    def paintEvent(self, _event) -> None:
        """Black behind the backdrop, so nothing shows through.

        :param _event: the paint event; not read, the whole window is filled.
        """
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        painter.end()


    def keyPressEvent(self, event) -> None:
        """Dismiss on any key.

        :param event: the Qt key event.
        """
        event.accept()
        self.close()

    def mousePressEvent(self, event) -> None:
        """Dismiss on any click.

        :param event: the Qt mouse event.
        """
        event.accept()
        self.close()

    def closeEvent(self, event) -> None:
        """Stop the backdrop before the window goes.

        A canvas whose widget is destroyed while its timer is still running
        is the crash this module must not cause: `pause` is the documented
        way to make one give its threads back.

        :param event: the close event, passed on to the base class after the
            backdrop is paused.
        """
        backdrop = getattr(self, "_backdrop", None)
        pause = getattr(backdrop, "pause", None)
        if callable(pause):
            try:
                pause()
            except Exception:                                # noqa: BLE001
                LOG.debug("could not pause the screensaver backdrop",
                          exc_info=True)
        super().closeEvent(event)


def show_screensaver(parent: Optional[QWidget] = None) -> Optional[Screensaver]:
    """Open the backdrop full screen on the parent's screen.

    :returns: the window, or None when it could not be opened.
    """
    try:
        saver = Screensaver(parent)
        if parent is not None:
            from .hidpi import screen_for_widget
            handle = screen_for_widget(parent)
            if handle is not None:
                saver.setGeometry(handle.geometry())
        saver.showFullScreen()
        saver.raise_()
        saver.activateWindow()
        saver.setFocus(Qt.FocusReason.OtherFocusReason)
        return saver
    except Exception:                                        # noqa: BLE001
        LOG.exception("could not open the screensaver")
        return None
