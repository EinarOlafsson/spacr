"""
First-launch tour — one-time coach-marks over the home screen.

Fires the first time ``spacr`` boots (state stored in QSettings). A
translucent full-window overlay dims the app; a numbered card walks
the user through: sidebar → Demos menu → home tiles → hint bar. The
user can dismiss at any point via Skip / Esc; the "seen" flag is
saved on skip OR after the last step so the tour never fires twice
unless they hit "Reset" in Preferences.

Public API::

    from spacr.qt.first_run import (
        maybe_show_tour, was_tour_shown, mark_tour_seen,
        reset_tour_state,
    )

    # In MainWindow.__init__ after everything is built:
    maybe_show_tour(self)

    # From a Preferences reset button:
    reset_tour_state()

The tour is deliberately spartan — five steps, ~90 seconds tops.
Users who don't want it hit Esc and never see it again.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

LOG = logging.getLogger("spacr.qt.first_run")

_ORG = "spacr"
_APP = "qt"
_KEY_TOUR_SEEN = "onboarding/first_run_tour_seen"


def _settings():
    """Open spaCR's ``QSettings``.

    Imported inside the call so this module can be read without Qt.

    :returns: the settings store.
    """
    from PySide6.QtCore import QSettings
    return QSettings(_ORG, _APP)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def was_tour_shown() -> bool:
    """Return True iff the user has completed or dismissed the tour."""
    raw = _settings().value(_KEY_TOUR_SEEN, False)
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("true", "1", "yes")


def mark_tour_seen() -> None:
    """Persist the "seen" flag so the tour doesn't fire on future boots."""
    _settings().setValue(_KEY_TOUR_SEEN, True)


def reset_tour_state() -> None:
    """Clear the "seen" flag — next launch shows the tour again."""
    _settings().remove(_KEY_TOUR_SEEN)


# ---------------------------------------------------------------------------
# Tour steps
# ---------------------------------------------------------------------------

@dataclass
class TourStep:
    """One narrated coach-mark.

    :ivar title: short headline shown on the card.
    :ivar body: 1-2 sentences under the title.
    :ivar highlight: callable returning the widget to highlight, or
        None to centre the card without a highlight box.
    """
    title:     str
    body:      str
    highlight: Optional[Callable[[QMainWindow], Optional[QWidget]]] = None


def _section_names_sentence() -> str:
    """List the real home-page sections, read from the app registry.

    Hard-coding them here is how this line came to advertise "Core,
    Analysis, Cellpose and Sequencing" long after those sections stopped
    existing. Reading the registry keeps the tour honest the next time
    the grouping changes.

    It walks :data:`spacr.qt.app.APPS` in APPS order rather than
    :data:`spacr.qt.app.SECTIONS`, because the headings the sidebar
    *draws* are the ones its rows produce — naming a section with no
    rows under it would send the reader looking for a heading that is
    not there.
    """
    try:
        from .app import APPS
        names = list(dict.fromkeys(str(row[3]) for row in APPS))
    except Exception:
        names = []
    if not names:
        return (
            "Primary modules are grouped here by purpose; related workflows "
            "are reached from their host module."
        )
    if len(names) == 1:
        listed = names[0]
    else:
        listed = ", ".join(names[:-1]) + " and " + names[-1]
    return (
        f"Primary modules are grouped here into {listed}; related workflows "
        "are reached from their host module."
    )


DEFAULT_TOUR: List[TourStep] = [
    TourStep(
        title="Welcome to spaCR",
        body="This quick 5-step tour will show you the home layout. "
             "Press Esc at any time to skip.",
        highlight=None,
    ),
    TourStep(
        title="Sidebar — apps by category",
        body=_section_names_sentence()
             + " Click any name to open it. Ctrl+1 through Ctrl+9 opens "
               "the first nine "
               "apps in sidebar order.",
        highlight=lambda w: getattr(w, "_sidebar", None),
    ),
    TourStep(
        title="Demos menu",
        body="Load a synthetic demo dataset for a selected core workflow "
             "in one click — no data of your own required. Use it to try "
             "spaCR before loading an experiment.",
        highlight=lambda w: find_menu(w, "Demos"),
    ),
    TourStep(
        title="Drag & drop",
        body="Drop a folder of acquisition images onto Mask to set its "
             "input; Mask detects the filename regex and displays a metadata "
             "validation summary in the Console. Measure, Annotate and other modules "
             "accept the files or folders described by their input controls.",
        highlight=None,
    ),
    TourStep(
        title="Command palette",
        body="Ctrl+K opens a searchable list of every app, every "
             "recent run, and every menu action. Ctrl+, opens "
             "Preferences. F1 shows the shortcut cheat sheet.",
        highlight=None,
    ),
]


def find_menu(window: QMainWindow, title: str) -> Optional[QWidget]:
    """The window's menu-bar menu titled ``title``, ignoring ``&``.

    Found through ``findChildren`` rather than by walking the menu bar's
    actions and calling ``QAction.menu()``. That reading is the obvious one
    and it does not survive on PySide6 6.11: the QMenu wrapper it returns is
    only valid while the QAction wrapper it came off is alive, so the menu
    went stale the moment this function returned — "Internal C++ object
    (PySide6.QtWidgets.QMenu) already deleted" on the very next line — and
    keeping the owners alive as attributes segfaulted during the next event
    dispatch instead. ``findChildren`` hands back children the menu bar owns
    in C++, which stay valid for as long as the window does.

    :param window: the live main window.
    :param title: menu title without its mnemonic ampersand.
    """
    from PySide6.QtWidgets import QMenu
    try:
        bar = window.menuBar()
        if bar is None:
            return None
        menus = bar.findChildren(QMenu)
    except Exception:
        return None
    for menu in menus:
        try:
            if menu.title().replace("&", "") == title:
                return menu
        except RuntimeError:
            continue
    return None


#: Retained under the old private name for anything that imported it.
_find_menu = find_menu


# ---------------------------------------------------------------------------
# Overlay widget
# ---------------------------------------------------------------------------

class _TourOverlay(QWidget):
    """Translucent overlay + step card. Owns the tour lifecycle."""

    def __init__(self, window: QMainWindow, steps: List[TourStep],
                 on_finish: Optional[Callable[[], None]] = None):
        """
        :param window: the main window the overlay covers.
        :param steps: the narrated coach-marks, in order.
        :param on_finish: called once when the tour is finished or skipped,
            instead of marking the app-wide first-run flag. This is what
            lets :mod:`spacr.qt.walkthrough` reuse the overlay for a
            per-module tour without its own copy of the rendering — a second
            dimmed card would be a second thing to keep looking like this
            one.
        """
        super().__init__(window)
        self._window = window
        self._steps = steps
        self._idx = 0
        self._on_finish = on_finish

        # Full-window frameless overlay
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setGeometry(window.rect())
        self.setStyleSheet("background: transparent;")
        self.raise_()

        # Step card
        self._card = QWidget(self)
        self._card.setObjectName("TourCard")
        self._card.setStyleSheet(
            "QWidget#TourCard {"
            "  background: #0d0e10;"
            "  border: 1px solid #4A9EFF;"
            "  border-radius: 10px;"
            "  padding: 20px;"
            "}"
        )
        self._card.setFixedWidth(420)

        col = QVBoxLayout(self._card)
        col.setContentsMargins(20, 20, 20, 20)
        col.setSpacing(8)

        from .i18n import tr
        from .theme import font_px
        # THE STEP COUNTER IS COMPOSED FROM A TEMPLATE, so the catalog is
        # asked for a key that exists rather than for the numbers baked in.
        self._step_lbl = QLabel(tr("Step {n} / {total}", n=1,
                                   total=len(steps)))
        self._step_lbl.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            f"font-weight: 600; font-size: {font_px(10)}px;"
            "letter-spacing: 2px; color: #4A9EFF;"
        )
        col.addWidget(self._step_lbl)

        self._title_lbl = QLabel(tr(steps[0].title))
        self._title_lbl.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            f"font-weight: 400; font-size: {font_px(20)}px; color: #e5e5e5;"
        )
        col.addWidget(self._title_lbl)

        self._body_lbl = QLabel(tr(steps[0].body))
        self._body_lbl.setWordWrap(True)
        self._body_lbl.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            f"font-weight: 300; font-size: {font_px(13)}px;"
            "color: #a1a6ad;"
        )
        col.addWidget(self._body_lbl)

        # Buttons
        btn_row = QWidget()
        from PySide6.QtWidgets import QHBoxLayout
        row = QHBoxLayout(btn_row)
        row.setContentsMargins(0, 8, 0, 0)
        row.setSpacing(8)

        self._skip_btn = QPushButton(tr("Skip"))
        self._skip_btn.setStyleSheet(_ghost_btn_qss())
        self._skip_btn.clicked.connect(self._skip)
        row.addWidget(self._skip_btn)

        row.addStretch(1)

        self._next_btn = QPushButton(tr("Next"))
        self._next_btn.setStyleSheet(_primary_btn_qss())
        self._next_btn.clicked.connect(self._next)
        row.addWidget(self._next_btn)
        col.addWidget(btn_row)

        self._update_card_position()
        self._card.show()
        window.installEventFilter(self)

    # -- painting -----------------------------------------------------
    def paintEvent(self, event) -> None:
        """Dim the window and cut a lit ring around this step's target.

        The dimming is CLEARED inside the ring rather than merely outlined, so
        the widget being pointed at is seen in its own colours -- a highlight
        that leaves its subject dimmed points at something the viewer still
        cannot read. A target that cannot be resolved leaves the dim intact
        rather than failing: a tour missing one ring is better than no tour.

        :param event: the paint event.
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Dim overlay
        p.fillRect(self.rect(), QColor(0, 0, 0, 170))

        # Cut a hole around the highlighted widget, if any
        highlight_fn = self._steps[self._idx].highlight
        if highlight_fn is not None:
            try:
                target = highlight_fn(self._window)
                if target is not None:
                    rect = _widget_rect_in_window(target, self._window)
                    if rect is not None:
                        # Draw a bright ring around it
                        p.setBrush(Qt.transparent)
                        pen = QPen(QColor("#4A9EFF"), 3)
                        p.setPen(pen)
                        expanded = rect.adjusted(-4, -4, 4, 4)
                        p.drawRoundedRect(expanded, 6, 6)
                        # Clear the dimming inside the ring so users
                        # see the widget in its natural colour.
                        p.setCompositionMode(
                            QPainter.CompositionMode_Clear)
                        p.fillRect(rect, Qt.transparent)
            except Exception:
                pass
        p.end()

    def resizeEvent(self, event) -> None:
        """Keep the caption card in place when the overlay resizes.

        :param event: the resize event.
        """
        self._update_card_position()

    def _update_card_position(self) -> None:
        # Bottom-centre
        """Keep the card bottom-centre as the overlay resizes."""
        w = self.width()
        h = self.height()
        cw = self._card.width()
        ch = self._card.sizeHint().height()
        self._card.setGeometry(
            (w - cw) // 2, h - ch - 60, cw, ch,
        )

    # -- events -------------------------------------------------------
    def eventFilter(self, obj, event):
        """Follow the window's size, so the overlay always covers it.

        :param obj: the object the event is for.
        :param event: the event.
        :returns: whatever the base filter returns -- the resize is observed,
            never consumed.
        """
        if obj is self._window and event.type() == QEvent.Resize:
            self.setGeometry(self._window.rect())
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Take Escape to skip the tour and Return to advance it.

        :param event: the key event.
        """
        if event.key() == Qt.Key_Escape:
            self._skip()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._next()
            return
        super().keyPressEvent(event)

    # -- lifecycle ----------------------------------------------------
    def _next(self) -> None:
        """Advance one step, finishing when the last one is past."""
        self._idx += 1
        if self._idx >= len(self._steps):
            self._finish()
            return
        from .i18n import tr

        step = self._steps[self._idx]
        self._step_lbl.setText(tr("Step {n} / {total}", n=self._idx + 1,
                                  total=len(self._steps)))
        self._title_lbl.setText(tr(step.title))
        self._body_lbl.setText(tr(step.body))
        if self._idx == len(self._steps) - 1:
            self._next_btn.setText(tr("Finish"))
        self._update_card_position()
        self.update()

    def _skip(self) -> None:
        """End the tour now. Same finish as reaching the last step.

        Skipping and completing are the SAME outcome deliberately: a tour that
        reappeared because it was dismissed rather than read is one the user
        cannot get rid of.
        """
        self._finish()

    def _finish(self) -> None:
        """Close the overlay and tell the caller it is done.

        A failing callback does not stop the overlay closing: the tour is
        finished either way, and leaving it on screen because something
        downstream raised is the worse of the two outcomes.
        """
        if self._on_finish is not None:
            try:
                self._on_finish()
            except Exception:
                LOG.debug("tour finish callback failed", exc_info=True)
        else:
            mark_tour_seen()
        self._window.removeEventFilter(self)
        self.close()
        self.deleteLater()


def _widget_rect_in_window(widget: QWidget,
                             window: QMainWindow) -> Optional[QRect]:
    """Return ``widget``'s bounding rectangle in the window's coord space."""
    try:
        top_left = widget.mapTo(window, QPoint(0, 0))
        return QRect(top_left, widget.size())
    except Exception:
        return None


def _ghost_btn_qss() -> str:
    """Return the stylesheet for the tour's secondary button.

    :returns: the QSS. Literal colours rather than the theme's, because the
        first-run tour is shown before a theme has been chosen.
    """
    return (
        "QPushButton {"
        "  background: transparent;"
        "  color: #a1a6ad;"
        "  border: 1px solid #2a2d33;"
        "  border-radius: 6px;"
        "  padding: 6px 14px;"
        "  font-family: 'Open Sans', sans-serif;"
        "}"
        "QPushButton:hover { color: #e5e5e5; border-color: #4A9EFF; }"
    )


def _primary_btn_qss() -> str:
    """Return the stylesheet for the tour's primary button.

    :returns: the QSS. Literal colours, for the same reason as the ghost
        button.
    """
    return (
        "QPushButton {"
        "  background: #4A9EFF;"
        "  color: #000;"
        "  border: none;"
        "  border-radius: 6px;"
        "  padding: 6px 18px;"
        "  font-family: 'Open Sans', sans-serif;"
        "  font-weight: 600;"
        "}"
        "QPushButton:hover { background: #66B2FF; }"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def maybe_show_tour(window: QMainWindow,
                      force: bool = False) -> Optional[_TourOverlay]:
    """Show the tour if it hasn't been seen (or if ``force=True``).

    :param window: the MainWindow to overlay.
    :param force: skip the "seen" check and show anyway.
    :returns: the overlay widget (already visible) or None if the
        tour was skipped because it had been seen.
    """
    if not force and was_tour_shown():
        return None
    overlay = _TourOverlay(window, DEFAULT_TOUR)
    overlay.show()
    overlay.raise_()
    overlay.setFocus()
    return overlay
