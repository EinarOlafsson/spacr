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


DEFAULT_TOUR: List[TourStep] = [
    TourStep(
        title="Welcome to spaCR",
        body="This quick 5-step tour will show you the home layout. "
             "Press Esc at any time to skip.",
        highlight=None,
    ),
    TourStep(
        title="Sidebar — apps by category",
        body="Every pipeline lives here, grouped into Core, Analysis, "
             "Cellpose, and Sequencing. Click any name to open it. "
             "Ctrl+1..9 jumps between them.",
        highlight=lambda w: getattr(w, "_sidebar", None),
    ),
    TourStep(
        title="Demos menu",
        body="Load a synthetic demo dataset for any module in one "
             "click — no data of your own required. Perfect for "
             "trying spaCR out.",
        highlight=lambda w: _find_menu(w, "Demos"),
    ),
    TourStep(
        title="Drag & drop",
        body="Drop a folder of microscopy images onto Mask (or "
             "Measure, Annotate, etc.) to point that module at it. "
             "spaCR auto-detects the filename regex and shows a "
             "sanity-check in the Console.",
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


def _find_menu(window: QMainWindow, title: str) -> Optional[QWidget]:
    for act in window.menuBar().actions():
        if act.text().replace("&", "") == title:
            m = act.menu()
            if m is not None:
                return m
    return None


# ---------------------------------------------------------------------------
# Overlay widget
# ---------------------------------------------------------------------------

class _TourOverlay(QWidget):
    """Translucent overlay + step card. Owns the tour lifecycle."""

    def __init__(self, window: QMainWindow, steps: List[TourStep]):
        super().__init__(window)
        self._window = window
        self._steps = steps
        self._idx = 0

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

        self._step_lbl = QLabel("Step 1 / 5")
        self._step_lbl.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 600; font-size: 10px;"
            "letter-spacing: 2px; color: #4A9EFF;"
        )
        col.addWidget(self._step_lbl)

        self._title_lbl = QLabel(steps[0].title)
        self._title_lbl.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 400; font-size: 20px; color: #e5e5e5;"
        )
        col.addWidget(self._title_lbl)

        self._body_lbl = QLabel(steps[0].body)
        self._body_lbl.setWordWrap(True)
        self._body_lbl.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 300; font-size: 13px;"
            "color: #a1a6ad;"
        )
        col.addWidget(self._body_lbl)

        # Buttons
        btn_row = QWidget()
        from PySide6.QtWidgets import QHBoxLayout
        row = QHBoxLayout(btn_row)
        row.setContentsMargins(0, 8, 0, 0)
        row.setSpacing(8)

        self._skip_btn = QPushButton("Skip")
        self._skip_btn.setStyleSheet(_ghost_btn_qss())
        self._skip_btn.clicked.connect(self._skip)
        row.addWidget(self._skip_btn)

        row.addStretch(1)

        self._next_btn = QPushButton("Next")
        self._next_btn.setStyleSheet(_primary_btn_qss())
        self._next_btn.clicked.connect(self._next)
        row.addWidget(self._next_btn)
        col.addWidget(btn_row)

        self._update_card_position()
        self._card.show()
        window.installEventFilter(self)

    # -- painting -----------------------------------------------------
    def paintEvent(self, event) -> None:
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
        self._update_card_position()

    def _update_card_position(self) -> None:
        # Bottom-centre
        w = self.width()
        h = self.height()
        cw = self._card.width()
        ch = self._card.sizeHint().height()
        self._card.setGeometry(
            (w - cw) // 2, h - ch - 60, cw, ch,
        )

    # -- events -------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self._window and event.type() == QEvent.Resize:
            self.setGeometry(self._window.rect())
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self._skip()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._next()
            return
        super().keyPressEvent(event)

    # -- lifecycle ----------------------------------------------------
    def _next(self) -> None:
        self._idx += 1
        if self._idx >= len(self._steps):
            self._finish()
            return
        step = self._steps[self._idx]
        self._step_lbl.setText(f"Step {self._idx + 1} / {len(self._steps)}")
        self._title_lbl.setText(step.title)
        self._body_lbl.setText(step.body)
        if self._idx == len(self._steps) - 1:
            self._next_btn.setText("Finish")
        self._update_card_position()
        self.update()

    def _skip(self) -> None:
        self._finish()

    def _finish(self) -> None:
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
