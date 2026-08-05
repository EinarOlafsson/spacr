"""Semantic styling and busy-state handling for action buttons.

Qt stylesheets cannot select a button by its visible text. spaCR also creates
many buttons dynamically through ``QDialogButtonBox``, so this application
event filter tags buttons when they appear rather than relying on every screen
to repeat styling code.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QPushButton


POSITIVE_PREFIXES = ("run", "propagate")
NEGATIVE_PREFIXES = (
    "stop", "close", "cancel", "abort", "delete", "remove", "clear",
    "discard", "reject", "reset", "quit", "terminate",
)

_FILTER_ATTRIBUTE = "_spacr_semantic_button_filter"
_WIRED_PROPERTY = "_spacrButtonRoleWired"
#: Latch so :func:`_adopt_activity_spinner` walks a button's parents once.
_SPINNER_PROPERTY = "_spacrActivitySpinnerChecked"


def _normalise(text: str) -> str:
    """Return visible button text in a comparison-friendly form."""
    return " ".join(
        str(text).replace("&", "").replace("…", " ").strip().casefold().split())


def action_role(text: str) -> Optional[str]:
    """Classify a visible label as ``positive``, ``negative``, or neutral."""
    normalised = _normalise(text)
    if any(normalised == word or normalised.startswith(f"{word} ")
           for word in POSITIVE_PREFIXES):
        return "positive"
    if any(normalised == word or normalised.startswith(f"{word} ")
           for word in NEGATIVE_PREFIXES):
        return "negative"
    return None


def _repolish(button: QPushButton) -> None:
    style = button.style()
    if style is not None:
        style.unpolish(button)
        style.polish(button)
    button.update()


def set_button_busy(button: QPushButton, busy: bool) -> None:
    """Show or clear the persistent solid operation state."""
    busy = bool(busy)
    if button.property("buttonActionBusy") == busy:
        return
    button.setProperty("buttonActionBusy", busy)
    _repolish(button)


def _adopt_activity_spinner(button: QPushButton) -> None:
    """Give the *Clear console* button its background-activity indicator.

    The spinner belongs immediately to the right of that one button, and
    this filter is already the application's single place for "do something
    to a button wherever it appears" -- which is why the hook lives here
    rather than in the screen that builds the row. Every module screen builds
    its own actions row from the same code, so one hook covers all of them,
    and a screen that has no such button is simply never matched.

    Identification is by **object identity** against the owning screen's
    ``_btn_clear`` attribute, not by button text: ``retranslate_widget_tree``
    runs over each screen as it opens, so in any non-English locale the text
    is not "Clear console" by the time this filter sees it.

    The latch below is set **on success only**, and that is not a detail.
    ``AppScreen`` builds its actions row as a parentless ``QWidget`` and
    reparents it when it is added to a layout, so the button's first Polish
    arrives while the walk cannot yet reach the screen. Latching on the
    first attempt meant the walk never ran again once the tree was complete,
    and the spinner was never installed -- which is exactly what happened,
    and is why this comment exists. Retrying is cheap: at most eight
    ``getattr`` calls, and only until it finds the one button it is looking
    for.
    """
    if button.property(_SPINNER_PROPERTY):
        return
    host = button.parentWidget()
    depth = 0
    while host is not None and depth < 8:
        if getattr(host, "_btn_clear", None) is button:
            from .widgets.activity_spinner import attach_activity_spinner
            if attach_activity_spinner(host) is not None:
                button.setProperty(_SPINNER_PROPERTY, True)
            return
        host = host.parentWidget()
        depth += 1


class _SemanticButtonFilter(QObject):
    """Tag buttons globally and preserve Run's fill until work completes."""

    def classify(self, button: QPushButton) -> None:
        # spaCR's dialog buttons are text, not text-plus-glyph. Qt's platform
        # styles put a standard icon on the standard roles — a cross on
        # Cancel and Close, a downward arrow on Save — which reads as system
        # chrome dropped into the app's own type. Stripped here rather than at
        # each call site, because this filter already sees every button in
        # every dialog, including ones built after startup.
        if not button.icon().isNull():
            button.setIcon(QIcon())

        role = action_role(button.text())
        if role is None:
            if button.objectName() == "PrimaryButton":
                role = "positive"
            elif button.objectName() == "DangerButton":
                role = "negative"
        if role is None:
            return

        changed = button.property("buttonActionRole") != role
        button.setProperty("buttonActionRole", role)
        if button.property("buttonActionBusy") is None:
            button.setProperty("buttonActionBusy", False)
            changed = True
        if not bool(button.property(_WIRED_PROPERTY)):
            button.setProperty(_WIRED_PROPERTY, True)
            button.pressed.connect(
                lambda target=button: set_button_busy(target, True))
            button.clicked.connect(
                lambda _checked=False, target=button:
                self._after_clicked(target))
        if changed:
            _repolish(button)

    def _after_clicked(self, button: QPushButton) -> None:
        QTimer.singleShot(
            0, lambda target=button: self._settle_after_handler(target))

    @staticmethod
    def _settle_after_handler(button: QPushButton) -> None:
        try:
            # Disabled Run/Stop buttons conventionally mean their asynchronous
            # worker is still starting or stopping. Keep the solid operation
            # fill until the owning screen re-enables/clears the button.
            running = (
                _normalise(button.text()).startswith(("run", "stop"))
                and not button.isEnabled()
            )
            if not running:
                set_button_busy(button, False)
        except RuntimeError:
            # Close can delete its dialog before this queued callback runs.
            pass

    def eventFilter(self, watched, event):  # noqa: N802 (Qt naming)
        if isinstance(watched, QPushButton):
            event_type = event.type()
            if event_type in (
                    QEvent.Show, QEvent.Polish, QEvent.ParentChange):
                self.classify(watched)
                _adopt_activity_spinner(watched)
            elif event_type == QEvent.EnabledChange:
                self.classify(watched)
                if (watched.isEnabled()
                        and watched.property("buttonActionBusy") is True):
                    set_button_busy(watched, False)
        return False


def install_button_roles(app=None) -> None:
    """Install semantic action-button behavior on a running application."""
    app = app or QApplication.instance()
    if app is None:
        return
    event_filter = getattr(app, _FILTER_ATTRIBUTE, None)
    if event_filter is None:
        # QApplication is always a QObject in production. Keeping the parent
        # optional also supports lightweight application adapters used by
        # embedding hosts and tests; the attribute below retains the filter.
        parent = app if isinstance(app, QObject) else None
        event_filter = _SemanticButtonFilter(parent)
        app.installEventFilter(event_filter)
        setattr(app, _FILTER_ATTRIBUTE, event_filter)
    for widget in app.allWidgets():
        if isinstance(widget, QPushButton):
            event_filter.classify(widget)
