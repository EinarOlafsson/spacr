"""Semantic styling and busy-state handling for action buttons.

Qt stylesheets cannot select a button by its visible text. spaCR also creates
many buttons dynamically through ``QDialogButtonBox``, so this application
event filter tags buttons when they appear rather than relying on every screen
to repeat styling code.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QApplication, QPushButton


POSITIVE_PREFIXES = ("run", "propagate")
NEGATIVE_PREFIXES = (
    "stop", "close", "cancel", "abort", "delete", "remove", "clear",
    "discard", "reject", "reset", "quit", "terminate",
)

_FILTER_ATTRIBUTE = "_spacr_semantic_button_filter"
_WIRED_PROPERTY = "_spacrButtonRoleWired"


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


class _SemanticButtonFilter(QObject):
    """Tag buttons globally and preserve Run's fill until work completes."""

    def classify(self, button: QPushButton) -> None:
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
        event_filter = _SemanticButtonFilter(app)
        app.installEventFilter(event_filter)
        setattr(app, _FILTER_ATTRIBUTE, event_filter)
    for widget in app.allWidgets():
        if isinstance(widget, QPushButton):
            event_filter.classify(widget)
