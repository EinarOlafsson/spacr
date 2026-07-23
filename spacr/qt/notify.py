"""
Cross-platform desktop notifications.

Used to fire an OS-level notification when a long pipeline finishes,
so users don't have to sit and watch a progress bar. Fails silently
on any error — a missing notification is never worth crashing over.

Backends, in preference order:

* Linux — ``notify-send`` (libnotify)
* macOS — ``osascript -e 'display notification …'``
* Windows — win32 ``ToastNotifier`` if available, else no-op

Also exposes an in-app fallback via a Qt system-tray message so
users who disabled OS notifications still get a subtle badge.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from typing import Optional

LOG = logging.getLogger("spacr.qt.notify")


def notify(title: str, body: str = "",
             app_name: str = "spaCR") -> bool:
    """Best-effort OS notification.

    :param title: short headline.
    :param body: optional longer body text.
    :param app_name: sender name shown alongside the notification.
    :returns: True iff a backend accepted the request.
    """
    system = platform.system()
    try:
        if system == "Linux" and shutil.which("notify-send"):
            subprocess.run(
                ["notify-send", "-a", app_name, title, body],
                check=False, timeout=3,
            )
            return True
        if system == "Darwin":
            script = (
                f'display notification "{_esc(body)}" '
                f'with title "{_esc(app_name)}" '
                f'subtitle "{_esc(title)}"'
            )
            subprocess.run(["osascript", "-e", script],
                            check=False, timeout=3)
            return True
        if system == "Windows":
            try:
                from win10toast import ToastNotifier    # type: ignore
                ToastNotifier().show_toast(
                    title, body, duration=6, threaded=True,
                )
                return True
            except Exception:
                return False
    except Exception as e:
        LOG.debug("notify failed: %s", e)
        return False
    return False


def _esc(s: str) -> str:
    """Escape a string for AppleScript embedding."""
    return (s or "").replace('"', r'\"')


def notify_tray(title: str, body: str = "",
                 icon: Optional[str] = None) -> bool:
    """In-app fallback via ``QSystemTrayIcon``.

    Called by the pipeline runner when :func:`notify` fails. Requires
    a running ``QApplication`` — safe no-op headless.
    """
    try:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication, QSystemTrayIcon
        app = QApplication.instance()
        if app is None:
            return False
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return False
        tray = QSystemTrayIcon(QIcon(icon or ""), parent=app)
        tray.show()
        tray.showMessage(title, body, QSystemTrayIcon.Information, 6000)
        return True
    except Exception as e:
        LOG.debug("tray notify failed: %s", e)
        return False


def announce_pipeline_finished(app_key: str, status: str,
                                 elapsed_s: float) -> None:
    """Convenience wrapper: notify the user a pipeline finished.

    Called from the Qt runtime when a pipeline worker emits its
    finished signal.

    :param app_key: id of the pipeline app (``"mask"`` / …).
    :param status: ``"success"`` / ``"failed"`` / ``"cancelled"``.
    :param elapsed_s: wall-clock seconds the run took.
    """
    icon = "✓" if status == "success" else "⚠"
    title = f"{icon} spaCR — {app_key} {status}"
    body = f"Finished in {elapsed_s:.1f}s."
    if not notify(title, body):
        notify_tray(title, body)
