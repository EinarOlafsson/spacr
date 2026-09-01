"""Provide graceful-stop, force-quit, and force-restart controls for Qt.

Cooperative cancellation remains the default because it lets active writes
finish safely. When a worker cannot respond, this module presents the user
with the consequences of stopping immediately, supports a verified restart
record, and can repeat the choice after :data:`RECHECK_MS`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Callable, Iterable, Optional

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

LOG = logging.getLogger(__name__)

#: How long a graceful attempt runs before the question is asked again.
#: Five minutes, from the report: "if graceful quitting is not finished in
#: 5 min they get prompted again to force quit. this happens every 5 min."
RECHECK_MS = 5 * 60 * 1000

#: What :func:`ask_how_to_quit` returns.
GRACEFUL = "graceful"
FORCE = "force"
CANCEL = "cancel"
#: Save the current module and settings before starting a fresh process.
RESTART = "restart"


def ask_how_to_quit(parent: Optional[QWidget], *, what: str,
                    detail: str = "", verb: str = "Quit",
                    offer_restart: bool = False,
                    restart_detail: str = "") -> str:
    """Ask whether to stop cooperatively or to kill.

    :param parent: widget that owns and centres the modal question, or
        ``None`` for an application-level dialog.
    :param what: what is being quit, in the user's words -- "spaCR" or the
        name of a module. It is used in the sentence, so it reads as a
        noun: "Quit Mask Generation?".
    :param verb: the word for what is about to happen. "Quit" for the
        application and the Home banner, "Stop" for a module's Stop button --
        a button labelled Stop that opens a dialog headed "Quit" reads as
        the wrong dialog, and a user who thinks they have mis-clicked
        cancels out of the thing they wanted.
    :param detail: appended under the question. Callers use it to name what
        is still running, because "something is still running" is not
        enough information to choose with.
    :param offer_restart: add a Force restart button. This option is disabled
        by default for ordinary quit dialogs.
    :param restart_detail: what Force restart will cost, from
        :func:`spacr.restart_state.warning_text`. Shown only when the button
        is offered, and REQUIRED to be meaningful when it is -- a button whose
        consequences are not on screen beside it is one people press once.
    :returns: :data:`GRACEFUL`, :data:`FORCE`, :data:`RESTART` or
        :data:`CANCEL`.

    Cancel is the default button and the escape action. Force quit is
    reachable in one click but is never what a stray Return key does, and
    Force restart is LAST because it is the most destructive thing on the
    dialog.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(f"{verb} {what}")
    box.setText(f"{verb} {what}?")
    box.setInformativeText(
        (detail + "\n\n" if detail else "")
        + "Finish current work stops when the running steps reach a point "
          "they can be interrupted safely. This can take minutes, and you "
          "will be asked again every five minutes.\n\n"
          "Force quit stops immediately. Anything being written right now "
          "is left half-written."
    )
    if offer_restart:
        box.setInformativeText(
            box.informativeText()
            + "\n\nForce restart saves this module and its settings, closes "
              "spaCR, starts it again and reopens the module where you left "
              "it. Use it when Force stop does not stop.\n\n"
            + (restart_detail or ""))
    graceful = box.addButton("Finish current work", QMessageBox.AcceptRole)
    force = box.addButton(f"Force {verb.lower()}", QMessageBox.DestructiveRole)
    restart = (box.addButton("Force restart", QMessageBox.DestructiveRole)
               if offer_restart else None)
    cancel = box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    # Red, because it is the one that loses data. The role alone does not
    # colour it on every style.
    force.setObjectName("DangerButton")
    if restart is not None:
        restart.setObjectName("DangerButton")
    box.exec()

    clicked = box.clickedButton()
    if clicked is graceful:
        return GRACEFUL
    if restart is not None and clicked is restart:
        return RESTART
    if clicked is force:
        return FORCE
    return CANCEL


def restart_spacr(module: str, settings=None, *, running=(), run_folders=(),
                  launcher=None, exiter=None) -> bool:
    """Save the current state and restart spaCR in a new process.

    The restart is cancelled if the state cannot be written and verified.

    :param module: key of the module to reopen.
    :param settings: module settings to restore after launch.
    :param running: active-run records for the restart summary.
    :param run_folders: paths that may contain interrupted-run output.
    :param launcher: optional process-launch function. Defaults to a detached
        :class:`subprocess.Popen` call.
    :param exiter: optional exit function. Defaults to
        :func:`force_quit_now`.
    :returns: ``True`` when the replacement process was started; ``False``
        when saving or launching failed and the current process remains open.
    """
    from ..restart_state import command, save

    if save(module=module, settings=settings, running=running,
            run_folders=run_folders) is None:
        LOG.error("the restart state could not be written; NOT restarting")
        return False

    started = command()
    try:
        if launcher is None:
            import subprocess

            # DETACHED. `start_new_session` puts the child in its own process
            # group, so the signal that takes this process down does not
            # follow it, and it survives the terminal that started us.
            subprocess.Popen(started, start_new_session=True,
                             close_fds=True)
        else:
            launcher(started)
    except Exception as exc:                          # noqa: BLE001
        # THE STATE IS LEFT ON DISK DELIBERATELY. spaCR did not restart, so
        # the user will start it themselves, and when they do they should
        # land back where they were.
        LOG.error("could not start spaCR again (%s); NOT quitting", exc)
        return False

    LOG.warning("restarting spaCR: %s", " ".join(started))
    (exiter or force_quit_now)(0)
    return True


def force_quit_now(exit_code: int = 1) -> None:
    """Flush available logs and terminate the process immediately.

    This function uses :func:`os._exit`, so Python finalizers, ``atexit``
    handlers, and Qt teardown do not run. Use it only after the user confirms
    a force quit.

    :param exit_code: process exit status.
    """
    LOG.warning("Force quit requested; leaving without cleanup")
    for handler in list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:
            # A BROKEN SINK MUST NOT BLOCK. This runs when a graceful
            # stop has already failed, so a handler that will not flush
            # cannot be what stops the process leaving -- a force quit
            # that hangs is the original complaint, twice.
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            # Same contract for stdout and stderr: a terminal that has
            # gone takes its flush with it.
            pass
    os._exit(exit_code)


class GracefulQuitWatcher(QObject):
    """Re-ask about force quitting while a graceful stop is still running.

    Owned by whoever started the graceful attempt. It holds no reference to
    the jobs themselves -- it calls ``still_running()`` each time, so a
    caller is free to retire handles underneath it.

    Stops asking as soon as ``still_running()`` reports False, and stops
    for good once force is chosen, so a user who is already leaving is not
    asked a second time on the way out.
    """

    def __init__(self, parent: Optional[QWidget],
                 still_running: Callable[[], bool],
                 *,
                 what: str,
                 describe: Optional[Callable[[], str]] = None,
                 on_force: Optional[Callable[[], None]] = None,
                 interval_ms: int = RECHECK_MS):
        super().__init__(parent)
        self._parent = parent
        self._still_running = still_running
        self._what = what
        self._describe = describe
        self._on_force = on_force or force_quit_now
        self._asking = False

        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._recheck)

    def start(self) -> None:
        """Begin the five-minute cycle, unless it is already finished."""
        if not self._still_running():
            return
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _recheck(self) -> None:
        """Ask again, once, if it is still going.

        The re-entrancy guard is not decoration: the prompt runs a nested
        event loop, the timer keeps firing inside it, and without this the
        user gets a second dialog stacked on the first every five minutes
        they spend reading the first one.
        """
        if self._asking:
            return
        if not self._still_running():
            self._timer.stop()
            return

        self._asking = True
        try:
            detail = self._describe() if self._describe else ""
            box = QMessageBox(self._parent)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(f"Still stopping {self._what}")
            box.setText(f"{self._what} is still finishing.")
            box.setInformativeText(
                (detail + "\n\n" if detail else "")
                + "Force quit stops it now and leaves anything being "
                  "written half-written. Otherwise it keeps going and you "
                  "will be asked again in five minutes."
            )
            force = box.addButton("Force quit", QMessageBox.DestructiveRole)
            wait = box.addButton("Keep waiting", QMessageBox.RejectRole)
            box.setDefaultButton(wait)
            box.setEscapeButton(wait)
            force.setObjectName("DangerButton")
            box.exec()
            if box.clickedButton() is force:
                self._timer.stop()
                self._on_force()
        finally:
            self._asking = False


def describe_active(handles: Iterable) -> str:
    """One line per running job, for the prompts above.

    "Something is still running" is not information anybody can decide
    with; the name of the module and how long it has been going is.
    """
    lines = []
    for handle in handles:
        name = getattr(handle, "app_key", None) or "job"
        try:
            minutes = max(0, int(handle.elapsed() // 60))
        except Exception:
            minutes = 0
        lines.append(f"  • {name} — running for {minutes} min")
    if not lines:
        return ""
    return "Still running:\n" + "\n".join(lines)


def style_as_danger(button: QPushButton, palette: Optional[dict] = None) -> None:
    """Paint ``button`` in the theme's danger colour.

    Scoped to the one button rather than added to the application sheet:
    this is the only red control on its row, and a global
    ``#DangerButton`` rule would be a new thing for every later screen to
    trip over.
    """
    from .theme import active_palette

    P = palette or active_palette()
    colour = P.get("danger") or P.get("error") or "#e5484d"
    # The ink on a FILLED danger surface is `bg`, not white. theme.py's
    # CONTRAST_RULES carries ("bg", "error", 4.5) with the comment "`bg` is
    # the ink on filled accent/danger surfaces: the selected menu row, a
    # pressed button, DangerButton on hover", and the application sheet's
    # own `#DangerButton:pressed` rule inks with `P["bg"]` for that reason.
    # This helper hard-coded `#ffffff` instead, which is only right on the
    # light theme: `error` is a PALE red on cell and glass, so white ink on
    # the hover fill measured 2.20:1 and 2.04:1 — below AA-large, on the
    # one control that force-quits a run. `bg` measures 6.23:1 (light)
    # through 9.55:1 (glass) and is guaranteed by the contrast rule above.
    ink = P.get("bg") or "#000000"
    # Key the rule on whatever the button is already called. Setting a
    # name here would take the caller's: `QuitSpacrButton` became
    # `DangerButton` and every lookup for it stopped finding anything --
    # a styling helper must not decide a widget's identity.
    name = button.objectName()
    if not name:
        name = "DangerButton"
        button.setObjectName(name)
    button.setProperty("spacrDanger", True)
    button.setStyleSheet(
        f"QPushButton#{name} {{"
        f"color: {colour};"
        f"border: 1px solid {colour};"
        "background: transparent; }"
        f"QPushButton#{name}:hover {{"
        f"background: {colour}; color: {ink}; }}"
    )
