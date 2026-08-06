"""Quitting spaCR, and quitting one run, when asking nicely is not enough.

``RunRegistry.cancel_all`` is cooperative: it sets a flag, interrupts the
thread, and waits. That is the right default -- a pipeline killed mid-write
leaves a half-written ``.npy`` or a database row with no rows behind it, and
recovering from that costs more than waiting. But cooperative cancellation
has one failure mode with no way out from inside the application: a worker
that is wedged in a C extension never checks the flag, so it never stops,
and ``closeEvent`` refuses to close for as long as it lives.

That is the state this module exists for. A user watching a run that will
never finish, in a window that will not close, whose only remaining option
is to find the process ID from a terminal the desktop entry never opened.

Two things follow from that:

* **Force is always offered, never taken silently.** Every entry point here
  asks first, and the prompt says what force costs, because the caller is
  the only one who knows whether the artefact being written matters.
* **A graceful attempt is not a commitment to wait forever.** Choosing to
  wait starts a watcher that comes back every :data:`RECHECK_MS`, so the
  answer "give it another five minutes" can be given repeatedly and the
  question never has to be remembered.
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


def ask_how_to_quit(parent: Optional[QWidget], *, what: str,
                    detail: str = "", verb: str = "Quit") -> str:
    """Ask whether to stop cooperatively or to kill.

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
    :returns: :data:`GRACEFUL`, :data:`FORCE` or :data:`CANCEL`.

    Cancel is the default button and the escape action. Force quit is
    reachable in one click but is never what a stray Return key does.
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
    graceful = box.addButton("Finish current work", QMessageBox.AcceptRole)
    force = box.addButton(f"Force {verb.lower()}", QMessageBox.DestructiveRole)
    cancel = box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    # Red, because it is the one that loses data. The role alone does not
    # colour it on every style.
    force.setObjectName("DangerButton")
    box.exec()

    clicked = box.clickedButton()
    if clicked is graceful:
        return GRACEFUL
    if clicked is force:
        return FORCE
    return CANCEL


def force_quit_now(exit_code: int = 1) -> None:
    """Leave immediately, without unwinding anything.

    ``os._exit`` rather than ``sys.exit`` or ``QApplication.quit``, and the
    difference is the entire point: both of those unwind: they run atexit
    handlers, Python finalisation and Qt's own teardown, and every one of
    those can block on the very thread that is already wedged. A force quit
    that can hang is not a force quit.

    Logs are flushed first because the reason a run wedged is usually in
    them, and this is the one exit path that will not flush them itself.
    """
    LOG.warning("Force quit requested; leaving without cleanup")
    for handler in list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:  # pragma: no cover - a broken sink must not block
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # pragma: no cover
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
        f"background: {colour}; color: #ffffff; }}"
    )
