"""Notice that spaCR keeps dying on launch, and start without the part that kills it.

The crash log records `Fatal Python error: Segmentation fault` and `Aborted`
with `<no Python frame>` on the crashing thread -- the fault is in native
code, in Qt's render thread or the GL driver, where no Python stack exists to
report and no `except` can run. The animated backdrop and its optional GL
canvas are the only things spaCR asks a driver to do at startup.

A user cannot act on that. What they see is an application that will not
open, and the setting that would turn the backdrop off is behind the window
that never appears. `safespacr` is the deliberate way in; this is the
automatic one, for the user who does not know it exists.

HOW IT KNOWS: a marker file is written when a launch begins and removed when
one shuts down cleanly. Finding it already there means the last run died
without shutting down. Two of those in a row is treated as a pattern rather
than an accident, and the next start is made without the backdrop.
"""
from __future__ import annotations

import logging
import os
from typing import Optional
from ..logging_util import _spacr_home

LOG = logging.getLogger("spacr.qt.crash_recovery")

#: How many unclean exits in a row before the backdrop is dropped.
#:
#: TWO, NOT ONE. A single unclean exit is as likely to be a machine going to
#: sleep, a `kill -9`, or the user closing a laptop lid as it is a crash --
#: and turning the interface's appearance off because somebody rebooted
#: would be its own defect. Two in a row is a pattern.
CRASHES_BEFORE_DROPPING_THE_BACKDROP = 2

_MARKER = "running.marker"
_COUNTER = "unclean-exits"


def _folder() -> str:
    """Where the markers live. Beside the logs, created on demand."""
    try:
        from ..logging_util import log_dir

        folder = log_dir()
    except Exception:                                        # noqa: BLE001
        folder = ""
    if not folder:
        folder = os.path.join(str(_spacr_home()), "logs")
    os.makedirs(folder, exist_ok=True)
    return folder


def _read_counter() -> int:
    """Read how many unclean exits have been recorded.

    :returns: the count, and ``0`` for a missing or unreadable file -- a
        counter that cannot be read is not evidence of a crash.
    """
    try:
        with open(os.path.join(_folder(), _COUNTER)) as handle:
            return max(0, int(handle.read().strip() or 0))
    except Exception:                                        # noqa: BLE001
        return 0


def _write_counter(value: int) -> None:
    """Record the unclean-exit count.

    :param value: the new count, floored at zero. A failure is logged and
        swallowed: this runs during startup and shutdown, where refusing to
        continue over a bookkeeping file would be worse than losing it.
    """
    try:
        with open(os.path.join(_folder(), _COUNTER), "w") as handle:
            handle.write(str(max(0, int(value))))
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not record the unclean-exit count", exc_info=True)


def note_that_a_launch_began() -> int:
    """Record that a launch started, and count how many died before it.

    :returns: the number of consecutive unclean exits, this launch's
        predecessor included.

    Call once, early, before the backdrop is built. It also takes the
    per-instance lock: a marker left by a window that is still running is
    not a crash, so a second window neither counts it nor overwrites it.
    """
    marker = os.path.join(_folder(), _MARKER)
    unclean = _read_counter()
    if _claim_the_instance():
        return unclean
    if os.path.isfile(marker):
        unclean += 1
        _write_counter(unclean)
    elif os.path.exists(marker):
        LOG.warning(
            "crash detection is disabled: %s exists but is not a file, so "
            "spaCR can neither write nor clear its running marker", marker)
    try:
        with open(marker, "w") as handle:
            handle.write(str(os.getpid()))
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not write the running marker", exc_info=True)
    return unclean


def note_a_clean_shutdown() -> None:
    """Record that this run ended properly, clearing the count.

    THE COUNT RESETS RATHER THAN DECREMENTS. The question is "is spaCR
    crashing right now", not "how many times has it ever crashed", and a
    total that only ever grew would eventually disable the backdrop on a
    machine where it works. Every instance and project lock this window
    holds is released first.
    """
    _release_all_locks()
    if _OTHER_INSTANCE:
        return
    try:
        os.remove(os.path.join(_folder(), _MARKER))
    except FileNotFoundError:
        pass
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not remove the running marker", exc_info=True)
    _write_counter(0)


def should_start_without_the_backdrop(unclean: Optional[int] = None) -> bool:
    """Whether this launch should skip the backdrop and any GL.

    :param unclean: the count from :func:`note_that_a_launch_began`; read
        from disk when omitted.
    :returns: ``True`` when spaCR has died repeatedly without shutting down.
    """
    count = _read_counter() if unclean is None else unclean
    return count >= CRASHES_BEFORE_DROPPING_THE_BACKDROP


def take_the_backdrop_out_of_this_launch() -> None:
    """Turn off, for this process only, everything that asks a driver to draw.

    NOT A SAVED PREFERENCE. The user did not choose this and must not have
    to undo it: the next clean run clears the count and the backdrop comes
    back on its own. Writing it to the store would turn a diagnosis into a
    setting the user never made and cannot explain.
    """
    os.environ["SPACR_NO_GL"] = "1"
    os.environ["SPACR_NO_BACKDROP"] = "1"


_INSTANCE_LOCK_NAME = "instance.lock"

_HELD_LOCKS: dict = {}

_PROJECT_CHOICES: dict = {}

_OTHER_INSTANCE: dict = {}


def _locks_folder() -> str:
    """Where the instance and project lock files live, created on demand.

    One folder in the user's spaCR directory rather than a file inside each
    project, so a read-only data share can still be protected and nothing
    is ever written beside the user's images. ``SPACR_LOCK_DIR`` moves it.
    """
    folder = (os.environ.get("SPACR_LOCK_DIR")
              or os.path.join(os.path.expanduser("~"), ".spacr", "locks"))
    os.makedirs(folder, exist_ok=True)
    return folder


def _lock_file_for(path) -> str:
    """The lock file that guards one project folder, queue or database.

    :param path: the folder or file being protected.
    :returns: a path in :func:`_locks_folder` named after a hash of the
        normalised absolute path, so two spellings of one folder share a
        lock and case-insensitive file systems compare as they should.
    """
    import hashlib

    key = os.path.normcase(os.path.realpath(os.path.expanduser(str(path))))
    digest = hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:20]
    return os.path.join(_locks_folder(), f"project-{digest}.lock")


def _try_lock(lock_path: str):
    """Take a lock file for this process, clearing a stale one first.

    A lock is stale when the process that wrote it is no longer running on
    this machine: Qt's own check, with the age limit switched off so a
    run that lasts all weekend is never mistaken for a dead one.

    :param lock_path: the lock file.
    :returns: ``(True, None)`` when this process now holds it, or
        ``(False, holder)`` with ``holder`` a dict of ``pid``, ``host`` and
        ``app`` for the other process (empty when unreadable).
    """
    from PySide6.QtCore import QLockFile

    held = _HELD_LOCKS.get(lock_path)
    if held is not None:
        return True, None
    lock = QLockFile(lock_path)
    lock.setStaleLockTime(0)
    if lock.tryLock(0):
        _HELD_LOCKS[lock_path] = lock
        return True, None
    holder: dict = {}
    try:
        info = lock.getLockInfo()
    except Exception:                                        # noqa: BLE001
        info = None
    if isinstance(info, tuple) and len(info) >= 3:
        offset = 1 if isinstance(info[0], bool) else 0
        try:
            holder = {"pid": int(info[offset]), "host": str(info[offset + 1]),
                      "app": str(info[offset + 2])}
        except (TypeError, ValueError, IndexError):
            holder = {}
    import socket

    if (holder.get("pid") == os.getpid()
            and str(holder.get("host", "")).casefold()
            == socket.gethostname().casefold()):
        return True, None
    return False, holder


def _release_all_locks() -> None:
    """Give back every lock this process holds. Safe to call twice."""
    for path, lock in list(_HELD_LOCKS.items()):
        try:
            lock.unlock()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not release %s", path, exc_info=True)
        _HELD_LOCKS.pop(path, None)
    _PROJECT_CHOICES.clear()


def _claim_the_instance() -> dict:
    """Take the per-instance lock, and report another spaCR already running.

    The first window holds the instance lock for as long as it runs. A
    second window cannot take it and gets the first one's process details
    back instead; it keeps working, and its projects are then protected by
    the per-project locks.

    :returns: an empty dict when this is the only spaCR window, otherwise
        the ``pid``, ``host`` and ``app`` of the one already running.
    """
    _OTHER_INSTANCE.clear()
    try:
        ok, holder = _try_lock(
            os.path.join(_locks_folder(), _INSTANCE_LOCK_NAME))
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not take the instance lock", exc_info=True)
        return {}
    if not ok:
        _OTHER_INSTANCE.update(holder or {"pid": 0})
        LOG.warning("another spaCR window is already running (pid %s)",
                    _OTHER_INSTANCE.get("pid"))
    return dict(_OTHER_INSTANCE)


def _ask_about_a_locked_project(parent, path, holder) -> str:
    """Warn that another window has this project open, and ask what to do.

    :param parent: the widget the dialog belongs to.
    :param path: the folder, queue or database that is locked.
    :param holder: the other process's ``pid``, ``host`` and ``app``.
    :returns: ``"read_only"``, ``"continue"`` or ``"cancel"``.
    """
    from PySide6.QtWidgets import QMessageBox

    from .i18n import tr

    box = QMessageBox(parent)
    box.setObjectName("ProjectLockedDialog")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(tr("Project open in another window"))
    box.setText(tr(
        "Another spaCR window (process {pid} on {host}) is using {path}.",
        pid=holder.get("pid", "?"), host=holder.get("host") or "?",
        path=str(path)))
    box.setInformativeText(tr(
        "Two windows writing the same queue, measurements.db or results "
        "folder can corrupt them. Open it read-only to look without "
        "writing, or continue if you are sure the other window is idle."))
    read_only = box.addButton(tr("Open read-only"),
                              QMessageBox.ButtonRole.AcceptRole)
    carry_on = box.addButton(tr("Continue anyway"),
                             QMessageBox.ButtonRole.DestructiveRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(read_only)
    box.exec()
    clicked = box.clickedButton()
    if clicked is read_only:
        return "read_only"
    if clicked is carry_on:
        return "continue"
    return "cancel"


def _claim_project(parent, path) -> str:
    """Lock a project folder, queue or database for this window.

    The answer to a locked project is remembered for the rest of the
    session, so the user is asked once per project rather than on every run.

    :param parent: the widget a warning dialog belongs to.
    :param path: what is about to be opened or written.
    :returns: ``"locked"`` when this window holds the lock (or there is no
        path to protect), else the user's answer: ``"read_only"``,
        ``"continue"`` or ``"cancel"``.
    """
    if not path:
        return "locked"
    try:
        lock_path = _lock_file_for(path)
        ok, holder = _try_lock(lock_path)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not lock %s", path, exc_info=True)
        return "locked"
    if ok:
        _PROJECT_CHOICES.pop(lock_path, None)
        return "locked"
    remembered = _PROJECT_CHOICES.get(lock_path)
    if remembered in ("read_only", "continue"):
        return remembered
    answer = _ask_about_a_locked_project(parent, path, holder or {})
    if answer in ("read_only", "continue"):
        _PROJECT_CHOICES[lock_path] = answer
    return answer
