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
        folder = os.path.join(os.path.expanduser("~"), ".spacr", "logs")
    os.makedirs(folder, exist_ok=True)
    return folder


def _read_counter() -> int:
    try:
        with open(os.path.join(_folder(), _COUNTER)) as handle:
            return max(0, int(handle.read().strip() or 0))
    except Exception:                                        # noqa: BLE001
        return 0


def _write_counter(value: int) -> None:
    try:
        with open(os.path.join(_folder(), _COUNTER), "w") as handle:
            handle.write(str(max(0, int(value))))
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not record the unclean-exit count", exc_info=True)


def note_that_a_launch_began() -> int:
    """Record that a launch started, and count how many died before it.

    :returns: the number of consecutive unclean exits, this launch's
        predecessor included.

    Call once, early, before the backdrop is built.
    """
    marker = os.path.join(_folder(), _MARKER)
    unclean = _read_counter()
    if os.path.exists(marker):
        # The last run wrote this and never removed it.
        unclean += 1
        _write_counter(unclean)
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
    machine where it works.
    """
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
    # ONLY THE BACKDROP. Reading every preference as its default -- what
    # safe mode does -- would be the right response to "a saved setting is
    # killing it" and the wrong one here: the evidence points at the
    # backdrop, and silently resetting the user's language, theme and paths
    # to diagnose a driver crash is a bigger surprise than the crash.
    os.environ["SPACR_NO_GL"] = "1"
    os.environ["SPACR_NO_BACKDROP"] = "1"
