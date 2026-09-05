"""Answer "is this path there?" without ever blocking the interface.

WHY THIS EXISTS. `spacr/qt/widgets/file_list.py` asked `os.path.exists` for
every remembered path, on the GUI thread, to colour the missing ones red.
That is free for a local disk and it is not free for a network one: measured
on one workstation, ``os.path.exists`` on a path under
``/nas_mnt`` -- an ``autofs`` mount with ``timeout=600`` -- had not returned
after TWENTY SECONDS, because the stat is what triggers the automount and the
share was asleep.

A blocked GUI thread does not look like a slow path check. It looks like
this, and every one of these was reported as a separate defect on 2026-09-04:

  * "opening map barcodes crashes spacr" -- it is not a crash, it is a
    freeze with no traceback, which is why nothing reached the logs. The
    journal shows `automount request ... triggered by spacr` immediately
    before each force-quit.
  * "it kind of flickers ... several events are happening upon hover and
    they sometimes lag for a couple of seconds" -- hover events queue while
    the thread is stalled and then replay in a burst.
  * "i see millisecond glimmers of parts of the module screens on the home
    screen" -- repaints are deferred, so the stack shows what it has not
    been given time to paint over.
  * "it seems to happen ... after i have opened one or two moduals" -- each
    module adds its own remembered paths, so there is more to stat.

So the rule is absolute: NOTHING on the GUI thread may touch the
filesystem for a path the user supplied. This module is how that is kept.

HOW IT ANSWERS. From a cache, immediately. A path it has not seen is
reported as PRESENT and queued for a background check; when the answer
arrives the cache is updated and :data:`probes` emits, so a widget can
redraw. Optimistic on purpose: a path drawn as missing for a moment and
then corrected is a widget that cried wolf, while a path drawn as present
and then corrected is one that simply learned something.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Dict, Optional

from PySide6.QtCore import QObject, Signal

#: How long a worker waits for one stat before giving up on it. A path that
#: cannot answer in this long is not going to be useful to the user either,
#: and the answer it eventually gives is not worth a thread parked forever.
PROBE_TIMEOUT_S = 5.0

#: How many paths are probed at once. Small: these are stat calls, and the
#: point is to keep one slow mount from starving the rest, not throughput.
WORKERS = 4


class _Probes(QObject):
    """Emits when a background check has changed what :func:`exists` says."""

    #: The path whose answer just changed, and what it changed to.
    answered = Signal(str, bool)


#: The signal source. Connect to `probes.answered` to redraw when a path's
#: state is finally known.
probes = _Probes()

#: Keyed on ``(path, want_dir)``: a path can exist and not be a directory,
#: and the two questions have different answers and different callers.
_cache: Dict[tuple, bool] = {}
_pending: set = set()
_lock = threading.Lock()
_queue: "queue.Queue[tuple]" = queue.Queue()
_started = False


def _worker() -> None:
    """Stat one path at a time, forever, off the GUI thread."""
    while True:
        key = _queue.get()
        path, want_dir = key
        answer = _stat_with_timeout(path, want_dir)
        with _lock:
            changed = _cache.get(key) != answer
            _cache[key] = answer
            _pending.discard(key)
        if changed:
            # Queued to the GUI thread by Qt, because this is a worker.
            probes.answered.emit(path, answer)
        _queue.task_done()


def _stat_with_timeout(path: str, want_dir: bool = False) -> bool:
    """``os.path.exists(path)``, but bounded by :data:`PROBE_TIMEOUT_S`.

    A stat cannot be cancelled, so the timeout is on WAITING for it rather
    than on the call: the thread that made it stays parked until the kernel
    lets go, and this one stops waiting and reports the path as present.
    Reporting it present is the conservative answer -- it is what the widget
    already assumed, and it does not paint a path red on the strength of a
    mount being slow.
    """
    done = threading.Event()
    result = [True]

    def run() -> None:
        """Ask the filesystem the question that is allowed to block.

        This body is the reason the module exists: it runs on a worker, where
        an `autofs` mount taking twenty seconds to wake costs nobody a frozen
        window. `OSError` is an answer of "no", not a failure -- a path that
        cannot be stat-ed is a path the user cannot use either.
        """
        try:
            result[0] = (os.path.isdir(path) if want_dir
                         else os.path.exists(path))
        except OSError:
            result[0] = False
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True,
                              name=f"spacr-path-probe:{path[:40]}")
    thread.start()
    done.wait(PROBE_TIMEOUT_S)
    return result[0]


def _ensure_started() -> None:
    """Start the probe worker threads, once.

    Daemon threads, so a probe still blocked on a sleeping mount cannot hold
    the application open at exit -- which is the whole reason path checks
    were moved off the GUI thread in the first place.
    """
    global _started
    with _lock:
        if _started:
            return
        _started = True
    for index in range(WORKERS):
        threading.Thread(target=_worker, daemon=True,
                         name=f"spacr-path-probe-{index}").start()


def exists(path, *, default: bool = True,
           want_dir: bool = False) -> bool:
    """Whether ``path`` is there, answered from cache and never blocking.

    :param path: the path to ask about. Anything falsy is ``False``.
    :param default: what to say while the answer is unknown. ``True`` by
        default -- see the module docstring on why optimism is the right
        way round here.
    :param want_dir: ask ``isdir`` rather than ``exists``. Cached
        separately, because a path can exist and not be a directory.
    :returns: the cached answer, or ``default`` with a check queued.
    """
    text = str(path or "")
    if not text:
        return False
    key = (text, bool(want_dir))
    with _lock:
        if key in _cache:
            return _cache[key]
        if key in _pending:
            return default
        _pending.add(key)
    _ensure_started()
    _queue.put(key)
    return default


def isdir(path, *, default: bool = False) -> bool:
    """Whether ``path`` is a directory, answered from cache, never blocking.

    ``default`` is False here and True in :func:`exists`, and the asymmetry
    is deliberate: the callers of this one are choosing a folder to OPEN a
    dialog in, and opening it somewhere that turns out not to exist is worse
    than opening it at the default location.
    """
    return exists(path, default=default, want_dir=True)


def known(path, *, want_dir: bool = False) -> Optional[bool]:
    """The cached answer for ``path``, or ``None`` when it is not known yet."""
    with _lock:
        return _cache.get((str(path or ""), bool(want_dir)))


def forget(path=None) -> None:
    """Drop what is cached, so the next :func:`exists` asks again.

    :param path: one path, or ``None`` for all of them. Called when the
        user has just created or deleted something and the cache would
        otherwise keep answering with what was true before.
    """
    with _lock:
        if path is None:
            _cache.clear()
        else:
            text = str(path or "")
            _cache.pop((text, False), None)
            _cache.pop((text, True), None)


def prime(path, answer: bool) -> None:
    """Record an answer somebody already has, without a probe.

    The file dialog has just told us a path exists; asking the filesystem
    again would be a second stat for a fact already in hand.
    """
    text = str(path or "")
    if text:
        key = (text, False)
        with _lock:
            _cache[key] = bool(answer)
            _pending.discard(key)
