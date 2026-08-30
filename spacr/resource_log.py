"""What spaCR's process TREE costs while it runs, under its own setting.

WHY THIS IS NOT THE READINGS spaCR ALREADY TAKES.

Every resource figure in
the package counts the CALLING process: `spacr.fit_resources.host_rss` reads
``/proc/self/statm``, `spacr.qt.timing` reads its own resident size, and the
parameter sweep's floor reads the MACHINE's free memory, which cannot tell
spaCR's own children from another tenant on a shared box.

spaCR's heaviest
work does not happen in the calling process -- `spacr.sequencing` starts a
saver process and `spacr.parameter_sweep` runs every trial in a child -- so
the parent looks healthy right up to the moment the out-of-memory reaper
takes the run, and afterwards there is nothing to read.

This module sums the
process and every descendant, and names each one, so "which trial was large"
is a question the record can answer.

WHY IT IS NOT VERBOSE LOGGING. Verbose logging installs a profile hook that
fires on every call and every return, measured at twenty times the startup.
An account taken through it would describe the traced program rather than the
real one, which is exactly the program nobody wants measured. So this samples
instead: one psutil read a second, on a daemon thread that is never the GUI
thread, on an otherwise unperturbed run. Three states rather than a checkbox,
because the useful default is not "off" -- the most valuable resource data
comes from runs nobody expected to fail.

WHICH NUMBER IS RECORDED. USS where the platform gives it, then PSS, then
RSS -- and every record NAMES the measure it used, because RSS double-counts
the pages a fork shares and would overstate a sweep badly. A number whose
definition is unrecorded cannot be compared between two machines.

Nothing here is required to succeed. A child that exits between being
enumerated and being read is an expected outcome and not an error: that child
is skipped, the rest of the tree is kept, and the count of skipped readings
goes in the sample so the record says what it missed. A platform that cannot
supply a per-thread time records that it could not, never a zero, because a
zero reads as "this thread was free". Per-thread GPU memory is absent on
purpose: a CUDA context belongs to a process, so a per-thread figure would be
fiction.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

from .fit_resources import readable

__all__ = [
    "LEVELS",
    "DEFAULT_LEVEL",
    "ENV_VAR",
    "SOURCES",
    "MEASURES",
    "DEFAULT_INTERVAL_SECONDS",
    "DEFAULT_CAPACITY",
    "THREAD_NAME",
    "resolve_level",
    "level_source",
    "preferred_measure",
    "tree_sample",
    "summarise",
    "describe",
    "read_log",
    "ResourceSampler",
]

LOG = logging.getLogger("spacr.resource_log")

#: The three states. A checkbox would have to choose between "off" and
#: "detailed", and neither is the right default: off throws away the runs
#: worth having, detailed carries a per-thread row for every thread of every
#: child once a second.
LEVELS: Tuple[str, ...] = ("off", "summary", "detailed")

#: Cheap enough to leave on: one psutil read a second is far below the noise
#: floor of anything spaCR does.
DEFAULT_LEVEL = "summary"

#: Read by CLI and worker processes, which have no Preferences dialog.
ENV_VAR = "SPACR_PERFORMANCE_LOG"

#: What :func:`level_source` can answer. A support request that says
#: "summary" is ambiguous until it says whether a person chose it.
SOURCES: Tuple[str, ...] = ("argument", "environment", "preference", "default")

#: Memory definitions, most private first. USS is what would be freed if the
#: process died; PSS shares each page between its users; RSS charges every
#: shared page to every process that maps it.
MEASURES: Tuple[str, ...] = ("uss", "pss", "rss")

#: About 1 Hz. Tighter buys detail nobody reads and starts to perturb the
#: thing being measured, which is the failure that rules verbose logging out.
DEFAULT_INTERVAL_SECONDS = 1.0

#: An hour of samples at the default interval. The buffer is a ring, so a
#: week-long run keeps the last hour rather than growing without limit.
DEFAULT_CAPACITY = 3600

#: Below this the loop stops being a sampler and starts being a spin.
MIN_INTERVAL_SECONDS = 0.01

#: The sampler thread's name, so a thread census can name it.
THREAD_NAME = "spacr-resource-log"


def _normalise(value: Any) -> Optional[str]:
    """One of :data:`LEVELS`, or ``None`` when the value names no level.

    :param value: text from the environment, a preference or a caller.
    :returns: the level in lower case, or ``None``.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text if text in LEVELS else None


def _preference_level() -> Optional[str]:
    """The Qt preference, when this install has one and it can be read.

    IMPORTED INSIDE THE CALL, never at module scope. This module runs in a
    CLI process and in a worker child, neither of which has Qt, and importing
    a GUI package to read one string would be the most expensive part of the
    measurement. An install whose preferences do not carry the setting is not
    an error either -- it means the environment variable and the default
    decide instead.

    :returns: the level the preference holds, or ``None``.
    """
    try:
        from .qt.preferences import get_performance_logging
    except Exception:                                            # noqa: BLE001
        LOG.debug("no Qt performance-logging preference to read",
                  exc_info=True)
        return None
    try:
        return _normalise(get_performance_logging())
    except Exception:                                            # noqa: BLE001
        LOG.debug("could not read the performance-logging preference",
                  exc_info=True)
        return None


def _resolve(level: Optional[str] = None) -> Tuple[str, str]:
    """The level in force and what decided it.

    THE ORDER, AND WHY. An explicit argument wins because the caller is
    holding the setting in its hand. The environment variable comes next
    because it is set per process, by the person starting THIS run, and it is
    how a headless run and a spawned worker are told anything at all. The
    stored Qt preference comes after it: it is a persisted choice that a
    worker inherits by accident rather than by intent, so a variable set for
    one run must be able to override it. The default is last and is not
    "off".

    :param level: an explicit level, or ``None`` to resolve one.
    :returns: ``(level, source)``, where source is one of :data:`SOURCES`.
    :raises ValueError: if an explicit level is not one of :data:`LEVELS`.
    """
    if level is not None:
        named = _normalise(level)
        if named is None:
            raise ValueError(
                f"Unknown performance-logging level {level!r}. "
                f"Choose from {LEVELS}.")
        return named, "argument"
    named = _normalise(os.environ.get(ENV_VAR))
    if named is not None:
        return named, "environment"
    named = _preference_level()
    if named is not None:
        return named, "preference"
    return DEFAULT_LEVEL, "default"


def resolve_level(level: Optional[str] = None) -> str:
    """Which of :data:`LEVELS` is in force.

    :param level: an explicit level, or ``None`` to resolve one from the
        environment, then the stored preference, then :data:`DEFAULT_LEVEL`.
    :returns: one of :data:`LEVELS`.
    :raises ValueError: if an explicit level is not one of :data:`LEVELS`.
    """
    return _resolve(level)[0]


def level_source(level: Optional[str] = None) -> str:
    """What decided the level, for a test and for a support request.

    :param level: the same argument :func:`resolve_level` takes.
    :returns: one of :data:`SOURCES`.
    :raises ValueError: if an explicit level is not one of :data:`LEVELS`.
    """
    return _resolve(level)[1]


def _psutil():
    """The psutil module, or ``None`` when this install has none.

    :returns: the module, or ``None``.
    """
    try:
        import psutil
    except ImportError:
        LOG.debug("psutil is absent; the process tree cannot be read")
        return None
    return psutil


def _seconds(value: Any) -> Optional[float]:
    """A CPU time as seconds, or ``None`` when none was supplied.

    Never zero for a missing figure. A zero reads as "this thread was free",
    which is the opposite of "nobody could measure it".

    :param value: whatever the platform returned.
    :returns: the time in seconds, or ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _quiet(call, cast):
    """A best-effort attribute of a process, or ``None``.

    Losing a name must not lose the memory figure beside it, so the parts of
    a row that are labels rather than measurements fail on their own.

    :param call: the zero-argument psutil accessor to try.
    :param cast: what to coerce the result to.
    :returns: the coerced value, or ``None``.
    """
    try:
        return cast(call())
    except Exception:                                            # noqa: BLE001
        return None


def _memory(psutil_module, process) -> Tuple[Optional[int], Optional[str]]:
    """One process's memory, and which definition it is.

    USS first, then PSS, then RSS. A platform that will not give the private
    figures still gives the resident one, and a row that says ``rss`` is
    worth more than no row -- but only because it says so.

    :param psutil_module: the psutil module in use.
    :param process: the process to read.
    :returns: ``(bytes, measure)``; ``(None, None)`` when the platform
        supplied no figure at all.
    :raises psutil.NoSuchProcess: if the process exited while being read.
    :raises psutil.AccessDenied: if nothing about it may be read.
    """
    full: Any = None
    try:
        full = process.memory_full_info()
    except psutil_module.NoSuchProcess:
        raise
    except Exception:                                            # noqa: BLE001
        # The private figures need permissions the resident one does not.
        LOG.debug("no private memory figures for pid %s",
                  getattr(process, "pid", None), exc_info=True)
    if full is not None:
        for measure in MEASURES:
            value = getattr(full, measure, None)
            if isinstance(value, int):
                return int(value), measure
    info = process.memory_info()
    resident = getattr(info, "rss", None)
    if not isinstance(resident, int):
        return None, None
    return int(resident), "rss"


def _cpu(process) -> Tuple[Optional[float], Optional[float]]:
    """A process's cumulative user and system time.

    :param process: the process to read.
    :returns: ``(user_seconds, system_seconds)``, either of which is ``None``
        when the platform did not supply it.
    """
    try:
        times = process.cpu_times()
    except Exception:                                            # noqa: BLE001
        LOG.debug("no CPU times for pid %s", getattr(process, "pid", None),
                  exc_info=True)
        return None, None
    return (_seconds(getattr(times, "user", None)),
            _seconds(getattr(times, "system", None)))


def _thread_rows(process) -> Optional[List[Dict[str, Any]]]:
    """Per-thread CPU times, or ``None`` where the platform has none.

    ``None`` rather than an empty list, which would say the process ran no
    threads, and rather than zeros, which would say its threads were free.

    :param process: the process to read.
    :returns: one row per thread, or ``None`` when unavailable.
    """
    try:
        threads = list(process.threads())
    except Exception:                                            # noqa: BLE001
        LOG.debug("no per-thread times for pid %s",
                  getattr(process, "pid", None), exc_info=True)
        return None
    rows: List[Dict[str, Any]] = []
    for thread in threads:
        ident = getattr(thread, "id", None)
        rows.append({
            "thread_id": (int(ident) if isinstance(ident, int)
                          and not isinstance(ident, bool) else None),
            "cpu_user": _seconds(getattr(thread, "user_time", None)),
            "cpu_system": _seconds(getattr(thread, "system_time", None)),
        })
    return rows


def _process_row(psutil_module, process,
                 detailed: bool) -> Dict[str, Any]:
    """One process's line in a sample.

    :param psutil_module: the psutil module in use.
    :param process: the process to read.
    :param detailed: whether to include per-thread CPU times.
    :returns: the row, keyed ``pid``, ``ppid``, ``name``, ``memory``,
        ``measure``, ``cpu_user``, ``cpu_system``, and under ``detailed``
        also ``threads``.
    :raises psutil.NoSuchProcess: if the process exited while being read.
    :raises psutil.AccessDenied: if nothing about it may be read.
    """
    memory, measure = _memory(psutil_module, process)
    user, system = _cpu(process)
    row: Dict[str, Any] = {
        "pid": _quiet(lambda: process.pid, int),
        "ppid": _quiet(process.ppid, int),
        "name": _quiet(process.name, str),
        "memory": memory,
        "measure": measure,
        "cpu_user": user,
        "cpu_system": system,
    }
    if detailed:
        row["threads"] = _thread_rows(process)
    return row


def _coarsest(measures: Iterable[Optional[str]]) -> Optional[str]:
    """The weakest definition among several.

    A total is only as comparable as its worst member, so a tree summed
    mostly in USS with one RSS row is reported as RSS.

    :param measures: the measures to reconcile.
    :returns: one of :data:`MEASURES`, or ``None`` when none was named.
    """
    seen = [m for m in measures if m in MEASURES]
    if not seen:
        return None
    return max(seen, key=MEASURES.index)


def preferred_measure(process: Any = None) -> Optional[str]:
    """Which memory definition this platform can supply.

    :param process: the process to probe, or ``None`` for this one.
    :returns: one of :data:`MEASURES`, or ``None`` when nothing can be read.
    """
    psutil_module = _psutil()
    if psutil_module is None:
        return None
    try:
        target = psutil_module.Process() if process is None else process
        return _memory(psutil_module, target)[1]
    except Exception:                                            # noqa: BLE001
        LOG.debug("could not decide a memory measure", exc_info=True)
        return None


def _unreadable_sample(level: str, when: float) -> Dict[str, Any]:
    """A sample from a machine that could not be read at all.

    Every key a readable sample has, so a reader never has to ask which shape
    it got, and ``None`` rather than ``0`` in the figures.

    :param level: the level in force.
    :param when: the timestamp to stamp.
    :returns: the sample.
    """
    return {"record": "sample", "time": when, "level": level,
            "measure": None, "unit": "bytes", "total": None,
            "processes": [], "missed": 0}


def tree_sample(level: Optional[str] = None, process: Any = None,
                now: Optional[float] = None) -> Dict[str, Any]:
    """One reading of this process and every descendant.

    :param level: ``"summary"`` or ``"detailed"``, resolved from the
        environment and the preference when ``None``. ``"detailed"`` adds
        per-thread CPU times. ``"off"`` governs the background sampler rather
        than a reading a caller asks for outright, and reads as ``"summary"``
        here.
    :param process: the root of the tree, or ``None`` for this process.
    :param now: the timestamp to stamp, or ``None`` for the wall clock.
    :returns: a record keyed ``record``, ``time``, ``level``, ``measure``,
        ``unit``, ``total``, ``processes`` and ``missed``. ``total`` and
        ``measure`` are ``None`` when nothing could be read, which is not the
        same as zero. ``missed`` counts processes that vanished or refused to
        be read while the tree was walked.
    """
    named = resolve_level(level)
    when = time.time() if now is None else float(now)
    psutil_module = _psutil()
    if psutil_module is None:
        return _unreadable_sample(named, when)
    try:
        root = psutil_module.Process() if process is None else process
        members = [root] + list(root.children(recursive=True))
    except Exception:                                            # noqa: BLE001
        LOG.debug("could not enumerate the process tree", exc_info=True)
        return _unreadable_sample(named, when)

    detailed = named == "detailed"
    rows: List[Dict[str, Any]] = []
    missed = 0
    for member in members:
        try:
            rows.append(_process_row(psutil_module, member, detailed))
        except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
            # A child exiting between enumeration and reading is what a
            # short-lived worker DOES. Skip it, keep the rest, and say so.
            missed += 1
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not read a process in the tree", exc_info=True)
            missed += 1
    figures = [row["memory"] for row in rows
               if isinstance(row["memory"], int)]
    total = sum(figures) if figures else None
    return {"record": "sample", "time": when, "level": named,
            "measure": _coarsest(row["measure"] for row in rows),
            "unit": "bytes", "total": total, "processes": rows,
            "missed": missed}


def summarise(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Totals and peaks over recorded samples, and which pid held the peak.

    Empty when nothing was recorded -- NOT zero, for the reason
    `spacr.fit_resources.peak` gives: "nothing was using memory" and "nobody
    measured" are opposite findings, and a summary that spells the second as
    the first invites a reader to conclude the run was cheap.

    :param samples: records from :func:`tree_sample`.
    :returns: ``samples``, ``measure``, ``missed`` and ``pids`` always;
        ``peak_total`` and ``peak_total_time`` when any tree total was read;
        ``peak_process`` naming the pid that held the largest single share;
        ``cpu_seconds``, the largest CPU total seen in one sample, when any
        CPU time was read.
    """
    rows = [s for s in samples if isinstance(s, Mapping)]
    if not rows:
        return {}
    out: Dict[str, Any] = {
        "samples": len(rows),
        "measure": _coarsest(row.get("measure") for row in rows),
        "missed": sum(int(row["missed"]) for row in rows
                      if isinstance(row.get("missed"), int)),
    }
    processes = [(row, member) for row in rows
                 for member in row.get("processes") or []]
    out["pids"] = sorted({member["pid"] for _row, member in processes
                          if isinstance(member.get("pid"), int)})

    totals = [(row["total"], row.get("time")) for row in rows
              if isinstance(row.get("total"), int)]
    if totals:
        out["peak_total"], out["peak_total_time"] = max(totals)

    largest = None
    for row, member in processes:
        figure = member.get("memory")
        if not isinstance(figure, int):
            continue
        if largest is None or figure > largest["memory"]:
            largest = {"pid": member.get("pid"), "name": member.get("name"),
                       "memory": figure, "measure": member.get("measure"),
                       "time": row.get("time")}
    if largest is not None:
        out["peak_process"] = largest

    # Cumulative counters, so the largest sample is the run's total rather
    # than a sum over samples, which would count the same seconds again once
    # a second.
    burned = []
    for row in rows:
        seconds = [value for member in row.get("processes") or []
                   for value in (member.get("cpu_user"),
                                 member.get("cpu_system"))
                   if isinstance(value, float)]
        if seconds:
            burned.append(sum(seconds))
    if burned:
        out["cpu_seconds"] = max(burned)
    return out


def _count(number: int, noun: str, plural: Optional[str] = None) -> str:
    """A number and its noun, singular when there is one of it.

    :param number: how many.
    :param noun: the singular form.
    :param plural: the plural form, when adding an "s" would not make it.
    :returns: the phrase.
    """
    if number == 1:
        return f"{number} {noun}"
    return f"{number} {plural or noun + 's'}"


def describe(samples: Sequence[Mapping[str, Any]]) -> str:
    """The peaks as a person reads them, for a log line or a support request.

    :param samples: records from :func:`tree_sample`.
    :returns: the lines, or ``""`` when nothing was recorded.
    """
    high = summarise(samples)
    if not high:
        return ""
    lines = [f"  performance log: {_count(high['samples'], 'sample')} over "
             f"{_count(len(high['pids']), 'process', 'processes')}, measured as "
             f"{high.get('measure') or 'not measured'}"]
    if "peak_total" in high:
        lines.append(f"  PEAK tree     {readable(high['peak_total'])}")
    if "peak_process" in high:
        worst = high["peak_process"]
        lines.append(f"  PEAK process  {readable(worst['memory'])} in pid "
                     f"{worst['pid']} ({worst['name'] or 'unnamed'})")
    if "cpu_seconds" in high:
        lines.append(f"  CPU           {high['cpu_seconds']:.1f} s")
    if high["missed"]:
        lines.append(f"  {_count(high['missed'], 'reading')} missed, which "
                     f"is what a child exiting mid-sample leaves")
    return "\n".join(lines)


def read_log(path: Any) -> Dict[str, Any]:
    """Read a written log back, tolerating a run that was killed mid-line.

    One JSON object per line is the format that survives a kill: everything
    written before the kill parses, and the partial last line is dropped
    rather than making the file unreadable.

    :param path: the file a :class:`ResourceSampler` wrote.
    :returns: ``header`` (empty when the file has none), ``samples``, and
        ``unreadable``, the number of lines that could not be parsed.
    """
    header: Dict[str, Any] = {}
    samples: List[Dict[str, Any]] = []
    unreadable = 0
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        LOG.debug("no resource log at %s", path, exc_info=True)
        return {"header": header, "samples": samples, "unreadable": 0}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except ValueError:
            unreadable += 1
            continue
        if not isinstance(record, dict):
            unreadable += 1
        elif record.get("record") == "header":
            header = record
        else:
            samples.append(record)
    return {"header": header, "samples": samples, "unreadable": unreadable}


class ResourceSampler:
    """A bounded background record of what the process tree costs.

    A daemon thread takes one reading every ``interval`` seconds into a ring
    buffer of ``capacity`` samples, so a run that lasts a week cannot grow
    the log without limit and still leaves the most recent hour when it dies.

    The thread is a daemon and is never the GUI thread: it cannot hold the
    process open at exit and it cannot delay a repaint.

    When a path is given, each sample is written as one JSON line and
    flushed, after a header line naming the level, the measure, the interval
    and the start. Registering that file against a run is the caller's job --
    this class is imported by worker processes that have no artifacts
    database and no GUI.

    At level ``"off"`` no thread is started and no file is opened, which is
    what a thread census before and after a run is entitled to see.
    """

    def __init__(self, path: Any = None, level: Optional[str] = None,
                 interval: float = DEFAULT_INTERVAL_SECONDS,
                 capacity: int = DEFAULT_CAPACITY,
                 label: Optional[str] = None, clock=time.time) -> None:
        """Prepare a sampler without starting it.

        :param path: where to write the series, or ``None`` to keep it only
            in memory.
        :param level: one of :data:`LEVELS`, or ``None`` to resolve one.
        :param interval: seconds between readings, floored at
            :data:`MIN_INTERVAL_SECONDS`.
        :param capacity: how many samples the ring buffer holds.
        :param label: what this record is OF -- a run id, a sweep trial -- so
            a file found later can be matched to the work that made it.
        :param clock: the time source, passed in so a test can drive it.
        :raises ValueError: if an explicit level is not one of
            :data:`LEVELS`.
        """
        self.level, self.level_source = _resolve(level)
        self.path = None if path is None else Path(path)
        self.interval = max(float(interval), MIN_INTERVAL_SECONDS)
        self.capacity = max(int(capacity), 1)
        self.label = label
        self.measure: Optional[str] = None
        self._clock = clock
        self._samples: deque = deque(maxlen=self.capacity)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._opened = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Begin sampling on a daemon thread.

        :returns: whether a sampler thread is now running, which is ``False``
            at level ``"off"``.
        """
        if self.level == "off":
            LOG.debug("performance logging is off; no sampler started")
            return False
        if self.is_running():
            return True
        self._probe_measure()
        self._stop.clear()
        self._ensure_log()
        thread = threading.Thread(target=self._loop, name=THREAD_NAME,
                                  daemon=True)
        self._thread = thread
        thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop sampling, join the thread and close the file.

        :param timeout: seconds to wait for the thread to end.
        :returns: whether no sampler thread remains.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)
        self._close()
        return thread is None or not thread.is_alive()

    def is_running(self) -> bool:
        """Whether a sampler thread is alive.

        :returns: ``True`` while the thread is running.
        """
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "ResourceSampler":
        """Start sampling for the duration of a block.

        :returns: this sampler.
        """
        self.start()
        return self

    def __exit__(self, *exc_info) -> bool:
        """Stop sampling, whatever ended the block.

        :param exc_info: the exception the block raised, if any.
        :returns: ``False``, so an exception in the block still propagates.
        """
        self.stop()
        return False

    def sample_once(self) -> Optional[Dict[str, Any]]:
        """Take one reading now, keep it and write it.

        Public and separate from the loop so a caller -- a stage boundary, a
        test -- can take a reading at a moment it chooses rather than waiting
        for the interval to come round.

        :returns: the sample, or ``None`` at level ``"off"``.
        """
        if self.level == "off":
            return None
        with self._lock:
            # The header is written before the first reading is taken, so a
            # file always names its measure before it carries a figure.
            self._ensure_log()
            sample = tree_sample(self.level, now=self._clock())
            self._samples.append(sample)
            self._write(sample)
        return sample

    def samples(self) -> List[Dict[str, Any]]:
        """Every sample still in the ring buffer, oldest first.

        :returns: a copy, so the caller can read it while sampling continues.
        """
        with self._lock:
            return list(self._samples)

    def summary(self) -> Dict[str, Any]:
        """Totals and peaks over what has been recorded.

        :returns: what :func:`summarise` returns, empty when nothing was
            recorded.
        """
        return summarise(self.samples())

    def describe(self) -> str:
        """The peaks as a person reads them.

        :returns: what :func:`describe` returns, ``""`` when nothing was
            recorded.
        """
        return describe(self.samples())

    def _loop(self) -> None:
        """Sample until asked to stop, surviving anything one sample does."""
        while True:
            try:
                self.sample_once()
            except Exception:                                    # noqa: BLE001
                # A sampler that dies on a bad reading stops recording the
                # run at exactly the point the run started going wrong.
                LOG.debug("a reading failed; sampling continues",
                          exc_info=True)
            if self._stop.wait(self.interval):
                return

    def _probe_measure(self) -> None:
        """Decide once which memory definition this platform can supply."""
        if self.measure is None:
            self.measure = preferred_measure()

    def _ensure_log(self) -> None:
        """Open the file on first use and write the header that names the run.

        ONCE, and not again after :meth:`stop` has closed it: reopening would
        truncate the record of the run that has just ended. Opening on first
        use rather than in the constructor means a sampler that is built and
        never used leaves no file, and a caller that takes readings at stage
        boundaries without starting the thread still gets one.
        """
        if self.path is None or self._opened:
            return
        self._opened = True
        self._probe_measure()
        try:
            self._handle = open(self.path, "w", encoding="utf-8")
        except OSError:
            LOG.debug("could not open the resource log at %s", self.path,
                      exc_info=True)
            self._handle = None
            return
        self._write({
            "record": "header",
            "level": self.level,
            "level_source": self.level_source,
            "measure": self.measure,
            "unit": "bytes",
            "interval": self.interval,
            "capacity": self.capacity,
            "started": self._clock(),
            "pid": os.getpid(),
            "platform": sys.platform,
            "label": self.label,
        })

    def _write(self, record: Mapping[str, Any]) -> None:
        """Append one JSON line and flush it, so a kill loses at most one.

        :param record: the header or sample to write.
        """
        handle = self._handle
        if handle is None:
            return
        try:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not write to the resource log", exc_info=True)

    def _close(self) -> None:
        """Close the file, if one was opened."""
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.close()
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not close the resource log", exc_info=True)
