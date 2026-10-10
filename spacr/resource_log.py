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

WHY IT IS NOT VERBOSE LOGGING. Verbose logging only decides which log records
are kept, so it has no account of memory to give. The function tracer that it
once installed fired on every call and every return, and it cost twenty times
the startup. An account taken through a tracer would describe the traced
program rather than the real one, which is exactly the program nobody wants
measured. So this samples
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
from multiprocessing.context import BaseContext
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

    A daemon thread takes one reading every effective ``interval`` seconds
    into a ring buffer of ``capacity`` samples. The default settings retain
    the most recent hour in memory; custom settings retain approximately
    ``capacity * interval`` seconds. A run that lasts a week therefore cannot
    grow the in-memory series without limit.

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
            :data:`MIN_INTERVAL_SECONDS`; together with ``capacity``, this
            determines the retained time span.
        :param capacity: maximum number of in-memory samples to retain,
            clamped to at least one; older samples are discarded.
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


_PROTECTED_PROCESS_NAMES = frozenset(name.lower() for name in (
    'systemd', 'init', 'login', 'sshd', 'ssh-agent', 'gpg-agent', 'dbus-daemon',
    'dbus-broker', 'Xorg', 'Xwayland', 'gnome-shell', 'gnome-session-binary',
    'gnome-keyring-daemon', 'gdm', 'gdm-wayland-session', 'gdm-x-session',
    'plasmashell', 'kwin_x11', 'kwin_wayland', 'ksmserver', 'kded5', 'kded6',
    'xfce4-session', 'xfwm4', 'xfce4-panel', 'cinnamon', 'mutter',
    'lightdm', 'sddm', 'pulseaudio', 'pipewire', 'pipewire-pulse',
    'wireplumber', 'at-spi-bus-launcher', 'at-spi2-registryd',
    'xdg-desktop-portal', 'xdg-desktop-portal-gnome', 'xdg-document-portal',
    'xdg-permission-store', 'ibus-daemon', 'nautilus-desktop',
    'bash', 'zsh', 'sh', 'fish', 'tmux', 'tmux: server', 'screen',
    'explorer.exe', 'dwm.exe', 'winlogon.exe', 'csrss.exe', 'smss.exe',
    'wininit.exe', 'services.exe', 'lsass.exe', 'svchost.exe', 'sihost.exe',
    'taskhostw.exe', 'fontdrvhost.exe', 'ctfmon.exe', 'conhost.exe',
    'runtimebroker.exe', 'shellexperiencehost.exe', 'searchhost.exe',
    'startmenuexperiencehost.exe', 'textinputhost.exe', 'dllhost.exe',
    'system', 'registry', 'memory compression', 'system idle process',
    'loginwindow', 'WindowServer', 'Dock', 'Finder', 'SystemUIServer',
    'launchd', 'ControlCenter', 'NotificationCenter', 'coreaudiod',
))

_PROTECTED_USERS = frozenset(('root', 'system', 'nt authority\\system',
                              'local service', 'network service',
                              'nt authority\\local service',
                              'nt authority\\network service'))


def _spacr_process_ids(psutil_module) -> set:
    """Process ids of this spaCR process, its ancestors and its children."""
    ids = set()
    try:
        me = psutil_module.Process()
        ids.add(me.pid)
        for relative in list(me.parents()) + list(me.children(recursive=True)):
            ids.add(relative.pid)
    except Exception:
        ids.add(os.getpid())
    return ids


def _current_username(psutil_module) -> Optional[str]:
    """The user name that owns this process, or ``None`` if unreadable."""
    try:
        return psutil_module.Process().username()
    except Exception:
        return None


def _closable_processes(psutil_module=None, limit: int = 15) -> List[Dict[str, Any]]:
    """The user's own processes using the most RAM, which spaCR may offer to close.

    spaCR itself, its parents and children, processes of other users or of
    root and the system, and desktop, session and shell processes are left
    out, so closing a listed row cannot end the session or the run.

    :param psutil_module: the psutil module to read; the installed one when
        omitted.
    :param limit: at most this many rows, largest first.
    :returns: dicts with ``pid``, ``name`` and ``rss`` (bytes); empty when
        psutil is missing.
    """
    psutil_module = psutil_module or _psutil()
    if psutil_module is None:
        return []
    owner = _current_username(psutil_module)
    skip = _spacr_process_ids(psutil_module)
    rows = []
    for proc in psutil_module.process_iter(['pid', 'name', 'username',
                                            'memory_info']):
        try:
            info = proc.info
            pid = int(info.get('pid') or 0)
            name = str(info.get('name') or '')
            user = str(info.get('username') or '')
            memory = info.get('memory_info')
        except Exception:
            continue
        if pid <= 4 or pid in skip or not name or memory is None:
            continue
        if not user or user.lower() in _PROTECTED_USERS:
            continue
        if owner is not None and user != owner:
            continue
        if name.lower() in _PROTECTED_PROCESS_NAMES:
            continue
        if 'spacr' in name.lower():
            continue
        rows.append({'pid': pid, 'name': name, 'rss': int(memory.rss)})
    rows.sort(key=lambda row: row['rss'], reverse=True)
    return rows[:max(0, int(limit))]


def _close_processes(pids: Iterable[int], psutil_module=None) -> Dict[int, str]:
    """Ask each process to close: SIGTERM on Linux and macOS, terminate on Windows.

    Nothing is killed outright; a program that wants to save its work gets
    the chance. Only processes :func:`_closable_processes` would list are
    touched, so a stale or edited pid list cannot reach spaCR or the session.

    :param pids: the process ids the user confirmed.
    :returns: ``{pid: outcome}`` with outcome ``'closed'``, ``'gone'``,
        ``'denied'`` or ``'refused'``.
    """
    psutil_module = psutil_module or _psutil()
    outcomes: Dict[int, str] = {}
    if psutil_module is None:
        return outcomes
    allowed = {row['pid'] for row in _closable_processes(psutil_module,
                                                          limit=10 ** 6)}
    for pid in pids:
        pid = int(pid)
        if pid not in allowed:
            outcomes[pid] = 'refused'
            continue
        try:
            psutil_module.Process(pid).terminate()
            outcomes[pid] = 'closed'
        except psutil_module.NoSuchProcess:
            outcomes[pid] = 'gone'
        except Exception:
            outcomes[pid] = 'denied'
    return outcomes


_RAM_RESERVE_FRACTION = 0.125
_RAM_DEFAULT_MULTIPLIER = 8.0
_RAM_MIN_UNIT_BYTES = 64 * 1024 ** 2

_RAM_WORKER_MULTIPLIERS: Dict[str, float] = {
    'measure': 8.0,
    'mask': 10.0,
    'classical_masks': 10.0,
    'adjust_masks': 14.0,
    'merge_split': 14.0,
    'motility': 6.0,
    'classify': 4.0,
    'dataset': 3.0,
    'augment': 10.0,
    'cellpose_dataset': 10.0,
    'map_barcodes': 4.0,
    'simulation': 1.0,
    'regression': 2.0,
    'sweep': 1.0,
    'umap': 1.0,
    'ml_analyze': 1.0,
    'ops_decode': 6.0,
}

_APP_RAM_UNITS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    'measure': ('measure', ('.npy',)),
    'mask': ('mask', ('.npy', '.npz', '.tif', '.tiff', '.png')),
    'timelapse': ('mask', ('.npy', '.npz', '.tif', '.tiff', '.png')),
    'motility': ('motility', ('.npy',)),
    'classify': ('classify', ('.png', '.tif', '.tiff')),
    'activation': ('classify', ('.png', '.tif', '.tiff', '.tar')),
    'train_cellpose': ('cellpose_dataset', ('.tif', '.tiff', '.png', '.npy')),
    'map_barcodes': ('map_barcodes', ()),
    'umap': ('umap', ('.db', '.csv', '.parquet')),
    'ml_analyze': ('ml_analyze', ('.db', '.csv', '.parquet')),
    'regression': ('regression', ('.db', '.csv', '.parquet')),
    'ops': ('ops_decode', ('.tif', '.tiff', '.nd2', '.npy')),
}

_SAMPLE_WALK_LIMIT = 2000

_RAM_GUARD_STATE = threading.local()


class _ram_guard_scope:
    """Carry a run's ``ram_guard`` setting to pool sites that see no settings.

    Entered around a pipeline call on the thread that runs it, so helpers
    deep in the pipeline honour the run's choice. Outside any scope the
    guard is on.
    """

    def __init__(self, settings: Optional[Mapping[str, Any]]):
        """Read ``ram_guard`` from ``settings``; on unless it is ``False``."""
        value = True
        if isinstance(settings, Mapping):
            value = settings.get('ram_guard', True) is not False
        self._value = value
        self._previous = None

    def __enter__(self):
        """Apply this run's choice for the duration of the block."""
        self._previous = getattr(_RAM_GUARD_STATE, 'enabled', None)
        _RAM_GUARD_STATE.enabled = self._value
        return self

    def __exit__(self, *exc):
        """Restore the previous choice; exceptions propagate."""
        _RAM_GUARD_STATE.enabled = self._previous
        return False


def _ram_guard_enabled(settings: Optional[Mapping[str, Any]] = None) -> bool:
    """Whether the RAM guard is on for this run.

    :param settings: the run settings when the caller has them; otherwise
        the enclosing :class:`_ram_guard_scope` decides, defaulting to on.
    """
    if isinstance(settings, Mapping) and 'ram_guard' in settings:
        return settings.get('ram_guard') is not False
    enabled = getattr(_RAM_GUARD_STATE, 'enabled', None)
    return True if enabled is None else bool(enabled)


def _ram_snapshot(psutil_module=None) -> Optional[Tuple[int, int]]:
    """Return ``(available_bytes, total_bytes)``, or ``None`` when unreadable.

    :param psutil_module: the psutil module to read; the installed one when
        omitted.
    """
    psutil_module = psutil_module or _psutil()
    if psutil_module is None:
        return None
    try:
        memory = psutil_module.virtual_memory()
        return int(memory.available), int(memory.total)
    except Exception:
        return None


def _ram_reserve_bytes(total_bytes: int) -> int:
    """Bytes of RAM that spaCR leaves free for the desktop and the system."""
    return int(total_bytes * _RAM_RESERVE_FRACTION)


def _max_safe_workers(available_bytes: int, total_bytes: int,
                      per_worker_bytes: int) -> Optional[int]:
    """How many workers fit in available RAM while keeping the reserve free.

    :returns: at least 1, or ``None`` when ``per_worker_bytes`` is unknown;
        one worker always runs.
    """
    if per_worker_bytes <= 0:
        return None
    spare = available_bytes - _ram_reserve_bytes(total_bytes)
    return max(1, int(spare // per_worker_bytes))


def _requested_workers(n_jobs: Any) -> int:
    """The worker count ``n_jobs`` asks for; ``None``, 0 and negatives mean every core."""
    try:
        value = int(n_jobs)
    except (TypeError, ValueError):
        value = 0
    if value >= 1:
        return value
    cores = os.cpu_count() or 1
    return max(1, cores + 1 + value) if value < 0 else cores


def _array_file_nbytes(path: Any) -> int:
    """In-memory size of one input file, read from its header where possible.

    ``.npy`` gives the array size, ``.npz`` the sum of its arrays, TIFF and
    PNG the decoded pixel size; anything else falls back to its size on disk.

    :returns: ``0`` when the file cannot be read.
    """
    try:
        path = os.fspath(path)
    except TypeError:
        return 0
    lower = path.lower()
    try:
        if lower.endswith('.npy'):
            import numpy as np
            return int(np.load(path, mmap_mode='r').nbytes)
        if lower.endswith('.npz'):
            import numpy as np
            with np.load(path) as archive:
                return int(sum(archive[key].nbytes for key in archive.files))
        if lower.endswith(('.tif', '.tiff')):
            import tifffile
            with tifffile.TiffFile(path) as tif:
                series = tif.series[0]
                size = 1
                for extent in series.shape:
                    size *= int(extent)
                return int(size * series.dtype.itemsize)
        if lower.endswith('.png'):
            from PIL import Image
            with Image.open(path) as image:
                width, height = image.size
                bands = len(image.getbands())
                depth = 2 if image.mode.startswith('I;16') or image.mode == 'I' else 1
                return int(width * height * bands * depth)
        return int(os.path.getsize(path))
    except Exception:
        try:
            return int(os.path.getsize(path))
        except OSError:
            return 0


def _sample_input_file(src: Any, suffixes: Sequence[str]) -> Optional[str]:
    """Find one input file under ``src`` with one of ``suffixes``.

    ``src`` may be a file, a folder or a list of either. Folders are walked
    in sorted order, stopping after a bounded number of entries so a huge
    tree cannot stall the check.

    :returns: the path, or ``None`` when none is found.
    """
    if isinstance(src, (list, tuple)):
        for item in src:
            found = _sample_input_file(item, suffixes)
            if found:
                return found
        return None
    if not src or not suffixes:
        return None
    src = os.fspath(src)
    wanted = tuple(s.lower() for s in suffixes)
    if os.path.isfile(src):
        return src if src.lower().endswith(wanted) else None
    if not os.path.isdir(src):
        return None
    seen = 0
    for root, dirs, files in os.walk(src):
        dirs.sort()
        for suffix in wanted:
            for name in sorted(files):
                if name.lower().endswith(suffix) and not name.startswith('.'):
                    return os.path.join(root, name)
        seen += len(files) + len(dirs)
        if seen > _SAMPLE_WALK_LIMIT:
            return None
    return None


def _ram_plan(unit_bytes: int, n_jobs: Any, *, module: str = 'measure',
              multiplier: Optional[float] = None,
              psutil_module=None) -> Optional[Dict[str, Any]]:
    """Estimate whether ``n_jobs`` workers of ``module`` fit in free RAM.

    Each worker is estimated as one input unit's in-memory size times the
    module's multiplier, the ratio of a worker's peak resident memory to its
    input measured on one unit.

    :param unit_bytes: in-memory size of one worker's input unit.
    :param n_jobs: the requested worker count.
    :param module: key into the per-module multipliers.
    :param multiplier: overrides the module's multiplier.
    :returns: a dict with ``module``, ``per_worker``, ``nbytes``,
        ``available``, ``total``, ``reserve``, ``max_safe``, ``requested``
        and ``exceeds``, or ``None`` when RAM or the unit size is unknown.
    """
    if not unit_bytes or unit_bytes <= 0:
        return None
    snapshot = _ram_snapshot(psutil_module)
    if snapshot is None:
        return None
    available, total = snapshot
    factor = float(multiplier or _RAM_WORKER_MULTIPLIERS.get(
        module, _RAM_DEFAULT_MULTIPLIER))
    per_worker = max(1, int(unit_bytes * factor))
    max_safe = _max_safe_workers(available, total, per_worker)
    requested = _requested_workers(n_jobs)
    return {'module': module, 'per_worker': per_worker,
            'nbytes': int(unit_bytes), 'available': available,
            'total': total, 'reserve': _ram_reserve_bytes(total),
            'max_safe': max_safe, 'requested': requested,
            'exceeds': requested > max_safe}


def _clamp_to_plan(n_jobs: Any, plan: Optional[Mapping[str, Any]],
                   ram_guard: Any = True) -> Any:
    """Lower ``n_jobs`` to the plan's RAM-safe count, printing a warning.

    :param ram_guard: ``False`` keeps ``n_jobs`` unchanged.
    :returns: ``n_jobs`` when it fits, the guard is off or there is no
        plan; otherwise the safe count.
    """
    if plan is None or ram_guard is False or not plan.get('exceeds'):
        return n_jobs
    gib = 1024 ** 3
    print(f"WARNING: {plan['module']}: {plan['requested']} workers would "
          f"need about {plan['requested'] * plan['per_worker'] / gib:.1f} GiB "
          f"of RAM but {plan['available'] / gib:.1f} GiB is available and "
          f"{plan['reserve'] / gib:.1f} GiB is kept free; using "
          f"{plan['max_safe']} workers. Set ram_guard to False to keep "
          f"n_jobs.")
    return plan['max_safe']


class _WorkerStartGate:
    """Space worker start attempts by ten seconds with cooperative stop.

    One gate belongs to one processing pool. Existing workers incur no task
    delay. Injecting clock and sleep allows deterministic scheduling checks.
    ``idle`` is an optional predicate set by the pool owner: once it holds,
    no queued work remains for a waiting start, so the wait ends at once and
    the late worker sees only its pool's exit signal. Without it a pool that
    finished its work early still joined every paced thread at shutdown,
    about ten seconds per unused worker.
    """

    idle = None

    def __init__(self, delay=10.0, *, clock=None, sleep=None):
        """Store a nonnegative interval and a parent/thread-owned start lock."""
        import math
        from .cancellation import current_token

        self.delay = float(delay)
        if not math.isfinite(self.delay) or self.delay < 0:
            raise ValueError('Worker start delay must be finite and nonnegative')
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._last = None
        self._lock = threading.Lock()
        self._token = current_token()

    def start(self, call):
        """Start immediately once, then wait between deployment attempts."""
        from .cancellation import checkpoint

        with self._lock:
            checkpoint()
            if self._token is not None:
                self._token.checkpoint()
            if self._last is not None:
                deadline = self._last + self.delay
                while True:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        break
                    if self.idle is not None and self.idle():
                        break
                    self._sleep(min(0.05, remaining))
                    checkpoint()
                    if self._token is not None:
                        self._token.checkpoint()
            self._last = self._clock()
            return call()


class _StaggeredProcess:
    """Parent-side process handle that paces start without changing its target.

    The real process, its spawn pickling, exit code and cleanup remain owned
    by multiprocessing. This wrapper is never sent to a child.
    """

    def __init__(self, process, gate):
        """Keep a real process and the shared parent-side pool gate."""
        object.__setattr__(self, '_process', process)
        object.__setattr__(self, '_gate', gate)

    def __getattr__(self, name):
        """Delegate process identity, liveness, joins and cleanup unchanged."""
        return getattr(self._process, name)

    def __setattr__(self, name, value):
        """Delegate writable attributes such as daemon to the real process."""
        setattr(self._process, name, value)

    def start(self):
        """Deploy this real process after the previous pool start attempt."""
        return self._gate.start(self._process.start)


class _StaggeredContext(BaseContext):
    """Preserve a multiprocessing context while pacing its worker starts."""

    def __init__(self, context, gate=None):
        """Store the caller's start method and one gate for this worker group."""
        self._context = context
        self._gate = gate or _WorkerStartGate()
        self._owned_processes = []

    def __getattr__(self, name):
        """Use the original context's queues, events, locks and start method."""
        return getattr(self._context, name)

    def get_context(self, method=None):
        """Use the original context for synchronization and serialization."""
        return (self._context.get_context(method) if method is not None else
                self._context)

    def get_start_method(self, allow_none=False):
        """Report the caller's actual multiprocessing start method."""
        return self._context.get_start_method(allow_none=allow_none)

    def Process(self, *args, **kwargs):
        """Wrap a real context process without altering its child arguments."""
        process = _StaggeredProcess(self._context.Process(*args, **kwargs), self._gate)
        self._owned_processes.append(process)
        return process

    def Pool(self, processes=None, initializer=None, initargs=(),
             maxtasksperchild=None):
        """Build a normal pool with this context's paced process factory."""
        from multiprocessing.pool import Pool

        return Pool(processes, initializer, initargs, maxtasksperchild,
                    context=self)


def _invoke_parallel_task(payload):
    """Return a task's outcome without aborting siblings on ordinary failure."""
    from .cancellation import PipelineCancelled

    index, function, arguments = payload
    try:
        return index, True, function(*arguments), None
    except PipelineCancelled:
        raise
    except Exception as error:
        import multiprocessing
        from multiprocessing.pool import ExceptionWithTraceback

        if (threading.current_thread() is threading.main_thread()
                and multiprocessing.current_process().name != 'MainProcess'):
            error = ExceptionWithTraceback(error, error.__traceback__)
        return index, False, error, arguments


def _iter_parallel_outcomes(outcomes, function, retry, *, ordered=True):
    """Stream primary successes, retaining input order when requested.

    An ordered consumer receives its successful prefix immediately. Only
    results behind an unresolved input are held until the separate final
    pass. An unordered consumer receives every primary success immediately.
    """
    from functools import partial
    from .runctx import _DeferredOverloadRetries

    pending, failures = {}, {}
    cursor = 0
    final = _DeferredOverloadRetries()
    for index, ok, value, arguments in outcomes:
        if ok:
            if ordered:
                pending[index] = value
            else:
                yield value
        else:
            failures[index] = value
            final.defer(index, value, partial(retry, function, arguments))
        if ordered:
            while cursor in pending:
                yield pending.pop(cursor)
                cursor += 1
    for index, result, error in final.drain():
        if error is None:
            failures.pop(index, None)
            if ordered:
                pending[index] = result
            else:
                yield result
        else:
            failures[index] = error
    if ordered:
        while cursor in pending:
            yield pending.pop(cursor)
            cursor += 1
        if cursor in failures:
            raise failures[cursor]
    elif failures:
        raise failures[min(failures)]


def _finish_parallel_outcomes(outcomes, function, retry):
    """Collect an ordered map without changing its list return contract."""
    return list(_iter_parallel_outcomes(outcomes, function, retry))


def _invoke_parallel_chunk(payloads):
    """Return every item outcome inside an original pool-sized chunk."""
    return [_invoke_parallel_task(payload) for payload in payloads]


def _parallel_chunks(payloads, chunksize):
    """Keep streaming submissions bounded to the requested chunk size."""
    from itertools import islice

    if chunksize < 1:
        raise ValueError('Chunksize must be 1+, not ' + str(chunksize))
    while True:
        chunk = tuple(islice(payloads, chunksize))
        if not chunk:
            return
        yield chunk


def _check_parallel_workers(pool, workers):
    """Refuse an unexplained worker loss instead of waiting for a lost result.

    Normal zero-exit worker recycling is permitted. Nonzero native exits do
    not establish a resource overload and never enter the deferred retry queue.
    """
    for worker in getattr(pool, '_pool', ()):
        if worker not in workers:
            workers.append(worker)
    for worker in workers:
        if worker.exitcode not in (None, 0):
            raise RuntimeError(
                f'Processing worker {worker.pid} exited with code {worker.exitcode}; '
                'no explicit overload result was received')


class _ParallelApplyResult:
    """Observe direct submissions without replacing their owner's retry policy."""

    def __init__(self, primary, pool, workers):
        """Keep the original asynchronous result and every owned worker handle."""
        self._primary = primary
        self._pool = pool
        self._workers = workers

    def __getattr__(self, name):
        """Delegate original readiness, callbacks and success reporting."""
        return getattr(self._primary, name)

    def ready(self):
        """Report readiness without concealing an unexplained worker loss."""
        _check_parallel_workers(self._pool, self._workers)
        return self._primary.ready()

    def get(self, timeout=None):
        """Return a direct result cooperatively or report its lost worker.

        :param timeout: original overall result timeout, or no time limit.
        """
        from multiprocessing import TimeoutError
        from .cancellation import checkpoint

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            checkpoint()
            _check_parallel_workers(self._pool, self._workers)
            remaining = None if deadline is None else deadline - time.monotonic()
            interval = 0.1 if remaining is None else max(0.0, min(0.1, remaining))
            try:
                return self._primary.get(timeout=interval)
            except TimeoutError:
                if remaining is not None and remaining <= 0:
                    raise

    def wait(self, timeout=None):
        """Wait for a direct result with cancellation and worker-loss checks.

        :param timeout: original overall wait timeout, or no time limit.
        """
        from .cancellation import checkpoint

        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.ready():
            checkpoint()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return
            self._primary.wait(0.1 if remaining is None else min(0.1, remaining))


class _ParallelAsyncResult:
    """Keep asynchronous readiness false until its distinct final pass ends."""

    def __init__(self, primary, pool, function, callback, error_callback, *, workers=None):
        """Collect primary results on an owned coordinator, then retry serially."""
        from .cancellation import current_token

        self._primary = primary
        self._pool = pool
        self._workers = list(getattr(pool, '_pool', ())) if workers is None else workers
        self._function = function
        self._callback = callback
        self._error_callback = error_callback
        self._token = current_token()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._value = None
        self._error = None
        self._thread = threading.Thread(target=self._run,
                                        name='spacr-final-retry', daemon=True)
        self._thread.start()

    def _wait_result(self, result):
        """Wait cooperatively without retaining a stopped pool coordinator."""
        from multiprocessing import TimeoutError
        from .cancellation import PipelineCancelled

        while True:
            self._checkpoint()
            _check_parallel_workers(self._pool, self._workers)
            try:
                return result.get(timeout=0.1)
            except TimeoutError:
                continue

    def _checkpoint(self):
        """Stop before submitting or consuming another processing task."""
        from .cancellation import PipelineCancelled

        if self._stop.is_set():
            raise PipelineCancelled('Processing pool stopped')
        if self._token is not None:
            self._token.checkpoint()

    def _retry(self, function, arguments):
        """Check Stop before submitting this final attempt to the real pool."""
        self._checkpoint()
        return self._wait_result(self._pool.apply_async(function, arguments))

    def _run(self):
        """Consume every primary result before starting one serial retry pass."""
        try:
            outcomes = self._wait_result(self._primary)
            self._value = _finish_parallel_outcomes(
                outcomes, self._function, self._retry)
        except BaseException as error:
            self._error = error
        try:
            if self._error is None and self._callback is not None:
                self._callback(self._value)
            elif self._error is not None and self._error_callback is not None:
                self._error_callback(self._error)
        except BaseException as error:
            self._error = error
        finally:
            self._done.set()

    def ready(self):
        """Whether primary processing and the final pass have both finished."""
        return self._done.is_set()

    def wait(self, timeout=None):
        """Wait for the complete asynchronous result, without raising it."""
        self._done.wait(timeout)

    def successful(self):
        """Report the final verdict only once the asynchronous job is ready."""
        if not self.ready():
            raise ValueError('Result is not ready')
        return self._error is None

    def get(self, timeout=None):
        """Return ordered results or raise the original final failure."""
        from multiprocessing import TimeoutError
        from .cancellation import checkpoint

        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.ready():
            checkpoint()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError
            self._done.wait(0.1 if remaining is None else min(0.1, remaining))
        if self._error is not None:
            raise self._error
        return self._value


class _ParallelPool:
    """A paced multiprocessing pool with a separate final overload queue.

    Map operations retain input order and original exceptions. Explicit
    apply_async owners retain their retry policy and use the common deferred
    queue at the end of their primary scheduler.
    """

    def __init__(self, pool, *, workers=None):
        """Wrap an already constructed pool without changing its lifecycle."""
        self._backend = pool
        self._workers = list(getattr(pool, '_pool', ())) if workers is None else workers
        self._async_results = []

    def __getattr__(self, name):
        """Preserve asynchronous submission, worker inspection and cleanup."""
        return getattr(self._backend, name)

    def __enter__(self):
        """Enter the underlying pool and expose the queued map interface."""
        self._backend.__enter__()
        return self

    def __exit__(self, *args):
        """Let the real pool stop and join its workers on context exit."""
        for result in self._async_results:
            result._stop.set()
        try:
            return self._backend.__exit__(*args)
        finally:
            for result in self._async_results:
                result._thread.join()

    def terminate(self):
        """Stop both real workers and every owned asynchronous coordinator."""
        for result in self._async_results:
            result._stop.set()
        try:
            self._backend.terminate()
        finally:
            for result in self._async_results:
                result._thread.join()

    def _map_async(self, function, payloads, chunksize, callback, error_callback):
        """Own one primary map and one final pass without closing the pool."""
        _check_parallel_workers(self._backend, self._workers)
        primary = self._backend.map_async(_invoke_parallel_task, payloads, chunksize)
        self._async_results[:] = [result for result in self._async_results
                                 if not result.ready()]
        result = _ParallelAsyncResult(primary, self._backend, function,
                                      callback, error_callback, workers=self._workers)
        self._async_results.append(result)
        return result

    def apply_async(self, function, args=(), kwds=None, callback=None,
                    error_callback=None):
        """Observe direct tasks while keeping the caller's final retry queue.

        :param function: original processing function.
        :param args: positional arguments for that function.
        :param kwds: keyword arguments for that function.
        :param callback: original success callback.
        :param error_callback: original processing-error callback.
        """
        _check_parallel_workers(self._backend, self._workers)
        primary = self._backend.apply_async(function, args, kwds or {},
                                             callback, error_callback)
        return _ParallelApplyResult(primary, self._backend, self._workers)

    def map_async(self, function, iterable, chunksize=None, callback=None,
                  error_callback=None):
        """Asynchronously map inputs, including their serial final overload pass."""
        payloads = ((index, function, (item,))
                    for index, item in enumerate(iterable))
        return self._map_async(function, payloads, chunksize, callback, error_callback)

    def starmap_async(self, function, iterable, chunksize=None, callback=None,
                      error_callback=None):
        """Asynchronously map argument tuples with one distinct final pass."""
        payloads = ((index, function, tuple(arguments))
                    for index, arguments in enumerate(iterable))
        return self._map_async(function, payloads, chunksize, callback, error_callback)

    def _retry(self, function, arguments):
        """Run one final task alone after every primary result was consumed."""
        from .cancellation import checkpoint
        from multiprocessing import TimeoutError

        checkpoint()
        result = self._backend.apply_async(function, arguments)
        while True:
            checkpoint()
            _check_parallel_workers(self._backend, self._workers)
            try:
                return result.get(timeout=0.1)
            except TimeoutError:
                continue

    def _stream(self, payloads, chunksize, *, unordered=False):
        """Poll chunked primary results without concealing a native worker loss."""
        from .cancellation import checkpoint
        from multiprocessing import TimeoutError

        chunks = _parallel_chunks(payloads, chunksize)
        method = self._backend.imap_unordered if unordered else self._backend.imap
        iterator = method(_invoke_parallel_chunk, chunks, 1)
        while True:
            checkpoint()
            _check_parallel_workers(self._backend, self._workers)
            try:
                chunk = iterator.next(timeout=0.1)
            except TimeoutError:
                continue
            except StopIteration:
                return
            yield from chunk

    def map(self, function, iterable, chunksize=None):
        """Map ordered inputs with paced workers and one final overload pass."""
        return self.map_async(function, iterable, chunksize).get()

    def starmap(self, function, iterable, chunksize=None):
        """Map argument tuples while keeping the normal pool chunk sizing."""
        return self.starmap_async(function, iterable, chunksize).get()

    def imap(self, function, iterable, chunksize=1):
        """Stream ordered results and finish overloads after the primary queue."""
        payloads = ((index, function, (item,))
                    for index, item in enumerate(iterable))
        outcomes = self._stream(payloads, chunksize)
        return _iter_parallel_outcomes(outcomes, function, self._retry)

    def imap_unordered(self, function, iterable, chunksize=1):
        """Stream primary completions, followed by the distinct final queue."""
        payloads = ((index, function, (item,))
                    for index, item in enumerate(iterable))
        outcomes = self._stream(payloads, chunksize, unordered=True)
        return _iter_parallel_outcomes(outcomes, function, self._retry, ordered=False)


def _parallel_pool(processes=None, initializer=None, initargs=(),
                   maxtasksperchild=None, *, context=None, gate=None):
    """Create a resource-guarded caller's pool with ten-second worker starts.

    Worker count and start method remain the caller's decision. A gate may
    be injected for deterministic tests; the production default is ten seconds.
    """
    import multiprocessing

    context = context or multiprocessing.get_context()
    paced = _StaggeredContext(context, gate)
    return _ParallelPool(paced.Pool(
        processes, initializer, initargs, maxtasksperchild), workers=paced._owned_processes)


def _call_parallel_task(task):
    """Call one original function with its positional and keyword arguments."""
    function, arguments, keywords = task
    return function(*arguments, **keywords)


def _cloudpickle_codec():
    """Use joblib's bundled codec or its separately installed successor."""
    try:
        from joblib.externals import cloudpickle
    except ImportError:
        import cloudpickle
    return cloudpickle


def _cloudpickled_parallel_task(payload):
    """Preserve closures and keyword arguments across a normal spawn pool."""
    cloudpickle = _cloudpickle_codec()

    return cloudpickle.dumps(_call_parallel_task(cloudpickle.loads(payload)))


def _parallel_cloudpickle_map(tasks, workers):
    """Process cloudpickle-compatible calls with the shared final retry pass.

    :param tasks: iterable of function, positional arguments and keyword dict.
    :param workers: already resource-guarded processing worker count.
    :returns: ordered results, decoded in the caller process.
    """
    import multiprocessing
    cloudpickle = _cloudpickle_codec()

    if workers == 1:
        outcomes = (_invoke_parallel_task((index, _call_parallel_task, (task,)))
                    for index, task in enumerate(tasks))
        return _finish_parallel_outcomes(
            outcomes, _call_parallel_task,
            lambda function, arguments: function(*arguments))
    payloads = (cloudpickle.dumps(task) for task in tasks)
    with _parallel_pool(workers, context=multiprocessing.get_context('spawn')) as pool:
        return [cloudpickle.loads(value) for value in
                pool.imap(_cloudpickled_parallel_task, payloads)]


def _data_loader_arguments(arguments, keywords):
    """Preserve DataLoader arguments while pacing real worker deployments."""
    import multiprocessing

    keywords = dict(keywords)
    workers = keywords.get('num_workers', arguments[5] if len(arguments) > 5 else 0)
    if workers:
        context = keywords.get('multiprocessing_context')
        if context is None or isinstance(context, str):
            context = multiprocessing.get_context(context)
        if not isinstance(context, _StaggeredContext):
            keywords['multiprocessing_context'] = _StaggeredContext(context)
    return keywords


def _parallel_data_loader(*arguments, **keywords):
    """Construct a standard Torch loader with paced processing workers."""
    from torch.utils.data import DataLoader

    return DataLoader(*arguments, **_data_loader_arguments(arguments, keywords))


def _initialize_staggered_thread(gate, initializer, arguments):
    """Pace a processing thread before it can initialize or consume tasks."""
    gate.start(lambda: initializer(*arguments) if initializer else None)


class _ParallelExecutor:
    """Preserve a normal executor and add a final overload pass to its map."""

    def __init__(self, executor):
        """Keep the executor's futures, shutdown and context ownership intact."""
        self._executor = executor

    def __getattr__(self, name):
        """Delegate explicit submit owners and executor lifecycle operations."""
        return getattr(self._executor, name)

    def __enter__(self):
        """Enter the real executor and return its queued map wrapper."""
        self._executor.__enter__()
        return self

    def __exit__(self, *args):
        """Join the real executor's workers at the original boundary."""
        return self._executor.__exit__(*args)

    def map(self, function, *iterables, timeout=None, chunksize=1):
        """Map each argument tuple, then replay only exhausted overloads."""
        from .cancellation import checkpoint

        deadline = None if timeout is None else time.monotonic() + timeout
        payloads = ((index, function, tuple(arguments))
                    for index, arguments in enumerate(zip(*iterables)))
        outcomes = self._executor.map(_invoke_parallel_task, payloads,
                                      timeout=timeout, chunksize=chunksize)

        def retry(call, arguments):
            """Honor cancellation and the original overall map timeout."""
            checkpoint()
            remaining = None if deadline is None else max(
                0.0, deadline - time.monotonic())
            return self._executor.submit(call, *arguments).result(remaining)

        return _iter_parallel_outcomes(outcomes, function, retry)


def _parallel_thread_executor(max_workers=None, thread_name_prefix='',
                              initializer=None, initargs=(), *, gate=None):
    """Start processing threads ten seconds apart, preserving executor limits."""
    from concurrent.futures import ThreadPoolExecutor

    gate = gate or _WorkerStartGate()
    executor = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix=thread_name_prefix,
        initializer=_initialize_staggered_thread,
        initargs=(gate, initializer, initargs))

    def idle():
        """Report a pool that is shutting down with no queued work left.

        Shutdown queues one exit sentinel after any pending work, and each
        exiting worker puts it back, so one remaining item is that sentinel.
        """
        queue = getattr(executor, '_work_queue', None)
        return bool(getattr(executor, '_shutdown', False)) and (
            queue is not None and queue.qsize() <= 1)

    gate.idle = idle
    return _ParallelExecutor(executor)


def _parallel_process_executor(max_workers=None, mp_context=None,
                               initializer=None, initargs=(),
                               max_tasks_per_child=None, *, gate=None):
    """Pace real process deployments without changing executor worker limits."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = mp_context or multiprocessing.get_context(
        'spawn' if max_tasks_per_child is not None else None)
    options = dict(max_workers=max_workers, mp_context=_StaggeredContext(context, gate),
                   initializer=initializer, initargs=initargs)
    if max_tasks_per_child is not None:
        options['max_tasks_per_child'] = max_tasks_per_child
    return _ParallelExecutor(ProcessPoolExecutor(
        **options))


def _guard_workers(module: str, n_jobs: Any, unit_bytes: int, *,
                   settings: Optional[Mapping[str, Any]] = None,
                   multiplier: Optional[float] = None,
                   psutil_module=None) -> Any:
    """Clamp a pool's worker count to what free RAM can hold.

    Every place spaCR starts worker processes or threads calls this with the
    size of one worker's input. Nothing happens when the workers fit, when
    the size or the RAM cannot be read, or when ``settings['ram_guard']`` is
    ``False``; otherwise a warning is printed and the safe count returned.

    :param module: which multiplier to use, e.g. ``'mask'``.
    :param n_jobs: the requested worker count.
    :param unit_bytes: in-memory size of one worker's input unit.
    :param settings: the run settings, read only for ``ram_guard``; the
        enclosing :class:`_ram_guard_scope` decides when omitted.
    :returns: the worker count to start.
    """
    if not _ram_guard_enabled(settings):
        return n_jobs
    try:
        plan = _ram_plan(unit_bytes, n_jobs, module=module,
                         multiplier=multiplier, psutil_module=psutil_module)
    except Exception:
        LOG.debug("could not estimate %s worker RAM", module, exc_info=True)
        return n_jobs
    return _clamp_to_plan(n_jobs, plan)


def _table_nbytes(table: Any) -> int:
    """In-memory size of an array or table a worker receives a copy of.

    :returns: ``0`` when the size cannot be read.
    """
    try:
        usage = getattr(table, 'memory_usage', None)
        if callable(usage) and hasattr(table, 'columns'):
            return int(usage(index=True, deep=False).sum())
        return int(getattr(table, 'nbytes', 0) or 0)
    except Exception:
        return 0


def _loader_unit_bytes(batch_size: Any, image_size: Any,
                       channels: Any = 3, prefetch: int = 2) -> int:
    """Bytes one data-loader worker holds: its prefetched batches of tensors.

    Each item is a float32 tensor of ``channels`` planes of
    ``image_size`` squared pixels; a worker keeps ``prefetch`` batches.

    :returns: ``0`` when a size cannot be read.
    """
    try:
        if isinstance(channels, (list, tuple)):
            channels = len(channels) or 3
        size = int(image_size or 224)
        return int(max(1, int(batch_size or 1)) * size * size
                   * max(1, int(channels or 3)) * 4 * max(1, prefetch))
    except (TypeError, ValueError):
        return 0


def _app_unit_bytes(app_key: str, settings: Mapping[str, Any]) -> int:
    """In-memory size of one worker's input for a module's run settings.

    :returns: ``0`` when the module starts no workers or no input is found.
    """
    profile = _APP_RAM_UNITS.get(app_key)
    if profile is None:
        return 0
    module, suffixes = profile
    if module == 'map_barcodes':
        try:
            chunk = int(settings.get('chunk_size') or 10000)
        except (TypeError, ValueError):
            chunk = 10000
        return chunk * 2 * 1024
    if module == 'measure':
        from .measure import _sample_field_path
        path = _sample_field_path(settings.get('src'))
    else:
        path = _sample_input_file(settings.get('src'), suffixes)
    if path is None:
        return 0
    size = _array_file_nbytes(path)
    if module in ('classify', 'cellpose_dataset'):
        try:
            batch = max(1, int(settings.get('batch_size') or 1))
        except (TypeError, ValueError):
            batch = 1
        if module == 'classify':
            size *= batch
    return size


def _app_ram_plan(app_key: str, settings: Mapping[str, Any], n_jobs: Any,
                  psutil_module=None) -> Optional[Dict[str, Any]]:
    """The RAM estimate for running ``app_key`` with ``n_jobs`` workers.

    :returns: the plan from :func:`_ram_plan`, or ``None`` when the module
        starts no workers or its input cannot be sized.
    """
    profile = _APP_RAM_UNITS.get(app_key)
    if profile is None:
        return None
    unit = _app_unit_bytes(app_key, settings)
    return _ram_plan(unit, n_jobs, module=profile[0],
                     psutil_module=psutil_module)
