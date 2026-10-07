"""Durable file-boundary RSS, assertion, and native-fault evidence for Qt.

Load explicitly with ``-p tools.pytest_plugins.qt_serial_rss_journal`` and set
``SPACR_QT_SERIAL_RSS_JOURNAL`` to a new JSONL path. Every record is written
and synced before pytest continues, so a later memory-guard ``os._exit`` does
not discard the boundaries or reported failures it already crossed. The
observer neither touches Qt objects nor changes test order, garbage collection,
or event processing. ``SPACR_QT_SERIAL_FAULT_LOG`` optionally names a separate
owned file descriptor for native stacks; it is rearmed after test teardown
because in-process application launches can redirect Python's fatal handler.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TextIO

import pytest

_journal: Path | None = None
_active_file: str | None = None
_completed_files = 0
_fault_file: TextIO | None = None
_prior_fault_enabled: bool | None = None
_real_enable = faulthandler.enable
_real_disable = faulthandler.disable


def _arm_fault_log() -> None:
    """Bind Python fatal signals to the descriptor this plugin keeps open."""
    if _fault_file is not None:
        _real_enable(file=_fault_file, all_threads=True)


def _rss_sample() -> dict[str, int | None]:
    """Read the same process RSS that the pytest memory guard watches."""
    values = {"rss_bytes": None, "hwm_bytes": None, "threads": None}
    fields = {"VmRSS:": "rss_bytes", "VmHWM:": "hwm_bytes", "Threads:": "threads"}
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            label, _, value = line.partition("\t")
            key = fields.get(label)
            if key:
                amount = int(value.split()[0])
                values[key] = amount * 1024 if key != "threads" else amount
    return values


def _cached_qt_counts(config: pytest.Config) -> dict[str, int | None]:
    """Read only the root test fixture's last safe widget-count snapshot."""
    counts = {"cached_qt_widgets": None, "cached_qt_top_levels": None}
    name = str((config.rootpath / "tests" / "conftest.py").resolve())
    plugin = config.pluginmanager.get_plugin(name)
    if plugin is None:
        return counts
    cached = vars(plugin).get("_LAST_LIVE_WIDGET_COUNT")
    if not isinstance(cached, tuple) or len(cached) != 2:
        return counts
    for key, value in zip(counts, cached):
        if type(value) is int and value >= 0:
            counts[key] = value
    return counts


def _write(event: str, **details: object) -> None:
    """Append and sync one complete record without retaining pytest objects."""
    assert _journal is not None
    record = {
        "event": event,
        "time_ns": time.time_ns(),
        "pid": os.getpid(),
        **_rss_sample(),
        **details,
    }
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    fd = os.open(_journal, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Create a fresh journal so separate acceptance attempts cannot mix."""
    global _journal, _active_file, _completed_files
    global _fault_file, _prior_fault_enabled
    name = os.environ.get("SPACR_QT_SERIAL_RSS_JOURNAL")
    if not name:
        raise pytest.UsageError("SPACR_QT_SERIAL_RSS_JOURNAL is required")
    _journal = Path(name).expanduser().absolute()
    _journal.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(_journal, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise pytest.UsageError(f"journal already exists: {_journal}") from error
    else:
        os.close(fd)
    _active_file = None
    _completed_files = 0
    _write(
        "session_start",
        source_sha=os.environ.get("GITHUB_SHA"),
        root=str(config.rootpath),
        guard_gb=os.environ.get("SPACR_TEST_MEMORY_GB"),
        fault_log=os.environ.get("SPACR_QT_SERIAL_FAULT_LOG"),
    )
    fault_name = os.environ.get("SPACR_QT_SERIAL_FAULT_LOG")
    if fault_name:
        path = Path(fault_name).expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except OSError as error:
            raise pytest.UsageError(f"cannot create serial fault log: {path}") from error
        _fault_file = os.fdopen(fd, "w", encoding="utf-8")
        _prior_fault_enabled = faulthandler.is_enabled()
        try:
            _arm_fault_log()
        except (OSError, ValueError) as error:
            _fault_file.close()
            _fault_file = None
            raise pytest.UsageError(f"cannot arm serial fault log: {path}") from error


def pytest_collection_finish(session: pytest.Session) -> None:
    """Bind the ordered test and file manifests without storing either."""
    nodes = hashlib.sha256()
    files = hashlib.sha256()
    last_file = None
    file_count = 0
    for item in session.items:
        name = os.path.relpath(str(item.path), session.config.rootpath)
        nodes.update(item.nodeid.encode("utf-8") + b"\n")
        if name != last_file:
            files.update(name.encode("utf-8") + b"\n")
            file_count += 1
            last_file = name
    _write(
        "collection",
        tests=len(session.items),
        files=file_count,
        ordered_nodeids_sha256=nodes.hexdigest(),
        ordered_files_sha256=files.hexdigest(),
    )
    _arm_fault_log()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Record the prior file's completed teardown before starting this one."""
    global _active_file, _completed_files
    name = os.path.relpath(str(item.path), item.config.rootpath)
    if name == _active_file:
        return
    if _active_file is not None:
        _write("file_end", file=_active_file, **_cached_qt_counts(item.config))
        _completed_files += 1
    _active_file = name
    _write("file_begin", file=name, first_nodeid=item.nodeid[:512],
           **_cached_qt_counts(item.config))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Sync failure details before a later native crash skips pytest's summary."""
    if report.failed:
        detail = report.longreprtext
        limit = 65536
        _write(
            "test_failure",
            nodeid=report.nodeid,
            when=report.when,
            detail=detail[:limit],
            detail_chars=len(detail),
            detail_truncated=len(detail) > limit,
        )
    if report.when == "teardown":
        _arm_fault_log()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Mark a normal terminal result; abrupt exits deliberately lack it."""
    global _active_file, _completed_files
    if _active_file is not None:
        _write("file_end", file=_active_file,
               **_cached_qt_counts(session.config))
        _completed_files += 1
        _active_file = None
    _write(
        "session_finish", exitstatus=int(exitstatus), completed_files=_completed_files
    )


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Keep the descriptor valid through teardown, then restore fatal signals."""
    global _fault_file, _prior_fault_enabled
    if _fault_file is None:
        return
    try:
        if _prior_fault_enabled:
            stream = sys.__stderr__ or sys.stderr
            _real_enable(file=stream, all_threads=True)
        else:
            _real_disable()
    except (OSError, ValueError):
        return
    _fault_file.close()
    _fault_file = None
    _prior_fault_enabled = None
