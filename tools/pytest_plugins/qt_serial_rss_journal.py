"""Durable file-boundary RSS and failure evidence for a serial Qt run.

Load explicitly with ``-p tools.pytest_plugins.qt_serial_rss_journal`` and set
``SPACR_QT_SERIAL_RSS_JOURNAL`` to a new JSONL path. Every record is written
and synced before pytest continues, so a later memory-guard ``os._exit`` does
not discard the boundaries or reported failures it already crossed. The
observer neither touches Qt objects nor changes test order, garbage collection,
or event processing.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

_journal: Path | None = None
_active_file: str | None = None
_completed_files = 0


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


def pytest_configure(config: pytest.Config) -> None:
    """Create a fresh journal so separate acceptance attempts cannot mix."""
    global _journal, _active_file, _completed_files
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
    )


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


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Record the prior file's completed teardown before starting this one."""
    global _active_file, _completed_files
    name = os.path.relpath(str(item.path), item.config.rootpath)
    if name == _active_file:
        return
    if _active_file is not None:
        _write("file_end", file=_active_file)
        _completed_files += 1
    _active_file = name
    _write("file_begin", file=name, first_nodeid=item.nodeid[:512])


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Sync failure details before a later native crash skips pytest's summary."""
    if not report.failed:
        return
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


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Mark a normal terminal result; abrupt exits deliberately lack it."""
    global _active_file, _completed_files
    if _active_file is not None:
        _write("file_end", file=_active_file)
        _completed_files += 1
        _active_file = None
    _write(
        "session_finish", exitstatus=int(exitstatus), completed_files=_completed_files
    )
