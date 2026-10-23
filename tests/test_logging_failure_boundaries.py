"""A failed diagnostic write or portable marker cannot interrupt a run."""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr import logging_util as lu


@pytest.fixture
def paced_file_handler(tmp_path):
    """Use a real owned file handler and retire every timer it starts."""
    path = tmp_path / "paced.log"
    handler = lu._quicken(logging.handlers.RotatingFileHandler(path))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield handler, path
    finally:
        lu._drain_package_logs()
        root.removeHandler(handler)
        handler.close()


def _write_record(handler, message, level=logging.INFO):
    """Send a record through the real locking and emission path."""
    handler.handle(logging.LogRecord("spacr.test", level, __file__, 1,
                                     message, (), None))


def test_explicit_flush_writes_the_tail_and_boundary_drains_owned_timers(
        paced_file_handler, monkeypatch):
    """A pending burst cannot retain its timer past the run boundary."""
    handler, path = paced_file_handler
    monkeypatch.setattr(lu, "_FLUSH_INTERVAL_S", 60)
    _write_record(handler, "first")
    _write_record(handler, "buffered tail")
    timer = handler._spacr_late_flush
    assert timer is not None and timer.is_alive()
    assert path.read_text() == "first\n"
    handler.flush()
    assert path.read_text() == "first\nbuffered tail\n"
    lu._drain_package_logs()
    assert not timer.is_alive()
    assert handler.stream is not None and not handler.stream.closed


def test_close_invalidates_an_awakened_callback_without_joining_under_lock(
        paced_file_handler, monkeypatch):
    """Closing under a logging-shutdown lock must not deadlock or reopen."""
    handler, path = paced_file_handler
    original = lu._late_flush
    entered = threading.Event()
    stale_flush = threading.Event()

    def waiting_callback(owner, real_flush, token):
        """Signal the real timer expiry before it acquires the handler lock."""
        def checked_flush():
            """Detect any stale write after the close has invalidated it."""
            stale_flush.set()
            real_flush()

        entered.set()
        original(owner, checked_flush, token)

    monkeypatch.setattr(lu, "_late_flush", waiting_callback)
    monkeypatch.setattr(lu, "_FLUSH_INTERVAL_S", .01)
    with handler.lock:
        _write_record(handler, "first")
        _write_record(handler, "tail before close")
        timer = handler._spacr_late_flush
        assert timer is not None
        assert entered.wait(5)
        handler.close()
        assert handler.stream is None
        assert path.read_text() == "first\ntail before close\n"
    timer.join(5)
    assert not timer.is_alive()
    assert not stale_flush.is_set()
    assert handler.stream is None
    assert path.read_text() == "first\ntail before close\n"
    handler.close()


def test_warning_flushes_the_buffered_burst_immediately(
        paced_file_handler, monkeypatch):
    """Warnings retain the earlier INFO tail and cancel the delayed write."""
    handler, path = paced_file_handler
    monkeypatch.setattr(lu, "_FLUSH_INTERVAL_S", 60)
    _write_record(handler, "first")
    _write_record(handler, "tail")
    timer = handler._spacr_late_flush
    _write_record(handler, "warning", logging.WARNING)
    assert path.read_text() == "first\ntail\nwarning\n"
    lu._drain_package_logs()
    assert not timer.is_alive()


def test_a_changed_process_id_does_not_reuse_or_join_parent_timers(
        paced_file_handler, monkeypatch):
    """The child-side bookkeeping owns only timers created in that process."""
    handler, path = paced_file_handler
    monkeypatch.setattr(lu, "_FLUSH_INTERVAL_S", 60)
    _write_record(handler, "parent first")
    _write_record(handler, "parent tail")
    parent_timer = handler._spacr_late_flush
    try:
        with monkeypatch.context() as patch:
            patch.setattr(lu, "os", SimpleNamespace(getpid=lambda: os.getpid() + 1))
            _write_record(handler, "child tail")
            child_timer = handler._spacr_late_flush
            assert child_timer is not parent_timer
            lu._drain_package_logs()
            assert not child_timer.is_alive()
            assert parent_timer.is_alive()
            assert path.read_text() == "parent first\nparent tail\nchild tail\n"
    finally:
        parent_timer.cancel()
        parent_timer.join(5)
    assert not parent_timer.is_alive()


def test_a_failed_handler_does_not_prevent_other_tails_or_timer_retirement(
        paced_file_handler, monkeypatch, tmp_path):
    """One real file's flush failure leaves the remaining handler durable."""
    handler, path = paced_file_handler
    monkeypatch.setattr(lu, "_FLUSH_INTERVAL_S", 60)
    failed = lu._quicken(logging.handlers.RotatingFileHandler(tmp_path / "bad.log"))
    root = logging.getLogger()
    root.removeHandler(handler)
    root.addHandler(failed)
    root.addHandler(handler)
    stream = failed.stream

    class UnavailableFlush:
        """Delegate the real stream except for its failed flush operation."""
        def __getattr__(self, name):
            return getattr(stream, name)

        def flush(self):
            raise OSError("diagnostic volume unavailable")

    try:
        for target in (failed, handler):
            _write_record(target, "first")
            _write_record(target, "tail")
        timers = [failed._spacr_late_flush, handler._spacr_late_flush]
        failed.stream = UnavailableFlush()
        with pytest.raises(OSError, match="diagnostic volume unavailable"):
            lu._drain_package_logs()
        assert all(not timer.is_alive() for timer in timers)
        assert path.read_text() == "first\ntail\n"
    finally:
        failed.stream = stream
        root.removeHandler(failed)
        failed.close()



def test_boundary_drain_flushes_a_burst_after_its_timer_cannot_start(
        paced_file_handler, monkeypatch):
    """A failed native thread start cannot lose a tail or cause an invalid join."""
    handler, path = paced_file_handler
    monkeypatch.setattr(lu, "_FLUSH_INTERVAL_S", 60)
    _write_record(handler, "first")
    errors = []
    monkeypatch.setattr(handler, "handleError", errors.append)

    original_start = threading.Thread.start

    def no_native_thread(thread):
        if thread is handler._spacr_late_flush:
            raise RuntimeError("can't start new thread")
        return original_start(thread)

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", no_native_thread)
        _write_record(handler, "tail without a timer thread")
    timer = handler._spacr_late_flush
    assert isinstance(timer, threading.Timer)
    assert timer.ident is None
    assert not timer.is_alive()
    assert timer in handler._spacr_flush_timers
    assert [record.getMessage() for record in errors] == [
        "tail without a timer thread"]
    assert path.read_text() == "first\n"

    lu._drain_package_logs()

    assert path.read_text() == "first\ntail without a timer thread\n"
    assert handler._spacr_late_flush is None
    assert handler._spacr_flush_token is None
    assert timer.finished.is_set()
    assert timer.ident is None
    assert handler.stream is not None and not handler.stream.closed


def test_a_non_rotating_handler_is_not_rewired():
    handler = logging.StreamHandler()
    assert lu._quicken(handler) is handler
    assert not hasattr(handler, "_spacr_late_flush_pid")


def test_a_failed_late_flush_releases_the_next_burst(monkeypatch):
    handler = SimpleNamespace(_spacr_late_flush_pid=os.getpid(),
                              _spacr_flushed_at=0.0)
    monkeypatch.setattr(lu.time, "monotonic", lambda: 12.5)

    def unavailable():
        raise OSError("log directory disappeared")

    lu._late_flush(handler, unavailable)
    assert handler._spacr_late_flush_pid == 0
    assert handler._spacr_flushed_at == 12.5


def test_a_warning_flush_failure_reports_the_record_and_keeps_it_emitted():
    emitted, errors = [], []
    handler = SimpleNamespace(_spacr_flushed_at=0.0,
                              handleError=errors.append)
    record = logging.LogRecord("spacr", logging.WARNING, __file__, 1,
                               "important warning", (), None)

    def unavailable():
        raise OSError("log directory disappeared")

    lu._emit_flushing_the_loud(handler, emitted.append, unavailable, record)
    assert emitted == [record]
    assert errors == [record]
    assert handler._spacr_flushed_at > 0


def test_rollover_opens_a_missing_stream_and_uses_its_tell_fallback():
    opened, checked = [], []

    class Stream:
        def tell(self):
            return lu._NEAR_LIMIT_BYTES

    class Handler:
        maxBytes = lu._NEAR_LIMIT_BYTES + 1
        stream = None

        def _open(self):
            opened.append(True)
            return Stream()

    handler = Handler()
    record = logging.LogRecord("spacr", logging.INFO, __file__, 1,
                               "next", (), None)
    assert lu._quick_should_rollover(
        handler, lambda current: checked.append(current) or True,
        record) is True
    assert opened == [True]
    assert checked == [record]


def test_an_unreadable_portable_marker_does_not_hide_a_later_one(
        tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / lu._PORTABLE_MARKER).touch()
    original_is_file = Path.is_file

    def marker_is_file(path):
        if path == first / lu._PORTABLE_MARKER:
            raise OSError("marker mount unavailable")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", marker_is_file)
    monkeypatch.setattr(lu, "_app_folders", lambda: [first, second])
    lu._portable_root_for.cache_clear()
    try:
        assert lu._portable_root_for("", "") == second
    finally:
        lu._portable_root_for.cache_clear()


def test_launcher_and_interpreter_in_one_folder_make_one_portable_candidate(
        monkeypatch):
    executable_folder = Path(os.path.abspath(sys.executable)).parent
    monkeypatch.setenv("SPACR_LAUNCHER_DIR", str(executable_folder))
    folders = lu._app_folders()
    assert folders.count(executable_folder) == 1


def test_portable_mode_still_exports_paths_when_data_cannot_be_created(
        tmp_path, monkeypatch):
    root = tmp_path / "portable"
    data = root / lu._PORTABLE_DATA
    monkeypatch.setattr(lu, "_portable_root", lambda: root)
    names = ("SPACR_HOME", "SPACR_LOG_DIR", "SPACR_BACKENDS_DIR",
             "SPACR_PLUGIN_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
             "TORCH_HOME", "HF_HOME", "MPLCONFIGDIR",
             "CELLPOSE_LOCAL_MODELS_PATH")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    original_mkdir = Path.mkdir

    def read_only(path, *args, **kwargs):
        if path == data:
            raise OSError("read-only portable drive")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", read_only)
    try:
        assert lu._apply_portable_mode() == data
        assert os.environ["SPACR_HOME"] == str(data)
        assert os.environ["SPACR_LOG_DIR"] == str(data / "logs")
        assert not data.exists()
    finally:
        for name in names:
            os.environ.pop(name, None)
