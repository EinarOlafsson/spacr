"""A failed diagnostic write or portable marker cannot interrupt a run."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from spacr import logging_util as lu


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
