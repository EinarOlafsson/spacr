"""Item 638: verbose logging is usable, headless.

The maintainer: "verbose logging last time I used it would crash the
program; make sure this feature is usable".

* A short Measure run with the module's ``verbose`` setting on and spaCR's
  log at DEBUG finishes and writes its measurements.
* Library loggers stay at WARNING while spaCR's own DEBUG is recorded: with
  the root logger at DEBUG every library's DEBUG reached the log files.
* The verbose call trace formats a bounded repr, however large the value.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time

import pytest

from spacr import logging_util as lu


@pytest.fixture
def logging_sandbox(tmp_path, monkeypatch):
    """Fresh spaCR logging into ``tmp_path``; process-wide state put back."""
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    root = logging.getLogger()
    spacr_logger = logging.getLogger("spacr")
    saved = {
        "handlers": list(root.handlers),
        "root_level": root.level,
        "spacr_level": spacr_logger.level,
        "initialised": lu._INITIALISED,
        "session_level": lu._SESSION_LEVEL,
        "log_path": lu._LOG_PATH,
        "file_filter": lu._FILE_FILTER,
        "level_handlers": dict(lu._LEVEL_HANDLERS),
        "sys_profile": sys.getprofile(),
    }
    lu._INITIALISED = False
    lu._FILE_FILTER = None
    lu._LEVEL_HANDLERS.clear()
    try:
        yield tmp_path / "logs"
    finally:
        for handler in list(root.handlers):
            if handler not in saved["handlers"]:
                root.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
        root.handlers[:] = saved["handlers"]
        root.setLevel(saved["root_level"])
        spacr_logger.setLevel(saved["spacr_level"])
        lu._INITIALISED = saved["initialised"]
        lu._SESSION_LEVEL = saved["session_level"]
        lu._LOG_PATH = saved["log_path"]
        lu._FILE_FILTER = saved["file_filter"]
        lu._LEVEL_HANDLERS.clear()
        lu._LEVEL_HANDLERS.update(saved["level_handlers"])
        sys.setprofile(saved["sys_profile"])


def _log_text(log_dir) -> str:
    """Everything the master log file holds, handlers flushed first."""
    for handler in logging.getLogger().handlers:
        handler.flush()
    return (log_dir / "spacr.log").read_text(encoding="utf-8")


def test_a_verbose_measure_run_finishes(logging_sandbox, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from spacr.measure import measure_crop
    from spacr.qt import synthetic as syn

    lu.setup_logging(level=logging.DEBUG,
                     log_file=logging_sandbox / "spacr.log")
    layout = syn.generate_measure_demo(
        tmp_path / "plate", wells=("A01",), fields=1)
    settings = syn.demo_settings("measure", str(layout.src))
    settings.update(verbose=True, n_jobs=1, plot=False)
    measure_crop(settings)

    db = layout.src / "measurements" / "measurements.db"
    with sqlite3.connect(db) as con:
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "cell" in tables
    assert "spacr.measure" in _log_text(logging_sandbox)


def test_library_debug_stays_out_of_a_verbose_log(logging_sandbox):
    lu.setup_logging(level=logging.DEBUG,
                     log_file=logging_sandbox / "spacr.log")
    lu.apply_level_policy(lu.LEVELS, lu.LEVELS)
    for name in ("httpcore.http11", "huggingface_hub.file_download",
                 "filelock", "numexpr.utils"):
        logging.getLogger(name).debug("library chatter")
        logging.getLogger(name).info("library chatter")
    logging.getLogger("spacr.core").debug("spaCR detail")
    logging.getLogger("filelock").warning("library warning")

    text = _log_text(logging_sandbox)
    assert "library chatter" not in text
    assert "spaCR detail" in text
    assert "library warning" in text
    assert logging.getLogger().level == logging.WARNING


def test_the_root_follows_a_quieter_choice(logging_sandbox):
    lu.setup_logging(level=logging.ERROR,
                     log_file=logging_sandbox / "spacr.log")
    assert logging.getLogger().level == logging.ERROR
    assert logging.getLogger("spacr").level == logging.ERROR


def test_the_call_trace_formats_a_bounded_repr():
    from spacr.qt.verbose_logger import _brief

    huge = {"values": list(range(5_000_000)), "text": "x" * 5_000_000}
    started = time.perf_counter()
    brief = _brief(huge)
    assert time.perf_counter() - started < 0.5
    assert len(brief) <= 240
    assert brief.startswith("{")


def test_a_traced_pipeline_returning_a_huge_value_is_cheap(caplog):
    from spacr.qt import verbose_logger as vl

    payload = [list(range(1000)) for _ in range(2000)]
    traced = vl.log_call(lambda settings: payload)
    was = vl._verbose
    vl._verbose = True
    try:
        with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
            started = time.perf_counter()
            assert traced({"src": os.sep}) is payload
            elapsed = time.perf_counter() - started
    finally:
        vl._verbose = was
    assert elapsed < 0.5
    lines = [r.getMessage() for r in caplog.records
             if r.name == "spacr.trace"]
    assert len(lines) == 2
    assert all(len(line) < 600 for line in lines)
