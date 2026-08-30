"""The console/file log plumbing's remaining seams: idempotent attachment,
foreign filters, and a verbose toggle with nowhere to write.

Every test here runs inside the ``vlog`` fixture, which snapshots the
module's process-wide singletons (``_handler``, ``_relay``, ``_console_ref``,
``_file_handler``) and every handler list and level it touches on the real
spaCR loggers, then puts them back. Without that, one test's gate decides
what every later test in the process is allowed to log.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture()
def vlog(tmp_path, monkeypatch):
    """The module with its singletons snapshotted and its log dir redirected.

    ``SPACR_LOG_DIR`` keeps ``_ensure_file_handler`` out of the operator's
    real ``~/.spacr/logs``; the save/restore keeps a test that re-gates the
    console from silencing the rest of the run.
    """
    from spacr.qt import verbose_logger as module

    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    saved = {name: getattr(module, name) for name in
             ("_console_ref", "_handler", "_relay", "_file_handler")}
    saved_handlers = {name: list(logging.getLogger(name).handlers)
                      for name in module._ATTACHED_LOGGERS}
    saved_levels = {name: logging.getLogger(name).level
                    for name in module._ATTACHED_LOGGERS}
    try:
        yield module
    finally:
        # apply_verbose_logging(True) installs a process-wide sys.setprofile
        # hook; leaving it on makes every later test log its own call graph.
        from spacr.logging_util import disable_function_trace
        disable_function_trace()
        for name, value in saved.items():
            setattr(module, name, value)
        for name, handlers in saved_handlers.items():
            logger = logging.getLogger(name)
            logger.handlers[:] = handlers
            logger.setLevel(saved_levels[name])


def _record(name: str, level: int, message: str = "m") -> logging.LogRecord:
    """One record to push through a handler's filter chain."""
    return logging.LogRecord(name, level, __file__, 1, message, (), None)


def test_a_descendant_record_reaches_the_file_handler_exactly_once(
        vlog, tmp_path, monkeypatch):
    """The log file a user attaches to a bug report must read once per event.

    Python logging invokes a handler once at the emitting logger and again at
    every ancestor carrying the same object. The sink therefore belongs only
    on ``spacr``; descendants propagate to it and must not carry another copy.
    """
    from logging.handlers import RotatingFileHandler

    log_path = tmp_path / "logs" / "already-attached.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    already = RotatingFileHandler(str(log_path), encoding="utf-8")
    spacr_logger = logging.getLogger("spacr")
    spacr_logger.addHandler(already)
    spacr_logger.setLevel(logging.NOTSET)
    vlog._file_handler = None
    monkeypatch.setattr(vlog, "RotatingFileHandler",
                        lambda *args, **kwargs: already)

    handler = vlog._ensure_file_handler()

    assert handler is already
    assert logging.getLogger("spacr").handlers.count(already) == 1
    for name in vlog._ATTACHED_LOGGERS[1:]:
        assert already not in logging.getLogger(name).handlers
    # The skipped logger still went through the level line below the guard.
    assert spacr_logger.level == logging.INFO

    logging.getLogger("spacr.qt.hf_download").info("one event, one line")
    already.flush()
    assert log_path.read_text().count("one event, one line") == 1

    spacr_logger.removeHandler(already)
    already.close()


def test_the_console_forwarder_is_attached_only_at_the_package_root(vlog):
    """A descendant console record must follow the same one-sink topology."""
    vlog._handler = None

    handler = vlog._ensure_handler()

    assert logging.getLogger("spacr").handlers.count(handler) == 1
    for name in vlog._ATTACHED_LOGGERS[1:]:
        assert handler not in logging.getLogger(name).handlers


def test_a_foreign_console_filter_survives_being_re_gated(vlog):
    """A filter somebody else installed must keep vetoing after a re-gate.

    ``apply_console_levels`` scans the handler's filters for its own
    ``LevelSetFilter`` and updates it in place. Anything else on the chain --
    a filter a screen added to keep one noisy logger out of its console --
    has to be stepped over, not overwritten and not dropped. If the scan
    stopped at the first filter it did not recognise, the level preference
    would silently stop applying; if it replaced the chain, the noisy logger
    would come flooding back into the console the user muted it from.
    """
    from spacr.logging_util import LevelSetFilter

    class _NotFromTheQtLayer(logging.Filter):
        def filter(self, record):
            return record.name != "spacr.qt"

    vlog._handler = None
    handler = vlog._ensure_handler()
    handler.filters[:] = []
    foreign = _NotFromTheQtLayer()
    handler.addFilter(foreign)

    vlog.apply_console_levels([logging.WARNING, logging.ERROR])
    # A second pass has to find its own filter *behind* the foreign one.
    vlog.apply_console_levels([logging.WARNING])

    assert foreign in handler.filters
    assert sum(isinstance(f, LevelSetFilter) for f in handler.filters) == 1
    assert vlog.console_levels() == frozenset({logging.WARNING})
    # The foreign veto still bites, and the new gate decides everything else.
    assert bool(handler.filter(_record("spacr.trace", logging.WARNING))) is True
    assert bool(handler.filter(_record("spacr.qt", logging.WARNING))) is False
    assert bool(handler.filter(_record("spacr.trace", logging.ERROR))) is False
    assert handler.level == logging.DEBUG


def test_the_console_gate_is_readable_from_behind_a_foreign_filter(vlog):
    """"What is the console showing?" must not depend on filter order.

    The Preferences dialog renders its level checkboxes from
    ``console_levels()``. A screen that added a filter of its own ahead of
    the level gate would, if the scan gave up on the first stranger, make
    that dialog draw every box unticked while the console went on showing
    warnings -- a preference page that lies about the state it is editing.
    """
    from spacr.logging_util import LevelSetFilter

    vlog._handler = None
    handler = vlog._ensure_handler()
    handler.filters[:] = []
    handler.addFilter(logging.Filter("spacr.trace"))

    # No gate behind the stranger yet: nothing is being shown deliberately.
    assert vlog.console_levels() == frozenset()

    handler.addFilter(LevelSetFilter({logging.DEBUG, logging.INFO}))

    assert vlog.console_levels() == frozenset({logging.DEBUG, logging.INFO})
    assert isinstance(handler.filters[0], logging.Filter)
    assert not isinstance(handler.filters[0], LevelSetFilter)


def test_verbose_still_flips_when_the_log_file_cannot_be_opened(
        vlog, monkeypatch):
    """A read-only log directory must not cost the user verbose logging.

    ``_ensure_file_handler`` answers ``None`` when the file will not open --
    a read-only home, a full disk, a locked file on Windows. Everything
    downstream of that in ``apply_verbose_logging`` still has to happen: the
    console forwarder and the spaCR loggers go to DEBUG and the function
    trace comes on, because the console is exactly where a user with an
    unwritable log directory has to read their diagnostics.
    """
    from logging.handlers import RotatingFileHandler

    vlog._handler = None
    vlog._file_handler = None
    vlog._console_ref = None

    vlog.apply_verbose_logging(False)
    opened = vlog._file_handler
    assert isinstance(opened, RotatingFileHandler)
    assert opened.level == logging.INFO
    assert vlog._handler.level == logging.INFO
    assert vlog.is_verbose() is False

    for name in vlog._ATTACHED_LOGGERS:
        logging.getLogger(name).removeHandler(opened)
    opened.close()

    def _refuse(*args, **kwargs):
        raise OSError("read-only file system")

    vlog._file_handler = None
    monkeypatch.setattr(vlog, "RotatingFileHandler", _refuse)

    vlog.apply_verbose_logging(True)
    try:
        assert vlog._file_handler is None
        assert vlog._handler.level == logging.DEBUG
        assert vlog.is_verbose() is True
        for name in vlog._ATTACHED_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG
    finally:
        vlog.apply_verbose_logging(False)

    assert vlog.is_verbose() is False
