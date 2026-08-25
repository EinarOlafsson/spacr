"""Console/file log plumbing: the sink's refusals, and the trace helpers.

The module keeps process-wide singletons (``_handler``, ``_relay``,
``_console_ref``, ``_file_handler``) and attaches handlers to the real
spaCR loggers, so every test runs inside a fixture that snapshots those
globals and detaches anything it added.
"""
from __future__ import annotations

import logging
import threading

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


class Widget:
    """A class at module scope: its methods' qualnames are ``Widget.method``.

    Defined here rather than inside a test because ``_label_for`` and
    ``_looks_bound`` both read ``__qualname__``, and a class built inside a
    function carries a ``<locals>`` prefix that is not the class name.
    """

    def press(self, which="Run"):
        return f"pressed {which}"


def free(which="Run"):
    """A plain function, whose qualname has no dot at all."""
    return f"called {which}"


class Weakrefable:
    """A do-nothing object that a ``weakref.ref`` can point at."""


@pytest.fixture()
def vlog(tmp_path, monkeypatch):
    """The module with its singletons snapshotted and its log dir redirected."""
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
        # hook. Leaving it on makes every later test log its own call graph
        # into spacr.trace, and it outlives the interpreter's own teardown.
        from spacr.logging_util import disable_function_trace
        disable_function_trace()
        for name, value in saved.items():
            setattr(module, name, value)
        for name, handlers in saved_handlers.items():
            logger = logging.getLogger(name)
            logger.handlers[:] = handlers
            logger.setLevel(saved_levels[name])


def test_the_log_file_lands_under_the_override_directory(vlog, tmp_path):
    """SPACR_LOG_DIR redirects today's rotating log."""
    path = vlog.current_log_file()

    assert path.parent == tmp_path / "logs"
    assert path.parent.is_dir()
    assert path.name.startswith("spacr-") and path.name.endswith(".log")


def test_without_the_override_the_log_lives_under_the_home_directory(
        vlog, tmp_path, monkeypatch):
    """The default is ``~/.spacr/logs``, created on demand."""
    from pathlib import Path

    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.delenv("SPACR_LOG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    assert vlog.log_dir() == fake_home / ".spacr" / "logs"
    assert (fake_home / ".spacr" / "logs").is_dir()


def test_a_file_handler_that_cannot_be_opened_is_skipped_not_fatal(
        vlog, monkeypatch):
    """An unopenable log file must not stop the application starting."""
    def _refuse(*args, **kwargs):
        raise PermissionError("the log directory is read-only")

    vlog._file_handler = None
    monkeypatch.setattr(vlog, "RotatingFileHandler", _refuse)

    assert vlog._ensure_file_handler() is None
    assert vlog._file_handler is None


def test_the_file_handler_attaches_once_to_every_spacr_logger(vlog):
    """Idempotent: a second call hands back the handler already attached."""
    vlog._file_handler = None

    first = vlog._ensure_file_handler()
    second = vlog._ensure_file_handler()

    assert first is second
    for name in vlog._ATTACHED_LOGGERS:
        assert logging.getLogger(name).handlers.count(first) == 1


def test_a_relay_built_on_a_worker_ends_up_on_the_gui_thread(vlog, qapp):
    """Affinity must not depend on which thread happened to log first."""
    from PySide6.QtCore import QThread

    built = {}

    def _build():
        built["born_on"] = QThread.currentThread()
        built["relay"] = vlog._ConsoleRelay()

    worker = threading.Thread(target=_build, name="a-worker")
    worker.start()
    worker.join()

    relay = built["relay"]
    assert built["born_on"] != qapp.thread()
    assert relay.thread() == qapp.thread()
    relay.deleteLater()


def test_a_relay_built_on_the_gui_thread_stays_put(vlog, qapp):
    """No pointless hop when the object is already where it belongs."""
    relay = vlog._ConsoleRelay()

    assert relay.thread() == qapp.thread()
    relay.deleteLater()


def test_a_record_produced_by_a_console_write_is_dropped(vlog, qapp):
    """Delivering it would re-enter the widget mid-``setPlainText``."""
    delivered = []

    class _Panel:
        def append_stdout(self, text):
            delivered.append(text)

    panel = _Panel()
    vlog._console_ref = lambda: panel
    relay = vlog._ConsoleRelay()

    relay._deliver("outside a write\n")
    with vlog.console_write():
        relay._deliver("inside a write\n")
    relay._deliver("after the write\n")

    assert delivered == ["outside a write\n", "after the write\n"]
    relay.deleteLater()


def test_the_latch_restores_the_previous_depth_not_zero(vlog):
    """``console_write`` is re-entrant: unwinding one level is not clearing."""
    assert vlog.console_write_in_progress() is False
    with vlog.console_write():
        with vlog.console_write():
            assert vlog.console_write_in_progress() is True
        assert vlog.console_write_in_progress() is True
    assert vlog.console_write_in_progress() is False


def test_a_line_for_a_collected_console_is_dropped(vlog, qapp):
    """A closed screen must not be resurrected by a late record."""
    vlog._console_ref = lambda: None
    relay = vlog._ConsoleRelay()

    relay._deliver("nobody is listening\n")

    assert vlog._console_ref is not None
    relay.deleteLater()


def test_no_registered_console_at_all_is_not_an_error(vlog, qapp):
    """``_console_ref`` is None before any panel registers."""
    vlog._console_ref = None
    relay = vlog._ConsoleRelay()

    relay._deliver("nobody has registered\n")

    relay.deleteLater()


def test_a_deleted_qwidget_target_is_forgotten_not_called(vlog, qapp):
    """A live Python wrapper around a dead C++ object must not be touched."""
    import shiboken6
    from PySide6.QtWidgets import QWidget

    calls = []

    class _Panel(QWidget):
        def append_stdout(self, text):
            calls.append(text)

    panel = _Panel()
    ref = vlog.weakref.ref(panel)
    vlog._console_ref = ref
    relay = vlog._ConsoleRelay()
    shiboken6.delete(panel)
    assert shiboken6.isValid(panel) is False

    relay._deliver("into the void\n")

    assert calls == []
    assert vlog._console_ref is None
    relay.deleteLater()


def test_a_target_without_append_stdout_is_not_called(vlog, qapp):
    """Anything can be registered; only a real console gets written to."""
    class _NotAConsole:
        pass

    target = _NotAConsole()
    vlog._console_ref = lambda: target
    relay = vlog._ConsoleRelay()

    relay._deliver("nothing to append to\n")

    relay.deleteLater()


def test_a_console_that_raises_does_not_take_the_app_down(vlog, qapp):
    """A logging failure must never escape into the application."""
    class _Broken:
        def append_stdout(self, text):
            raise RuntimeError("the document is gone")

    target = _Broken()
    vlog._console_ref = lambda: target
    relay = vlog._ConsoleRelay()

    relay._deliver("boom\n")

    relay.deleteLater()


def test_the_forwarder_drops_records_made_during_a_console_write(vlog, qapp):
    """The loop is cut in ``emit`` too, before formatting costs anything."""
    formatted = []

    class _Counting(vlog._ConsoleForwarder):
        def format(self, record):
            formatted.append(record)
            return super().format(record)

    handler = _Counting()
    handler.setFormatter(logging.Formatter("%(message)s"))
    target = object()
    vlog._console_ref = lambda: target
    record = logging.LogRecord("spacr", logging.INFO, __file__, 1,
                               "hello", (), None)

    with vlog.console_write():
        handler.emit(record)

    assert formatted == []


def test_the_forwarder_says_nothing_when_no_console_is_registered(vlog, qapp):
    """No panel means no work, not an exception."""
    handler = vlog._ConsoleForwarder()
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("spacr", logging.INFO, __file__, 1,
                               "hello", (), None)

    vlog._console_ref = None
    handler.emit(record)
    vlog._console_ref = lambda: None
    handler.emit(record)


def test_a_formatting_failure_inside_the_forwarder_is_swallowed(vlog, qapp):
    """Never let a logging failure escape into the app."""
    class _BrokenFormat(vlog._ConsoleForwarder):
        def format(self, record):
            raise ValueError("bad format string")

    handler = _BrokenFormat()
    target = object()
    vlog._console_ref = lambda: target
    record = logging.LogRecord("spacr", logging.INFO, __file__, 1,
                               "hello", (), None)

    handler.emit(record)


def test_the_forwarder_reaches_the_registered_console(vlog, qapp):
    """The happy path: emit formats and the relay delivers to the panel."""
    delivered = []

    class _Panel:
        def append_stdout(self, text):
            delivered.append(text)

    panel = _Panel()
    vlog._relay = None
    vlog._console_ref = lambda: panel
    handler = vlog._ConsoleForwarder()
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("spacr", logging.INFO, __file__, 1,
                               "hello", (), None)

    handler.emit(record)

    assert delivered == ["hello\n"]
    if vlog._relay is not None:
        vlog._relay.deleteLater()


def test_console_levels_is_empty_before_any_handler_exists(vlog):
    """Nothing is being shown when the handler has not been built."""
    vlog._handler = None

    assert vlog.console_levels() == frozenset()


def test_console_levels_reports_what_the_gate_was_set_to(vlog):
    """``apply_console_levels`` and ``console_levels`` are a round trip."""
    vlog._handler = None

    vlog.apply_console_levels([logging.WARNING, logging.ERROR])
    first = vlog.console_levels()
    vlog.apply_console_levels([logging.DEBUG])
    second = vlog.console_levels()

    assert first == frozenset({logging.WARNING, logging.ERROR})
    assert second == frozenset({logging.DEBUG})
    assert vlog._handler.level == logging.DEBUG


def test_console_levels_is_empty_when_the_handler_carries_no_gate(vlog):
    """A handler built but never gated is not showing a filtered set."""
    vlog._handler = None
    handler = vlog._ensure_handler()
    handler.filters[:] = []

    assert vlog.console_levels() == frozenset()


def test_is_verbose_follows_the_handler_level(vlog):
    """Verbose is DEBUG on the console forwarder and nothing else."""
    vlog._handler = None
    assert vlog.is_verbose() is False

    vlog.apply_verbose_logging(True)
    assert vlog.is_verbose() is True

    vlog.apply_verbose_logging(False)
    assert vlog.is_verbose() is False


def test_a_button_press_logs_nothing_when_verbose_is_off(vlog, caplog):
    """Decorated call sites must cost nothing with verbose off."""
    vlog._handler = None

    with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
        vlog.log_button_press("Run", {"src": "/data"})

    assert caplog.text == ""


def test_a_button_press_without_context_still_says_which_button(vlog, caplog):
    """The bare form records the press itself."""
    vlog._handler = None
    vlog.apply_verbose_logging(True)

    with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
        vlog.log_button_press("Run")
        vlog.log_button_press("Stop", {"src": "/data"})

    messages = [r.getMessage() for r in caplog.records]
    assert "[button:Run] pressed" in messages
    assert any("[button:Stop]" in m and "/data" in m for m in messages)


def test_log_call_hides_self_only_for_a_real_method(vlog, caplog):
    """``self`` is dropped from a bound method's args and nowhere else."""
    vlog._handler = None
    vlog.apply_verbose_logging(True)
    wrapped_method = vlog.log_call(Widget.press)
    wrapped_free = vlog.log_call(free)

    with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
        assert wrapped_method(Widget(), "Run") == "pressed Run"
        assert wrapped_free("Run") == "called Run"

    lines = [r.getMessage() for r in caplog.records]
    method_args = next(m for m in lines if m.startswith("[Widget.press] args="))
    free_args = next(m for m in lines if m.startswith("[free] args="))
    assert method_args.startswith("[Widget.press] args=('Run',)")
    assert free_args.startswith("[free] args=('Run',)")
    assert "-> 'pressed Run'" in "\n".join(lines)


def test_log_call_forwards_untouched_when_verbose_is_off(vlog, caplog):
    """With verbose off the wrapper is one attribute check and a forward."""
    vlog._handler = None
    wrapped = vlog.log_call(free)

    with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
        assert wrapped("Stop") == "called Stop"

    assert caplog.records == []


def test_looks_bound_needs_a_matching_class_name(vlog):
    """A qualname without a dot cannot be a method, so nothing is hidden."""
    assert vlog._looks_bound(free, ("a",)) is False
    assert vlog._looks_bound(free, ()) is False
    assert vlog._looks_bound(Widget.press, (Widget(), "a")) is True
    assert vlog._looks_bound(Widget.press, ("not a Widget", "a")) is False


def test_log_call_records_a_raise_and_re_raises_it(vlog, caplog):
    """The trace must show the exception, and the caller must still see it."""
    vlog._handler = None
    vlog.apply_verbose_logging(True)

    @vlog.log_call
    def explode():
        raise ValueError("no such column")

    with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
        with pytest.raises(ValueError, match="no such column"):
            explode()

    assert any("RAISED ValueError: no such column" in r.getMessage()
               for r in caplog.records)


def test_brief_truncates_a_giant_repr(vlog):
    """A 100-entry settings dict must not wreck the console."""
    text = vlog._brief({f"key{i}": "x" * 20 for i in range(100)})

    # 240 is the cap, not the length: three characters are reserved for the
    # marker and a single-character ellipsis is written into them.
    assert len(text) == 238
    assert text.endswith("…")
    assert vlog._brief("short") == "'short'"
    assert len(vlog._brief("y" * 500, max_chars=20)) == 18


def test_brief_survives_a_value_whose_repr_raises(vlog):
    """A broken ``__repr__`` is described rather than propagated."""
    class Hostile:
        def __repr__(self):
            raise RuntimeError("no repr for you")

    assert vlog._brief(Hostile()) == "<Hostile — repr failed>"


def test_the_label_is_the_qualified_name(vlog):
    """A bound method is labelled ``Class.method``; a function by its name."""
    assert vlog._label_for(Widget.press, ()) == "Widget.press"
    assert vlog._label_for(free, ()) == "free"


def test_dropping_a_stale_target_leaves_a_newer_one_alone(vlog):
    """A destroyed panel's callback must not unregister its replacement."""
    old = Weakrefable()
    new = Weakrefable()
    old_ref = vlog.weakref.ref(old)
    new_ref = vlog.weakref.ref(new)
    vlog._console_ref = new_ref

    vlog._drop_console_target(old_ref)
    assert vlog._console_ref is new_ref

    vlog._drop_console_target(new_ref)
    assert vlog._console_ref is None


def test_register_console_target_wires_the_destroyed_signal(vlog, qapp):
    """A panel that Qt destroys unregisters itself."""
    from PySide6.QtWidgets import QWidget

    class _Panel(QWidget):
        def append_stdout(self, text):
            return None

    vlog._console_ref = None
    vlog._handler = None
    vlog._relay = None
    panel = _Panel()

    vlog.register_console_target(panel)
    assert vlog._console_ref() is panel

    import shiboken6

    shiboken6.delete(panel)
    qapp.processEvents()

    assert vlog._console_ref is None
    if vlog._relay is not None:
        vlog._relay.deleteLater()
