"""Table-driven tests of ``spacr.qt.verbose_logger``: sinks, relay and trace helpers.

The verbose logger owns three process-wide pieces: the rotating log file
attached once at the ``spacr`` package logger, the console forwarder beside
it whose level gate the Preferences dialog edits, and the relay that carries
each formatted line onto the GUI thread into whichever console panel is
registered.  Every scenario here pins one of those: where the file lives,
that each sink is attached exactly once, what the console gate reports after
foreign filters join it, which deliveries are dropped and which reference
survives them, and the exact trace lines ``log_call``/``log_button_press``
write when verbose is on (and that they cost nothing when it is off).

Every test runs inside the ``vlog`` fixture, which snapshots the module's
singletons and the attached loggers' handlers and levels and puts them back.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

D, I, W, E = logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR


class Widget:
    """A module-level class, so its method's qualname is ``Widget.press``."""

    def press(self, which="Run"):
        return f"pressed {which}"


def free(which="Run"):
    """A plain function: its qualname has no dot."""
    return f"called {which}"


def explode():
    """A function that always raises, for the RAISED trace line."""
    raise ValueError("no such column")


class Weakrefable:
    """A do-nothing object a ``weakref.ref`` can point at."""


class Hostile:
    """A value whose ``__repr__`` raises."""

    def __repr__(self):
        raise RuntimeError("no repr for you")


class Unicode:
    """A short custom repr carrying non-ASCII text."""

    def __repr__(self):
        return "Sample(Δx=4, channels=(1, 3))"


class Recorder:
    """A console stand-in that records every appended line."""

    def __init__(self, fail=False):
        self.seen = []
        self.fail = fail

    def append_stdout(self, text):
        self.seen.append(text)
        if self.fail:
            raise RuntimeError("the document is gone")


@pytest.fixture()
def vlog(tmp_path, monkeypatch):
    """The module with its singletons snapshotted and its log dir redirected."""
    from spacr.qt import verbose_logger as module

    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    saved = {name: getattr(module, name) for name in
             ("_console_ref", "_handler", "_relay", "_file_handler",
              "_verbose")}
    names = tuple(module._ATTACHED_LOGGERS) + ("cellpose",)
    saved_handlers = {n: list(logging.getLogger(n).handlers) for n in names}
    saved_levels = {n: logging.getLogger(n).level for n in names}
    try:
        yield module
    finally:
        # apply_verbose_logging must never leave the process-wide trace on.
        from spacr.logging_util import disable_function_trace
        disable_function_trace()
        for name, value in saved.items():
            setattr(module, name, value)
        for name in names:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if handler not in saved_handlers[name] and isinstance(
                        handler, RotatingFileHandler):
                    handler.close()
            logger.handlers[:] = saved_handlers[name]
            logger.setLevel(saved_levels[name])


@contextlib.contextmanager
def _fresh_file_sink(vlog, monkeypatch, directory):
    """A scenario with its own log directory and no file handler yet."""
    monkeypatch.setenv("SPACR_LOG_DIR", str(directory))
    sink = logging.getLogger("spacr")
    before = list(sink.handlers)
    vlog._file_handler = None
    try:
        yield sink
    finally:
        for handler in sink.handlers:
            if handler not in before:
                handler.close()
        sink.handlers[:] = before
        vlog._file_handler = None


def _record(name, level, message="m"):
    """One record to push through a handler's filter chain."""
    return logging.LogRecord(name, level, __file__, 1, message, (), None)


def _trace_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "spacr.trace"]


def test_log_dir_honours_the_override_and_defaults_under_home(
        vlog, tmp_path, monkeypatch, subtests):
    """SPACR_LOG_DIR wins; without it the log lives in ~/.spacr/logs; both are created on demand."""
    from datetime import datetime

    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    today = f"spacr-{datetime.now().strftime('%Y%m%d')}.log"
    for case_id, override, expected in (
            ("override", str(tmp_path / "logs"), tmp_path / "logs"),
            ("home default", None, fake_home / ".spacr" / "logs")):
        with subtests.test(msg=case_id):
            if override is None:
                monkeypatch.delenv("SPACR_LOG_DIR", raising=False)
            else:
                monkeypatch.setenv("SPACR_LOG_DIR", override)
            assert vlog.log_dir() == expected
            assert expected.is_dir()
            assert vlog.current_log_file() == expected / today


def test_the_file_sink_is_attached_once_at_the_package_root(
        vlog, tmp_path, monkeypatch, subtests):
    """Idempotent attachment, one line per event, a clean re-attach after a detach, and None when the file will not open."""
    with subtests.test(msg="built once, only on spacr, at INFO"):
        with _fresh_file_sink(vlog, monkeypatch, tmp_path / "once") as sink:
            first = vlog._ensure_file_handler()
            assert vlog._ensure_file_handler() is first
            assert isinstance(first, RotatingFileHandler)
            assert first.baseFilename == str(vlog.current_log_file())
            assert first.level == I
            assert sink.handlers.count(first) == 1
            for name in vlog._ATTACHED_LOGGERS[1:]:
                assert first not in logging.getLogger(name).handlers

    with subtests.test(msg="a descendant record lands exactly once"):
        with _fresh_file_sink(vlog, monkeypatch, tmp_path / "pre") as sink:
            log_path = tmp_path / "pre" / "already-attached.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            already = RotatingFileHandler(str(log_path), encoding="utf-8")
            sink.addHandler(already)
            sink.setLevel(logging.NOTSET)
            with monkeypatch.context() as patch:
                patch.setattr(vlog, "RotatingFileHandler",
                              lambda *a, **k: already)
                assert vlog._ensure_file_handler() is already
            assert sink.handlers.count(already) == 1
            for name in vlog._ATTACHED_LOGGERS[1:]:
                assert already not in logging.getLogger(name).handlers
            # Levels are still lowered to INFO past the attach guard.
            assert sink.level == I
            logging.getLogger("spacr.qt.hf_download").info("one event")
            already.flush()
            assert log_path.read_text(encoding="utf-8").count("one event") == 1

    with subtests.test(msg="a detached handler is re-attached, not rebuilt"):
        with _fresh_file_sink(vlog, monkeypatch, tmp_path / "re") as sink:
            handler = vlog._ensure_file_handler()
            sink.handlers[:] = [handler]
            propagate, sink.propagate = sink.propagate, False
            stream, path = handler.stream, vlog.current_log_file()
            try:
                sink.handle(logging.makeLogRecord(
                    {"name": "spacr", "levelno": I, "levelname": "INFO",
                     "msg": "before detach"}))
                handler.flush()
                before = path.read_text(encoding="utf-8")
                sink.removeHandler(handler)
                assert vlog._ensure_file_handler() is handler
                assert vlog._ensure_file_handler() is handler
                assert sink.handlers.count(handler) == 1
                assert handler.stream is stream and not stream.closed
                assert path.read_text(encoding="utf-8") == before
                sink.handle(logging.makeLogRecord(
                    {"name": "spacr", "levelno": I, "levelname": "INFO",
                     "msg": "after reattach"}))
                handler.flush()
                lines = path.read_text(encoding="utf-8").splitlines()
                assert len(lines) == 2
                assert lines[0].endswith("before detach")
                assert lines[1].endswith("after reattach")
            finally:
                sink.propagate = propagate

    with subtests.test(msg="an unopenable file is skipped, not fatal"):
        with _fresh_file_sink(vlog, monkeypatch, tmp_path / "ro"):
            def _refuse(*args, **kwargs):
                raise PermissionError("the log directory is read-only")
            with monkeypatch.context() as patch:
                patch.setattr(vlog, "RotatingFileHandler", _refuse)
                assert vlog._ensure_file_handler() is None
                assert vlog._file_handler is None


def test_the_console_gate_round_trips_through_foreign_filters(vlog, subtests):
    """console_levels() reports exactly what apply_console_levels set, whatever other filters share the forwarder."""
    from spacr.logging_util import LevelSetFilter

    def fresh_handler():
        vlog._handler = None
        return vlog._ensure_handler()

    with subtests.test(msg="no handler yet shows nothing"):
        vlog._handler = None
        assert vlog.console_levels() == frozenset()

    with subtests.test(msg="the forwarder lives only on spacr, once"):
        handler = fresh_handler()
        assert vlog._ensure_handler() is handler
        assert logging.getLogger("spacr").handlers.count(handler) == 1
        for name in vlog._ATTACHED_LOGGERS[1:]:
            assert handler not in logging.getLogger(name).handlers

    with subtests.test(msg="a handler with no gate shows nothing"):
        handler = fresh_handler()
        saved = list(handler.filters)
        handler.filters[:] = []
        try:
            assert vlog.console_levels() == frozenset()
        finally:
            handler.filters[:] = saved

    with subtests.test(msg="round trip, and the loggers follow the lowest"):
        handler = fresh_handler()
        for name in vlog._ATTACHED_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)
        for levels, lowest in (([W, E], W), ([D], D), ([E], D)):
            vlog.apply_console_levels(levels)
            assert vlog.console_levels() == frozenset(levels)
            assert handler.level == D
            assert {logging.getLogger(n).level
                    for n in vlog._ATTACHED_LOGGERS} == {lowest}

    with subtests.test(msg="a foreign filter survives being re-gated"):
        class _NotFromTheQtLayer(logging.Filter):
            def filter(self, record):
                return record.name != "spacr.qt"

        handler = fresh_handler()
        saved = list(handler.filters)
        handler.filters[:] = []
        foreign = _NotFromTheQtLayer()
        handler.addFilter(foreign)
        try:
            vlog.apply_console_levels([W, E])
            vlog.apply_console_levels([W])
            assert handler.filters[0] is foreign
            assert sum(isinstance(f, LevelSetFilter)
                       for f in handler.filters) == 1
            assert vlog.console_levels() == frozenset({W})
            verdicts = [bool(handler.filter(_record(n, lv))) for n, lv in
                        (("spacr.trace", W), ("spacr.qt", W),
                         ("spacr.trace", E))]
            assert verdicts == [True, False, False]
        finally:
            handler.filters[:] = saved

    with subtests.test(msg="the gate is read from behind a stranger"):
        handler = fresh_handler()
        saved = list(handler.filters)
        handler.filters[:] = [logging.Filter("spacr.trace")]
        try:
            assert vlog.console_levels() == frozenset()
            handler.addFilter(LevelSetFilter({D, I}))
            assert vlog.console_levels() == frozenset({D, I})
            assert type(handler.filters[0]) is logging.Filter
        finally:
            handler.filters[:] = saved


def test_a_relay_lives_on_the_gui_thread_wherever_it_was_built(vlog, qapp,
                                                              subtests):
    """Built on a worker or on the GUI thread, the relay's affinity is the application's thread."""
    from PySide6.QtCore import QThread

    for case_id, on_worker in (("worker", True), ("gui thread", False)):
        with subtests.test(msg=case_id):
            built = {}

            def _build():
                built["born_on"] = QThread.currentThread()
                built["relay"] = vlog._ConsoleRelay()

            if on_worker:
                worker = threading.Thread(target=_build, name="a-worker")
                worker.start()
                worker.join()
            else:
                _build()
            relay = built["relay"]
            assert (built["born_on"] == qapp.thread()) is (not on_worker)
            assert relay.thread() == qapp.thread()
            relay.deleteLater()


def test_relay_delivery_drops_what_it_must_and_forgets_only_dead_widgets(
        vlog, qapp, subtests):
    """Each delivery outcome: exact lines appended, and whether the registration survives."""
    import shiboken6
    from PySide6.QtWidgets import QWidget

    class _QtPanel(QWidget):
        def __init__(self):
            super().__init__()
            self.seen = []

        def append_stdout(self, text):
            self.seen.append(text)

    def registered(target):
        return lambda: target

    def dead_widget():
        panel = _QtPanel()
        ref = vlog.weakref.ref(panel)
        shiboken6.delete(panel)
        assert shiboken6.isValid(panel) is False
        return panel, ref

    def plain(fail=False):
        target = Recorder(fail=fail)
        return target, registered(target)

    # id, factory -> (target, ref), lines to deliver, inside a console write,
    # -> lines the target saw, registration after ("kept" / None)
    cases = [
        ("a registered panel gets every line", plain,
         ["a\n", "b\n"], False, ["a\n", "b\n"], "kept"),
        ("a record made during a console write is dropped", plain,
         ["inside\n"], True, [], "kept"),
        ("a collected console drops the line but stays registered",
         lambda: (None, lambda: None), ["late\n"], False, None, "kept"),
        ("no console registered drops the line", lambda: (None, None),
         ["early\n"], False, None, None),
        ("a deleted QWidget is forgotten, not called", dead_widget,
         ["void\n"], False, [], None),
        ("a target without append_stdout is skipped, not forgotten",
         lambda: (Weakrefable(), registered(Weakrefable())),
         ["x\n"], False, None, "kept"),
        ("a console that raises keeps receiving", lambda: plain(fail=True),
         ["boom\n", "again\n"], False, ["boom\n", "again\n"], "kept"),
    ]
    # One relay for every case: a drop must leave nothing behind for the next.
    relay = vlog._ConsoleRelay()
    try:
        for case_id, factory, lines, inside, seen, after in cases:
            with subtests.test(msg=case_id):
                target, ref = factory()
                vlog._console_ref = ref
                for line in lines:
                    if inside:
                        with vlog.console_write():
                            relay._deliver(line)
                    else:
                        relay._deliver(line)
                if seen is not None:
                    assert target.seen == seen
                assert vlog._console_ref is (ref if after == "kept" else None)
    finally:
        relay.deleteLater()


def test_console_write_latch_and_forwarder_emit(vlog, qapp, subtests):
    """The re-entrant latch restores its depth; emit formats nothing during a write and delivers 'hello' otherwise."""
    with subtests.test(msg="the latch nests"):
        states = [vlog.console_write_in_progress()]
        with vlog.console_write():
            with vlog.console_write():
                states.append(vlog.console_write_in_progress())
            states.append(vlog.console_write_in_progress())
        states.append(vlog.console_write_in_progress())
        assert states == [False, True, True, False]

    formatted = []

    class _Counting(vlog._ConsoleForwarder):
        def format(self, record):
            formatted.append(record)
            return super().format(record)

    record = _record("spacr", I, "hello")
    for case_id, inside, delivered_expected, formatted_expected in (
            ("during a console write", True, [], 0),
            ("to the registered console", False, ["hello\n"], 1)):
        with subtests.test(msg=case_id):
            formatted.clear()
            panel = Recorder()
            vlog._relay = None
            vlog._console_ref = lambda: panel
            handler = _Counting()
            handler.setFormatter(logging.Formatter("%(message)s"))
            if inside:
                with vlog.console_write():
                    handler.emit(record)
            else:
                handler.emit(record)
            assert panel.seen == delivered_expected
            assert len(formatted) == formatted_expected
            if vlog._relay is not None:
                vlog._relay.deleteLater()


def test_verbose_follows_the_preference_even_without_a_log_file(
        vlog, monkeypatch):
    """is_verbose() tracks apply_verbose_logging, not the forwarder level, and flips even when the file cannot open."""
    vlog._handler = vlog._file_handler = vlog._console_ref = None
    vlog._verbose = False
    assert vlog.is_verbose() is False

    for on, level, cellpose in ((False, I, W), (True, D, I), (False, I, W)):
        vlog.apply_verbose_logging(on)
        assert vlog.is_verbose() is on
        assert isinstance(vlog._file_handler, RotatingFileHandler)
        assert (vlog._handler.level, vlog._file_handler.level) == (level, level)
        assert {logging.getLogger(n).level
                for n in vlog._ATTACHED_LOGGERS} == {level}
        assert logging.getLogger("cellpose").level == cellpose

    # The console showing DEBUG is not verbose.
    vlog.apply_console_levels({D, I})
    assert vlog._handler.level == D
    assert vlog.is_verbose() is False

    opened = vlog._file_handler
    logging.getLogger("spacr").removeHandler(opened)
    opened.close()

    def _refuse(*args, **kwargs):
        raise OSError("read-only file system")

    vlog._file_handler = None
    monkeypatch.setattr(vlog, "RotatingFileHandler", _refuse)
    vlog.apply_verbose_logging(True)
    try:
        assert vlog._file_handler is None
        assert vlog._handler.level == D
        assert vlog.is_verbose() is True
        assert {logging.getLogger(n).level
                for n in vlog._ATTACHED_LOGGERS} == {D}
    finally:
        vlog.apply_verbose_logging(False)
    assert vlog.is_verbose() is False


def test_log_call_and_button_press_write_exact_trace_lines(vlog, caplog,
                                                           subtests):
    """With verbose on the exact args/return/raise and button lines appear; with it off nothing is logged or repr'd."""
    built = []

    class Loud:
        def __repr__(self):
            built.append(1)
            return "Loud()"

    # id, verbose, action -> expected return, expected spacr.trace lines
    cases = [
        ("bound method hides self", True,
         lambda: vlog.log_call(Widget.press)(Widget(), "Run"), "pressed Run",
         ["[Widget.press] args=('Run',) kwargs=",
          "[Widget.press] -> 'pressed Run'"]),
        ("free function keeps its first arg", True,
         lambda: vlog.log_call(free)("Run", ), "called Run",
         ["[free] args=('Run',) kwargs=", "[free] -> 'called Run'"]),
        ("kwargs are shown", True,
         lambda: vlog.log_call(free)(which="Go"), "called Go",
         ["[free] args=() kwargs={'which': 'Go'}", "[free] -> 'called Go'"]),
        ("verbose off forwards untouched", False,
         lambda: vlog.log_call(free)("Stop"), "called Stop", []),
        ("verbose off builds no repr", False,
         lambda: vlog.log_call(lambda value: value)(Loud()) and None, None, []),
        ("bare button press", True,
         lambda: vlog.log_button_press("Run"), None, ["[button:Run] pressed"]),
        ("button press with context", True,
         lambda: vlog.log_button_press("Stop", {"src": "/data"}), None,
         ["[button:Stop] {'src': '/data'}"]),
        ("button press with verbose off", False,
         lambda: vlog.log_button_press("Run", {"src": "/data"}), None, []),
    ]
    vlog._handler = None
    for case_id, verbose, action, expected, lines in cases:
        with subtests.test(msg=case_id):
            vlog.apply_verbose_logging(verbose)
            # The console showing DEBUG must not make the wrapper verbose.
            vlog.apply_console_levels({D, I})
            caplog.clear()
            with caplog.at_level(D, logger="spacr.trace"):
                assert action() == expected
            assert _trace_lines(caplog) == lines
    assert built == []

    with subtests.test(msg="a raise is recorded and re-raised"):
        vlog.apply_verbose_logging(True)
        caplog.clear()
        with caplog.at_level(D, logger="spacr.trace"):
            with pytest.raises(ValueError, match="no such column"):
                vlog.log_call(explode)()
        assert _trace_lines(caplog) == [
            "[explode] args=() kwargs=",
            "[explode] RAISED ValueError: no such column"]


def test_brief_and_looks_bound_on_a_table_of_values(vlog, subtests):
    """_brief caps at max_chars-3 plus an ellipsis and describes a failing repr; _looks_bound needs a matching class name."""
    giant = {f"key{i}": "x" * 20 for i in range(100)}
    shown = ", ".join(f"'{k}': '{'x' * 20}'" for k in sorted(giant)[:12])
    giant_text = ("{" + shown + ", ...}")[:237] + "…"
    for case_id, value, kwargs, expected in (
            ("giant dict truncated", giant, {}, giant_text),
            ("short string untouched", "short", {}, "'short'"),
            ("custom cap", "y" * 500, {"max_chars": 20},
             "'" + "y" * 16 + "…"),
            ("long list elided by reprlib", list(range(100)), {},
             "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]"),
            ("failing repr described", Hostile(), {},
             "<Hostile — repr failed>"),
            ("short unicode repr kept whole", Unicode(), {},
             "Sample(Δx=4, channels=(1, 3))")):
        with subtests.test(msg=f"brief: {case_id}"):
            text = vlog._brief(value, **kwargs)
            assert text == expected
            assert len(text) <= kwargs.get("max_chars", 240) - 2
    assert vlog._BRIEF_REPR.repr(Unicode()) == repr(Unicode())

    for case_id, fn, args, expected in (
            ("free function, args", free, ("a",), False),
            ("free function, no args", free, (), False),
            ("method with its instance", Widget.press, (Widget(), "a"), True),
            ("method with a stranger first", Widget.press, ("x", "a"), False),
            ("method with no args", Widget.press, (), False)):
        with subtests.test(msg=f"looks_bound: {case_id}"):
            assert vlog._looks_bound(fn, args) is expected


def test_console_targets_unregister_only_themselves(vlog, qapp, subtests):
    """A stale drop leaves a newer target alone; a registered QWidget unregisters itself when Qt destroys it."""
    with subtests.test(msg="drop stale vs current"):
        old, new = Weakrefable(), Weakrefable()
        old_ref, new_ref = vlog.weakref.ref(old), vlog.weakref.ref(new)
        vlog._console_ref = new_ref
        vlog._drop_console_target(old_ref)
        assert vlog._console_ref is new_ref
        vlog._drop_console_target(new_ref)
        assert vlog._console_ref is None
        vlog._console_ref = new_ref
        vlog._drop_console_target()
        assert vlog._console_ref is None

    with subtests.test(msg="destroyed signal unregisters"):
        import shiboken6
        from PySide6.QtWidgets import QWidget

        class _Panel(QWidget):
            def append_stdout(self, text):
                return None

        vlog._console_ref = vlog._handler = vlog._relay = None
        panel = _Panel()
        vlog.register_console_target(panel)
        assert vlog._console_ref() is panel
        assert logging.getLogger("spacr").handlers.count(vlog._handler) == 1
        shiboken6.delete(panel)
        qapp.processEvents()
        assert vlog._console_ref is None
        if vlog._relay is not None:
            vlog._relay.deleteLater()
