"""``spacr.qt.app.launch``: what one start of spaCR builds, wires and survives.

Every case runs the real :func:`launch` against the stand-in QApplication of
the ``launched`` fixture (shared with ``tests/qt/test_cov_qt_app.py``), whose
``exec`` returns 0 at once.  Each launch's window is closed before the next
case, so every case sees exactly one new window.
"""
from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager

import pytest

from spacr.qt import app as app_mod
from spacr.qt.app import MainWindow
from tests.qt.test_cov_qt_app import (  # noqa: F401
    PREWARMED_MODULES,
    _close_owned_screens,
    launched,
)

pytestmark = pytest.mark.qt


class _LogBook:
    """Every message ``spacr.qt.app`` logs, whatever the logging setup does.

    ``launch`` installs spaCR's own logging, which can raise the module
    logger's level after the first launch; recording at the call keeps the
    assertions independent of the order cases run in.
    """

    def __init__(self, monkeypatch):
        self.lines: list = []
        for level in ("debug", "info", "warning", "error", "exception"):
            monkeypatch.setattr(app_mod.LOG, level, self._record)

    def _record(self, msg, *args, **_kwargs):
        self.lines.append(msg % args if args else str(msg))

    def text(self) -> str:
        return "\n".join(self.lines)


@contextmanager
def _one_launch(launched, qapp):
    """Yield a holder for the window one launch builds, and close it after."""
    launched["before"] = set(qapp.topLevelWidgets())
    holder: dict = {}
    try:
        yield holder
    finally:
        for window in [w for w in qapp.topLevelWidgets()
                       if w not in launched["before"]
                       and isinstance(w, MainWindow)]:
            try:
                _close_owned_screens(window)
                window.close()
            except Exception:                            # noqa: BLE001
                pass
            window.hide()
            window.deleteLater()
        qapp.processEvents()


def test_launch_builds_the_requested_window_and_wires_its_session(
        launched, qapp, qtbot, monkeypatch, tmp_path, subtests):
    """App name, theme, first screen, native-dialog order and the prewarm thread."""
    from spacr.qt.theme import window_stylesheet

    with subtests.test(msg="an app argument opens that app"), \
            _one_launch(launched, qapp):
        assert app_mod.launch(["measure"]) == 0
        shim = launched["shims"][-1]
        assert (shim.exec_calls, shim.app_name, shim.org_name) == (1, None, None)
        assert (shim.applicationName(), shim.organizationName()) == (
            "spaCR", "Olafsson Lab")
        assert "QWidget" in (window_stylesheet(shim) or "")
        win = launched["window"]()
        assert win.isVisible()
        assert win._status_app_label.text() == "Measure"
        assert "measure" in win._screens
        # Every field the opened screen shows is painted with the fade.
        from spacr.qt.widgets import field_fade
        qtbot.waitExposed(win)
        fields = [f for f in field_fade._visible_fields(qapp)
                  if f.window() is win and field_fade.fades(f)
                  and f not in field_fade._LENT]
        assert fields and field_fade.field_fade_enabled() is True
        # The launch installed the application-wide fade filter; it is put
        # on each field here so a synchronous repaint goes through it.
        assert isinstance(field_fade._filter, field_fade._FieldFadeFilter)
        for field in fields:
            field.installEventFilter(field_fade._filter)
        painted: list = []
        real_paint = field_fade.paint_field_fade
        with monkeypatch.context() as m:
            m.setattr(field_fade, "paint_field_fade", lambda w, p, theme=None: (
                painted.append(w), real_paint(w, p, theme)))
            for field in fields:
                field.repaint()
        # Qt paints only a field with something of it on screen.
        on_screen = {id(f) for f in fields if not f.visibleRegion().isEmpty()}
        assert on_screen and {id(f) for f in painted} == on_screen
        args, apps_made_so_far = launched["attributes"][-1]
        assert apps_made_so_far == len(launched["shims"]) - 1, (
            "AA_DontUseNativeDialogs set after the QApplication existed")
        assert args[-1] is True

    with subtests.test(msg="no arguments opens on Home"), \
            _one_launch(launched, qapp), monkeypatch.context() as m:
        m.setattr(sys, "argv", ["spacr-qt"])
        assert app_mod.launch() == 0
        if sys.platform == "darwin":
            from spacr.qt.menus import name_the_application
            expected = [name_the_application()[0]]
        else:
            expected = ["spacr-qt"]
        assert launched["shims"][-1].argv == expected
        win = launched["window"]()
        assert win._screens == {}
        assert win._stack.currentWidget() is win._startup
        assert win._status_app_label.text() == "Home"

    with subtests.test(msg="the heavy imports are prewarmed off thread"), \
            _one_launch(launched, qapp):
        before = len(launched["threads"])
        assert app_mod.launch([]) == 0
        new = launched["threads"][before:]
        assert [(t.name, t.daemon, t.started) for t in new] == [
            ("spacr-prewarm", True, True)]
        thread = new[0]
        requested: list = []
        real_import = importlib.import_module
        nesting = [0]

        def _record(name):
            if not nesting[0]:
                requested.append(name)
            nesting[0] += 1
            try:
                return real_import(name)
            finally:
                nesting[0] -= 1

        from spacr.qt import mask_engine
        catalogue_calls: list = []
        app_mod._HEAVY_WARM_GATE.set()
        with monkeypatch.context() as m:
            m.setattr(importlib, "import_module", _record)
            m.setattr(mask_engine, "_filter_catalogue",
                      lambda: catalogue_calls.append(True))
            thread.target()
        assert requested == list(PREWARMED_MODULES)
        assert catalogue_calls == [True]
        assert [n for n in PREWARMED_MODULES if n not in sys.modules] == []
        with monkeypatch.context() as m:
            m.setattr(importlib, "import_module", lambda name: (
                _ for _ in ()).throw(ImportError("no module today")))
            assert thread.target() is None

    with subtests.test(msg="Home left idle: screens, Preferences, ambient"), \
            _one_launch(launched, qapp), monkeypatch.context() as m:
        # The machine's free memory and core count must not decide the
        # outcome: prewarm is forced on and the memory floors set to zero,
        # so every line of the memory check still runs and answers "room".
        from spacr.qt import preferences
        from spacr.qt.widgets import ambient
        m.setenv("SPACR_PREWARM", "1")
        m.delenv("SPACR_PREWARM_ORDER", raising=False)
        m.setattr(preferences, "get_headroom_mb", lambda: 0)
        m.setattr(app_mod, "_PREWARM_SPARE_MB", -1e12)
        assert app_mod.launch([]) == 0
        win = launched["window"]()
        prewarm = win._screen_prewarm
        order = list(preferences._SCREEN_PREWARM_ORDER)
        assert prewarm._order == order
        warmth = win._spacr_preferences_warmth

        def _settled():
            dialog = warmth["dialog"]
            return (prewarm.finished and dialog is not None
                    and getattr(dialog, "_spacr_shown_off_screen", False)
                    and ambient._AMBIENT_STARTUP_READY.is_set())

        qtbot.waitUntil(_settled, timeout=120000)
        assert (prewarm.built, prewarm.skipped, prewarm._order) == (
            order, [], [])
        assert [k for k in order if k not in win._screens] == []
        assert prewarm._steps is None and prewarm._key is None
        assert win._stack.currentWidget() is win._startup
        assert type(warmth["dialog"]) is preferences._preferences_window_class()
        assert warmth["dialog"]._spacr_shown_off_screen is True
        assert prewarm.busy_reason() == ""
        win._opening_a_screen = True
        assert prewarm.busy_reason() == "opening"
        win._opening_a_screen = False

        # Stopped half way through a build: the half built screen is dropped.
        pages, scales = win._stack.count(), dict(win._screen_scales)
        prewarm.finished = False
        prewarm._key = "convert"
        prewarm._steps = win._prewarm_steps("convert")
        next(prewarm._steps)
        assert "convert" in win._screen_scales
        # A slice that finds the data-library turn taken does nothing.
        app_mod._DATA_LIBRARY_TURN.acquire()
        try:
            held = prewarm._steps
            win._run_one_prewarm_step(prewarm)
            assert (prewarm._steps, win._opening_a_screen) == (held, False)
            assert "convert" not in win._screens
        finally:
            app_mod._DATA_LIBRARY_TURN.release()
        prewarm.stop()
        assert (prewarm.finished, prewarm._steps) == (True, None)
        assert "convert" not in win._screens
        assert (win._stack.count(), win._screen_scales) == (pages, scales)

    with subtests.test(msg="a benchmark run keeps its controller"), \
            _one_launch(launched, qapp), monkeypatch.context() as m:
        from spacr.qt import startup_benchmark
        controller = object()
        started: list = []
        m.setenv("SPACR_BENCHMARK_JSON", str(tmp_path / "bench.json"))
        m.setattr(startup_benchmark, "maybe_start",
                  lambda app, win: started.append((app, win)) or controller)
        assert app_mod.launch([]) == 0
        window = launched["window"]()
        assert [pair[1] for pair in started] == [window]
        assert window._startup_benchmark_controller is controller

    with subtests.test(msg="matplotlib not yet imported: pinned by env"), \
            _one_launch(launched, qapp), monkeypatch.context() as m:
        m.setenv("MPLBACKEND", "TkAgg")
        m.delitem(sys.modules, "matplotlib", raising=False)
        assert app_mod.launch([]) == 0
        assert os.environ["MPLBACKEND"] == "Agg"

    with subtests.test(msg="matplotlib already imported: use(Agg, force)"), \
            _one_launch(launched, qapp), monkeypatch.context() as m:
        import matplotlib
        m.setitem(sys.modules, "matplotlib", matplotlib)
        used: list = []
        m.setattr(matplotlib, "use",
                  lambda backend, force=False: used.append((backend, force)))
        assert app_mod.launch([]) == 0
        assert used == [("Agg", True)]


def test_launch_asks_setup_and_recovery_questions_in_order(
        launched, qapp, monkeypatch, capsys, tmp_path, subtests):
    """Setup is skipped on --no-setup, answers re-apply preferences once more,
    unclean exits drop the backdrop, a clean exit is recorded, timing is announced."""
    from spacr.qt import crash_recovery, preferences
    from spacr.qt.widgets import setup_slides

    with subtests.test(msg="--no-setup never opens the setup screen"):
        asked: list = []
        with monkeypatch.context() as m:
            m.setattr(setup_slides, "open_setup_if_needed",
                      lambda parent: asked.append(parent))
            with _one_launch(launched, qapp):
                assert app_mod.launch(["--no-setup"]) == 0
            assert asked == []
            with _one_launch(launched, qapp):
                assert app_mod.launch([]) == 0
            assert asked == [None]

    with subtests.test(msg="setup answers re-apply preferences before the window"):
        real_apply = preferences.apply_preferences_to_app
        applied: list = []
        with monkeypatch.context() as m:
            m.setattr(preferences, "apply_preferences_to_app",
                      lambda app: applied.append(app) or real_apply(app))
            m.setattr(setup_slides, "open_setup_if_needed", lambda parent: None)
            with _one_launch(launched, qapp):
                assert app_mod.launch([]) == 0
            assert len(applied) == 1
            m.setattr(setup_slides, "open_setup_if_needed",
                      lambda parent: {"language": "en"})
            with _one_launch(launched, qapp):
                assert app_mod.launch([]) == 0
            assert len(applied) == 3
            assert applied[-1] is applied[-2]

    with subtests.test(msg="a run of unclean exits drops the backdrop"), \
            _one_launch(launched, qapp), monkeypatch.context() as m:
        dropped: list = []
        m.setattr(crash_recovery, "should_start_without_the_backdrop",
                  lambda unclean=None: True)
        m.setattr(crash_recovery, "take_the_backdrop_out_of_this_launch",
                  lambda: dropped.append("dropped"))
        assert app_mod.launch([]) == 0
        assert dropped == ["dropped"]

    with subtests.test(msg="a clean shutdown is recorded"), \
            _one_launch(launched, qapp), monkeypatch.context() as m:
        noted: list = []
        m.setattr(crash_recovery, "note_a_clean_shutdown",
                  lambda: noted.append("noted"))
        assert app_mod.launch([]) == 0
        assert noted == ["noted"]

    with subtests.test(msg="a written timing report says where it went"):
        target = tmp_path / "spacr-timing.log"

        def _write():
            target.write_text("spaCR timing report\n", encoding="utf-8")
            return str(target)

        capsys.readouterr()
        with monkeypatch.context() as m:
            m.setattr(app_mod._timing, "write_report", _write)
            with _one_launch(launched, qapp):
                assert app_mod.launch([]) == 0
            assert f"spaCR timing report written to {target}" in \
                capsys.readouterr().out
            assert target.read_text(encoding="utf-8") == "spaCR timing report\n"
            m.setattr(app_mod._timing, "write_report", lambda: "")
            with _one_launch(launched, qapp):
                assert app_mod.launch([]) == 0
            assert "timing report written" not in capsys.readouterr().out


def test_launch_quit_drains_work_consoles_and_ai_providers(
        launched, qapp, monkeypatch, subtests):
    """The one aboutToQuit handler retires live work, shuts every console down
    and cancels every AI provider, each step surviving the one before failing."""
    from spacr.qt import ai as ai_mod
    from spacr.qt import bridge
    from spacr.qt.widgets.console_panel import ConsolePanel

    retired: list = []
    monkeypatch.setattr(bridge, "retire_live_work",
                        lambda: retired.append("drained"))
    with _one_launch(launched, qapp):
        assert app_mod.launch(["mask"]) == 0
        win = launched["window"]()
        handlers = launched["shims"][-1].aboutToQuit.callbacks
        assert len(handlers) == 1
        drain = handlers[0]
        panels = win.findChildren(ConsolePanel)
        assert panels

        with subtests.test(msg="work, consoles and providers drained"):
            consoles: list = []
            cancelled: list = []

            class _Provider:
                def cancel_stream(self):
                    cancelled.append(self)

            with monkeypatch.context() as m:
                m.setattr(ConsolePanel, "shutdown",
                          lambda self: consoles.append(self))
                m.setattr(ai_mod, "list_providers", lambda: [_Provider()])
                assert drain() is None
            assert retired == ["drained"]
            assert set(consoles) == set(panels) and len(consoles) == len(panels)
            assert len(cancelled) == 1

        with subtests.test(msg="a runner registry that raises still lets "
                               "the consoles go"):
            book = _LogBook(monkeypatch)
            consoles = []
            with monkeypatch.context() as m:
                m.setattr(bridge, "retire_live_work", lambda: (
                    _ for _ in ()).throw(RuntimeError("the runner list is gone")))
                m.setattr(ConsolePanel, "shutdown",
                          lambda self: consoles.append(self))
                drain()
            assert "Could not drain the job runners" in book.text()
            assert consoles == panels

        with subtests.test(msg="raising panels and providers stop nothing"):
            book = _LogBook(monkeypatch)
            with monkeypatch.context() as m:
                m.setattr(ConsolePanel, "shutdown", lambda self: (
                    _ for _ in ()).throw(RuntimeError("stuck")))
                m.setattr(ai_mod, "list_providers", lambda: (
                    _ for _ in ()).throw(RuntimeError("providers unavailable")))
                assert drain() is None
            assert "Could not shut down a console panel" in book.text()
            assert "Could not cancel every AI provider" in book.text()


def _raise(exc):
    def _f(*_a, **_k):
        raise exc
    return _f


def test_launch_survives_every_optional_step_failing(
        launched, qapp, monkeypatch, subtests):
    """A broken optional collaborator costs only itself: launch still returns 0,
    the window is still shown, and the failure is logged (or merely survived)."""
    import PySide6.QtGui as qtgui
    import matplotlib

    from spacr.qt import crash_recovery, laptop_mode, thread_guard
    from spacr.qt.widgets import setup_slides

    class _OldQImageReader:
        @staticmethod
        def setAllocationLimit(limit):
            raise AttributeError("not in this Qt build")

    probe_calls: list = []

    def _describe_raises():
        probe_calls.append("describe")
        raise RuntimeError("no battery on this machine")

    def _apply_raises(_on=None):
        probe_calls.append("apply")
        raise OSError("cannot write the power governor")

    cases = [
        ("thread guard will not install",
         [(thread_guard, "install", _raise(RuntimeError("cannot patch QTimer")))],
         None, []),
        ("matplotlib refuses Agg",
         [(matplotlib, "use", _raise(ImportError("no Agg backend")))],
         None, []),
        ("setup screen raises",
         [(setup_slides, "open_setup_if_needed", _raise(RuntimeError("no slides")))],
         "could not open the setup screen", []),
        ("clean shutdown cannot be recorded",
         [(crash_recovery, "note_a_clean_shutdown",
           _raise(OSError("read-only profile")))],
         "could not record a clean shutdown", []),
        ("laptop probe raises",
         [(laptop_mode, "describe", _describe_raises)],
         "could not decide laptop mode", ["describe"]),
        ("laptop apply raises",
         [(laptop_mode, "describe", lambda: "a description"),
          (laptop_mode, "apply", _apply_raises)],
         "could not decide laptop mode", ["apply"]),
        ("QImageReader without an allocation limit",
         [(qtgui, "QImageReader", _OldQImageReader)],
         "This Qt build does not expose QImageReader allocation", []),
    ]
    for name, patches, logged, probed in cases:
        with subtests.test(msg=name), _one_launch(launched, qapp), \
                monkeypatch.context() as m:
            m.setitem(sys.modules, "matplotlib", matplotlib)
            for target, attr, value in patches:
                m.setattr(target, attr, value)
            book = _LogBook(m)
            probe_calls.clear()
            assert app_mod.launch([]) == 0
            assert launched["window"]().isVisible()
            assert probe_calls == probed
            if logged:
                assert logged in book.text()


def test_launch_says_what_laptop_mode_turned_down(launched, qapp, monkeypatch,
                                                  subtests):
    """A non-empty change list is logged by name; an empty one prints nothing."""
    from spacr.qt import laptop_mode

    cases = [
        ("changed something", ["backdrop", "shadows"],
         ["laptop mode changed: backdrop, shadows"]),
        ("changed threads and backdrop", ["threads", "backdrop"],
         ["laptop mode changed: threads, backdrop"]),
        ("changed nothing", [], []),
    ]
    for name, changed, expected in cases:
        with subtests.test(msg=name), _one_launch(launched, qapp), \
                monkeypatch.context() as m:
            applied: list = []
            m.setattr(laptop_mode, "describe", lambda: "on battery")
            m.setattr(laptop_mode, "apply", lambda _on=None, _c=changed:
                      applied.append(_on) or {"changed": list(_c)})
            book = _LogBook(m)
            assert app_mod.launch([]) == 0
            assert len(applied) == 1
            assert [line for line in book.lines
                    if line.startswith("laptop mode changed")] == expected
