"""The first Run of a session imports its pipeline off the GUI thread.

The tutorial capture of lesson 07 (clip run_20_batch_settings, Mask
Generation) showed the window freeze for about 3 s when Run was pressed, on
the first take only. Measured offscreen on 2026-10-13 with a 2 ms heartbeat
and cProfile: pressing Run asked :func:`spacr.qt.bridge.resolve_pipeline_entry`
for the entry point, and for Mask that is ``from spacr.core import ...`` --
torch, cellpose, pandas, Matplotlib, about 2,900 modules -- 3.2-3.6 s on the
GUI thread the first time and nothing after it. The busy bar first painted
3.27 s after the click.

Now the resolver can say that an import is still to come and run it on a
worker; the screen shows the busy bar, waits, and then carries on exactly as
before. The busy bar paints 3-9 ms after the click.

These tests drive the real :class:`~spacr.qt.screens.app_screen.AppScreen`
with a stubbed pipeline whose "module" is a file written here that takes a
measured time to import. Timing is compared with the same import done the old
way, on the GUI thread, in the same test, so a loaded runner slows both sides
alike.
"""
from __future__ import annotations

import ast
import gc
import importlib
import inspect
import sys
import textwrap
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QTimer                 # noqa: E402

from spacr.qt import bridge                                         # noqa: E402

IMPORT_S = 0.3


@pytest.fixture(autouse=True)
def fresh_imports(monkeypatch):
    """A private module table and import record; no real freeze of the GC."""
    frozen = []
    monkeypatch.setattr(bridge, "_ENTRY_MODULES", {})
    monkeypatch.setattr(bridge, "_ENTRY_IMPORTS", {})
    monkeypatch.setattr(gc, "freeze", lambda: frozen.append(True))
    return frozen


@pytest.fixture
def make_module(tmp_path, monkeypatch):
    """Write a module that takes ``seconds`` to import, or raises."""
    monkeypatch.syspath_prepend(str(tmp_path))
    made = []

    def make(seconds=IMPORT_S, *, fail=False):
        name = f"spacr_entry_probe_{uuid.uuid4().hex[:12]}"
        body = f"import time\ntime.sleep({seconds!r})\n"
        if fail:
            body += "raise RuntimeError('this pipeline cannot be imported')\n"
        (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
        made.append(name)
        return name

    yield make
    for name in made:
        sys.modules.pop(name, None)


def _resolver(name, received, *, hooks=True):
    """A resolver that imports ``name`` the way the real one imports a module.

    :param hooks: give it the real resolver's two off-thread hooks, as
        :func:`spacr.qt.bridge.resolve_pipeline_entry` has; without them the
        screen resolves on the GUI thread, which is the old behaviour.
    """
    def stub(settings):
        received.append(dict(settings))

    def resolve(key):
        try:
            importlib.import_module(name)
        except Exception:
            return None
        return bridge._tag(key, stub)

    if hooks:
        resolve.entry_import_pending = bridge._entry_import_pending
        resolve.import_off_the_gui_thread = (
            bridge._import_the_entry_off_the_gui_thread)
    return resolve


def _use(monkeypatch, resolve, name):
    monkeypatch.setitem(bridge._ENTRY_MODULES, "mask", name)
    monkeypatch.setattr(
        "spacr.qt.screens.app_screen.resolve_pipeline_entry", resolve)


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    scr = AppScreen("mask")
    qtbot.addWidget(scr)
    scr.resize(1280, 800)
    scr.show()
    qtbot.waitExposed(scr)
    return scr


class _Watch(QObject):
    """A 2 ms heartbeat and the first paint of the busy bar."""

    def __init__(self, progress):
        super().__init__()
        self.beats = []
        self.painted = None
        self._timer = QTimer()
        self._timer.setInterval(2)
        self._timer.timeout.connect(
            lambda: self.beats.append(time.perf_counter()))
        progress.installEventFilter(self)
        self._timer.start()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Paint and self.painted is None:
            self.painted = time.perf_counter()
        return False

    def worst_gap(self, start, end):
        """The longest the GUI thread went without a beat in [start, end]."""
        marks = [start] + [b for b in self.beats if start < b < end] + [end]
        return max(b - a for a, b in zip(marks, marks[1:]))


def _finish(qtbot, scr, received, count):
    qtbot.waitUntil(lambda: len(received) >= count, timeout=30000)
    qtbot.waitUntil(lambda: scr._btn_run.isEnabled(), timeout=30000)


def _click_and_watch(qtbot, scr):
    """Press Run and return (click time, the watch) once the bar painted."""
    scr._progress.setVisible(False)
    qtbot.wait(30)
    watch = _Watch(scr._progress)
    qtbot.wait(20)
    clicked = time.perf_counter()
    scr._btn_run.click()
    qtbot.waitUntil(lambda: watch.painted is not None, timeout=10000)
    return clicked, watch


def test_the_busy_bar_paints_before_the_import_and_the_gui_keeps_beating(
        qtbot, screen, monkeypatch, make_module):
    """Click to first paint holds the GUI far less than the import did."""
    received = []
    old = make_module()
    _use(monkeypatch, _resolver(old, received, hooks=False), old)
    clicked, watch = _click_and_watch(qtbot, screen)
    reference = watch.worst_gap(clicked, watch.painted)
    _finish(qtbot, screen, received, 1)
    assert reference >= IMPORT_S * 0.9

    new = make_module()
    _use(monkeypatch, _resolver(new, received), new)
    clicked, watch = _click_and_watch(qtbot, screen)
    painted_after = watch.painted - clicked
    worst = watch.worst_gap(clicked, watch.painted)
    _finish(qtbot, screen, received, 2)

    assert worst < reference / 4, (worst, reference)
    assert painted_after < reference / 2, (painted_after, reference)
    assert received[0] == received[1]


def test_the_wait_disables_run_shows_the_bar_and_resumes_once(
        qtbot, screen, monkeypatch, make_module):
    """One wait, one import, one run; Run and the bar are put back."""
    received = []
    name = make_module()
    _use(monkeypatch, _resolver(name, received), name)
    warmed = []
    real_warm = screen._console.warm_notice_glyphs
    monkeypatch.setattr(screen._console, "warm_notice_glyphs",
                        lambda: warmed.append(real_warm()))

    screen._on_run()
    assert screen._entry_import is not None
    assert not screen._btn_run.isEnabled()
    assert screen._progress.isVisibleTo(screen)
    screen._on_run()
    qtbot.waitUntil(lambda: bool(bridge._ENTRY_IMPORTS), timeout=5000)
    assert list(bridge._ENTRY_IMPORTS) == [name]
    _finish(qtbot, screen, received, 1)
    qtbot.wait(100)

    assert len(received) == 1
    assert screen._entry_import is None
    assert warmed
    assert not screen._progress.isVisibleTo(screen)


def test_a_second_run_does_not_wait(qtbot, screen, monkeypatch, make_module):
    """First run waits for the import; the second starts at once."""
    received = []
    name = make_module(0.05)
    _use(monkeypatch, _resolver(name, received), name)

    screen._on_run()
    assert screen._entry_import is not None
    assert getattr(screen, "_thread", None) is None
    _finish(qtbot, screen, received, 1)
    qtbot.wait(50)

    screen._on_run()
    assert screen._entry_import is None
    assert not screen._btn_run.isEnabled()
    _finish(qtbot, screen, received, 2)
    assert received[0] == received[1]


def test_an_override_is_carried_through_the_wait(qtbot, screen, monkeypatch,
                                                 make_module):
    """A re-fit's settings reach the pipeline unchanged after the wait."""
    received = []
    old = make_module(0.05)
    _use(monkeypatch, _resolver(old, received, hooks=False), old)
    screen._on_run(override={"src": "/data/plate", "marker": 7})
    _finish(qtbot, screen, received, 1)

    new = make_module(0.05)
    _use(monkeypatch, _resolver(new, received), new)
    screen._on_run(override={"src": "/data/plate", "marker": 7})
    assert screen._entry_import is not None
    _finish(qtbot, screen, received, 2)
    assert received[0] == received[1]
    assert received[1]["marker"] == 7


def _dialogs(monkeypatch):
    from spacr.qt.screens import app_screen

    shown = []
    monkeypatch.setattr(app_screen.QMessageBox, "warning",
                        lambda *a: shown.append(("warning",) + a[1:]))
    monkeypatch.setattr(app_screen.QMessageBox, "information",
                        lambda *a: shown.append(("information",) + a[1:]))
    return shown


def test_bad_settings_are_reported_the_same_after_the_wait(
        qtbot, screen, monkeypatch, make_module):
    """The settings are collected after the wait and fail the same way."""
    shown = _dialogs(monkeypatch)

    def refuse():
        raise ValueError("diameter must be a number")

    monkeypatch.setattr(screen._settings_model, "collect", refuse)
    old = make_module(0.05)
    _use(monkeypatch, _resolver(old, [], hooks=False), old)
    screen._on_run()
    assert len(shown) == 1

    new = make_module(0.05)
    _use(monkeypatch, _resolver(new, []), new)
    screen._on_run()
    assert len(shown) == 1
    qtbot.waitUntil(lambda: len(shown) == 2, timeout=5000)
    assert shown[0] == shown[1]
    assert shown[1][1:] == ("Bad settings", "diameter must be a number")
    assert screen._btn_run.isEnabled()
    assert not screen._progress.isVisibleTo(screen)


def test_an_import_that_fails_still_says_not_runnable(
        qtbot, screen, monkeypatch, make_module):
    """The worker's failure is retried where it always was, and reported."""
    shown = _dialogs(monkeypatch)
    old = make_module(0.0, fail=True)
    _use(monkeypatch, _resolver(old, [], hooks=False), old)
    screen._on_run()
    assert [s[0] for s in shown] == ["information"]

    new = make_module(0.05, fail=True)
    _use(monkeypatch, _resolver(new, []), new)
    screen._on_run()
    qtbot.waitUntil(lambda: len(shown) == 2, timeout=5000)
    assert shown[0][0] == shown[1][0] == "information"
    assert shown[1][1] == shown[0][1]
    assert not bridge._entry_import_pending("mask")
    assert screen._btn_run.isEnabled()
    assert screen._entry_import is None


def test_a_module_imported_meanwhile_resumes_on_the_first_tick(
        qtbot, screen, monkeypatch, make_module):
    """Nothing to start at the first tick: the run goes ahead."""
    received = []
    name = make_module(0.0)
    _use(monkeypatch, _resolver(name, received), name)
    screen._on_run()
    assert screen._entry_import is not None
    importlib.import_module(name)
    _finish(qtbot, screen, received, 1)
    assert bridge._ENTRY_IMPORTS == {}


def test_a_tick_with_nothing_waiting_does_nothing(screen):
    screen._check_the_entry_import()
    assert getattr(screen, "_entry_import", None) is None


def test_the_notice_glyphs_are_laid_out_once_per_size(screen):
    console = screen._console
    console.__dict__.pop("_warmed_glyph_sizes", None)
    assert console.warm_notice_glyphs() is True
    assert console.warm_notice_glyphs() is False
    console.set_console_font_pt(console._font_pt + 3)
    assert console.warm_notice_glyphs() is True


def _timed_notice(console, pt, *, warm):
    console.set_console_font_pt(pt)
    console.append_notice("{word}\n", word="ascii")
    if warm:
        console.warm_notice_glyphs()
    started = time.perf_counter()
    console.append_notice("→ Starting {module}…\n", module="mask")
    return time.perf_counter() - started


def test_the_warmed_glyph_makes_the_first_notice_cheap(screen):
    """Self-calibrated: a cold size's first arrow against a warmed size's."""
    console = screen._console
    console.__dict__.pop("_warmed_glyph_sizes", None)
    base = console._font_pt
    cold = _timed_notice(console, base + 5, warm=False)
    warmed = _timed_notice(console, base + 7, warm=True)
    assert warmed < max(cold / 3, 0.005), (warmed, cold)


def test_the_real_resolver_carries_both_hooks():
    resolve = bridge.resolve_pipeline_entry
    assert resolve.entry_import_pending is bridge._entry_import_pending
    assert (resolve.import_off_the_gui_thread
            is bridge._import_the_entry_off_the_gui_thread)


def test_the_module_table_names_what_the_resolver_imports(monkeypatch):
    """Every ``if app_key == K: from M import f`` in the resolver is K -> M."""
    monkeypatch.undo()
    source = textwrap.dedent(inspect.getsource(
        bridge.resolve_pipeline_entry))
    found = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "app_key"
                and isinstance(test.comparators[0], ast.Constant)):
            continue
        first = node.body[0]
        if isinstance(first, ast.ImportFrom):
            found[test.comparators[0].value] = first.module
    assert found
    assert found == bridge._ENTRY_MODULES


def test_is_imported_waits_for_a_module_still_initialising(monkeypatch):
    name = f"spacr_entry_half_{uuid.uuid4().hex[:8]}"
    assert not bridge._module_is_imported(name)
    half = SimpleNamespace(__spec__=SimpleNamespace(_initializing=True))
    monkeypatch.setitem(sys.modules, name, half)
    assert not bridge._module_is_imported(name)
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(__spec__=None))
    assert bridge._module_is_imported(name)


def test_the_import_thread_lifecycle(monkeypatch, make_module):
    name = make_module()
    monkeypatch.setitem(bridge._ENTRY_MODULES, "mask", name)
    assert bridge._import_the_entry_off_the_gui_thread("unknown") is None
    assert not bridge._entry_import_pending("unknown")
    assert bridge._entry_import_pending("mask")

    thread = bridge._import_the_entry_off_the_gui_thread("mask")
    assert thread is not None and thread.is_alive()
    assert thread is not threading.current_thread()
    assert bridge._import_the_entry_off_the_gui_thread("mask") is thread
    assert bridge._entry_import_pending("mask")
    thread.join(10)
    assert name in sys.modules
    assert bridge._import_the_entry_off_the_gui_thread("mask") is None
    assert not bridge._entry_import_pending("mask")

    other = make_module(0.0)
    importlib.import_module(other)
    monkeypatch.setitem(bridge._ENTRY_MODULES, "measure", other)
    assert not bridge._entry_import_pending("measure")
    assert bridge._import_the_entry_off_the_gui_thread("measure") is None
    assert other not in bridge._ENTRY_IMPORTS


class _Lock:
    def __init__(self):
        self.calls = []

    def acquire(self):
        self.calls.append("acquire")

    def release(self):
        self.calls.append("release")


def test_the_import_holds_the_heavy_lock_and_brings_the_gpu_up(
        monkeypatch, make_module, fresh_imports):
    lock = _Lock()
    probes = []
    monkeypatch.setitem(sys.modules, "spacr.qt.app",
                        SimpleNamespace(HEAVY_IMPORT_LOCK=lock))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: probes.append(1))))
    name = make_module(0.0)
    was_enabled = gc.isenabled()
    gc.enable()
    try:
        bridge._import_the_entry_module(name)
        assert gc.isenabled()
    finally:
        if not was_enabled:
            gc.disable()
    assert name in sys.modules
    assert lock.calls == ["acquire", "release"]
    assert probes == [1]
    assert fresh_imports == [True]


def test_the_import_without_the_app_or_torch_and_with_gc_off(
        monkeypatch, make_module, fresh_imports):
    monkeypatch.delitem(sys.modules, "spacr.qt.app", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    name = make_module(0.0)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        bridge._import_the_entry_module(name)
        assert not gc.isenabled()
    finally:
        if was_enabled:
            gc.enable()
    assert name in sys.modules
    assert fresh_imports == [True]


def test_a_failing_import_is_logged_not_raised(monkeypatch, make_module,
                                               fresh_imports, caplog):
    lock = _Lock()
    monkeypatch.setitem(sys.modules, "spacr.qt.app",
                        SimpleNamespace(HEAVY_IMPORT_LOCK=lock))
    name = make_module(0.0, fail=True)
    with caplog.at_level("DEBUG", logger=bridge.LOG.name):
        bridge._import_the_entry_module(name)
    assert name not in sys.modules
    assert lock.calls == ["acquire", "release"]
    assert fresh_imports == [True]
    assert any("off the GUI thread" in r.getMessage() for r in caplog.records)
