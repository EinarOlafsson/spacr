"""Opening a folded module builds nothing and writes nothing in the click.

Make Masks' fold strip opened Train Cellpose by building the whole screen
inside the click handler (about two seconds of GUI stalls) and rewrote the
crash-report state file on the GUI thread while doing so. The screen is now
built while Make Masks sits idle on show, and an unchanged state is not
rewritten.
"""
from __future__ import annotations

import os
import threading

import pytest


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    monkeypatch.delenv("SPACR_PREWARM", raising=False)
    monkeypatch.delenv("SPACR_BENCHMARK_JSON", raising=False)
    from spacr.qt import app as app_module
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    monkeypatch.setattr(preferences, "_screen_prewarm_allowed",
                        lambda: (True, ""))
    monkeypatch.setattr(app_module, "_prewarm_memory_is_short", lambda: False)
    monkeypatch.setattr(MakeMasksScreen, "_PREBUILD_FIRST_S", 0.05)
    monkeypatch.setattr(MakeMasksScreen, "_PREBUILD_RETRY_S", 0.05)
    win = app_module.MainWindow(initial_app="__home__")
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    yield win
    prewarm = getattr(win, "_screen_prewarm", None)
    if prewarm is not None:
        prewarm.stop()
    win.close()
    qapp.processEvents()


def test_the_click_joins_a_screen_that_was_built_while_idle(
        window, qapp, qtbot, monkeypatch):
    from spacr.qt.screens import train_cellpose

    window._on_nav_selected("make_masks")
    screen = window._screens["make_masks"]
    qtbot.waitUntil(lambda: "train_cellpose" in screen._fold_screens,
                    timeout=20000)
    built = []
    original = train_cellpose.CellposeWorkbenchScreen.__init__

    def counting(self, *args, **kwargs):
        built.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(train_cellpose.CellposeWorkbenchScreen, "__init__",
                        counting)
    main = threading.get_ident()
    writes = []
    original_replace = os.replace

    def recording(source, destination, *rest, **kw):
        if threading.get_ident() == main:
            writes.append(str(destination))
        return original_replace(source, destination, *rest, **kw)

    monkeypatch.setattr(os, "replace", recording)
    prebuilt = screen._fold_screens["train_cellpose"]
    panel = screen.open_folded("train_cellpose")
    qapp.processEvents()
    assert panel is not None
    assert built == []
    assert screen.folded_screen("train_cellpose") is prebuilt
    assert not [w for w in writes if w.endswith("state.json")]


def test_the_prebuild_waits_while_another_screen_is_opening(
        window, qapp, qtbot):
    window._on_nav_selected("make_masks")
    screen = window._screens["make_masks"]
    window._opening_a_screen = True
    qtbot.wait(400)
    assert "train_cellpose" not in screen._fold_screens
    window._opening_a_screen = False
    qtbot.waitUntil(lambda: "train_cellpose" in screen._fold_screens,
                    timeout=20000)


def test_an_unchanged_crash_state_is_not_rewritten(tmp_path, monkeypatch):
    from spacr import crashreport

    monkeypatch.setattr(crashreport, "_reports_dir", lambda: tmp_path)
    crashreport._update_state(affirmed_policy={"explain": True})
    state = tmp_path / "state.json"
    before = state.stat().st_mtime_ns
    written = []
    original = crashreport._write_state
    monkeypatch.setattr(crashreport, "_write_state",
                        lambda s: (written.append(dict(s)), original(s)))
    crashreport._update_state(affirmed_policy={"explain": True})
    assert written == []
    assert state.stat().st_mtime_ns == before
    crashreport._update_state(affirmed_policy={"explain": False})
    assert written[-1]["affirmed_policy"] == {"explain": False}


#: Measured on the workstation (offscreen and native xcb): the longest GUI
#: stall while the idle prebuild runs is 0.28-0.34 s, and while the click joins
#: the built screen to the page strip 0.13-0.34 s. The indivisible Qt calls
#: left (a full restyle of the new screen, the console panel's glyph warm-up and
#: the translate pass) are each 0.1-0.28 s. The budget is about twice the
#: measurement; the single block this replaced was 1.4 s. After the layout of
#: the built screen is also done while idle the click's worst stall is 0.12 s
#: offscreen (two Qt style recomputations of about 0.1 s remain). It may be lowered as
#: those calls are split, never raised.
STALL_BUDGET_S = 0.5


def _longest_gui_gap(qapp, run):
    import time

    from PySide6.QtCore import QTimer

    gaps = []
    last = [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now

    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(tick)
    timer.start()
    last[0] = time.perf_counter()
    try:
        run()
    finally:
        timer.stop()
    return max(gaps or [0.0])


def test_the_idle_build_and_the_click_keep_the_gui_turning(
        window, qapp, qtbot):
    from spacr.qt.screens import make_masks as module

    window._on_nav_selected("make_masks")
    screen = window._screens["make_masks"]
    assert screen._cold_imports is not None
    qtbot.waitUntil(lambda: not screen._cold_imports.is_alive(),
                    timeout=60000)
    assert all(name in __import__("sys").modules
               for name in module._COLD_IMPORTS)

    def build():
        qtbot.waitUntil(lambda: "train_cellpose" in screen._fold_screens,
                        timeout=60000)
        qapp.processEvents()

    assert _longest_gui_gap(qapp, build) < STALL_BUDGET_S
    assert _longest_gui_gap(
        qapp, lambda: screen.open_folded("train_cellpose")) < STALL_BUDGET_S


def test_the_built_screen_is_laid_out_while_idle_too(window, qapp, qtbot):
    window._on_nav_selected("make_masks")
    screen = window._screens["make_masks"]
    qtbot.waitUntil(lambda: "train_cellpose" in screen._fold_screens,
                    timeout=60000)
    qtbot.waitUntil(
        lambda: "train_cellpose" in getattr(screen, "_laid_out_folds", ()),
        timeout=60000)
    built = screen._fold_screens["train_cellpose"]
    assert built.size() == screen.size()
    assert built.layout() is not None and built.layout().isEmpty() is False


def test_a_late_registered_style_block_does_not_restyle_the_whole_screen(
        window, qapp, qtbot, monkeypatch):
    from spacr.qt import gui_scale
    from spacr.qt.theme import register_widget_qss

    window._on_nav_selected("make_masks")
    screen = window._screens["make_masks"]
    qtbot.waitUntil(lambda: "train_cellpose" in screen._fold_screens,
                    timeout=60000)
    qtbot.waitUntil(
        lambda: "train_cellpose" in getattr(screen, "_laid_out_folds", ()),
        timeout=60000)
    register_widget_qss("ProbeLateBlock",
                        lambda palette, opacity: "QLabel#NoSuchProbe { color: red; }",
                        replace=True)
    from spacr.qt import theme

    restyled, roots = [], []
    original = gui_scale._apply_now
    original_ensure = theme.ensure_widget_qss_applied

    def watching(obj, text, raw_set):
        restyled.append(type(obj).__name__)
        return original(obj, text, raw_set)

    def ensuring(*names, root=None):
        roots.append(root)
        return original_ensure(*names, root=root)

    monkeypatch.setattr(gui_scale, "_apply_now", watching)
    monkeypatch.setattr(theme, "ensure_widget_qss_applied", ensuring)
    screen.open_folded("train_cellpose")
    qapp.processEvents()
    assert "MakeMasksScreen" not in restyled
    assert roots and all(root is not screen for root in roots)
    assert any(root is screen._fold_pages for root in roots)


#: The longest gap between two paints of the animated field while the fold
#: opens, measured offscreen and native (0.12-0.14 s); the body moving onto
#: the page strip is one Qt call. It may be lowered, never raised.
FIELD_PAINT_GAP_BUDGET_S = 0.45


def test_the_animated_field_keeps_painting_while_a_fold_opens(
        window, qapp, qtbot, monkeypatch):
    import time

    from spacr.qt.widgets import ambient

    window._on_nav_selected("make_masks")
    screen = window._screens["make_masks"]
    qtbot.waitUntil(lambda: "train_cellpose" in screen._fold_screens,
                    timeout=60000)
    qtbot.waitUntil(
        lambda: "train_cellpose" in getattr(screen, "_laid_out_folds", ()),
        timeout=60000)
    stamps = []
    original = ambient.AmbientWidget.paintEvent

    def counting(self, event):
        stamps.append(time.perf_counter())
        return original(self, event)

    monkeypatch.setattr(ambient.AmbientWidget, "paintEvent", counting)
    if not any(w.isVisible() for w in window.findChildren(ambient.AmbientWidget)):
        pytest.skip("no animated field is showing in this configuration")
    end = time.time() + 1.0
    while time.time() < end:
        qapp.processEvents()
    started = time.perf_counter()
    screen.open_folded("train_cellpose")
    finished = time.perf_counter()
    end = time.time() + 1.0
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.004)
    during = [t for t in stamps if started - 0.5 <= t <= finished + 1.0]
    assert len(during) >= 3
    gaps = [b - a for a, b in zip(during, during[1:])
            if a >= started - 0.5 and b <= finished + 0.5]
    assert max(gaps or [0.0]) < FIELD_PAINT_GAP_BUDGET_S


#: The slowest visible animated field's paint rate (paints per second) in the
#: 1.2 s after a fold opens, as the workstation's gate takes it: the minimum
#: over every visible ``AmbientWidget`` that is not a popup backdrop, a field
#: that appears with the page counted from its own first paint. Measured
#: offscreen: the window's field 80-100/s, the folded screen's own field
#: 55-65/s, from its first paint. It may be raised, never lowered.
FIELD_MIN_PAINTS_PER_S = 10.0


def test_every_visible_field_paints_at_rate_after_a_fold_opens(
        window, qapp, qtbot, monkeypatch):
    import time

    from spacr.qt.widgets import ambient

    window._on_nav_selected("make_masks")
    screen = window._screens["make_masks"]
    qtbot.waitUntil(lambda: "train_cellpose" in screen._fold_screens,
                    timeout=60000)
    qtbot.waitUntil(
        lambda: "train_cellpose" in getattr(screen, "_laid_out_folds", ()),
        timeout=60000)
    seen = {}
    original = ambient.AmbientWidget.paintEvent

    def counting(self, event):
        seen.setdefault(id(self), (self, []))[1].append(time.perf_counter())
        return original(self, event)

    monkeypatch.setattr(ambient.AmbientWidget, "paintEvent", counting)
    if not any(w.isVisible() for w in window.findChildren(ambient.AmbientWidget)):
        pytest.skip("no animated field is showing in this configuration")
    end = time.time() + 0.5
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.004)
    screen.open_folded("train_cellpose")
    opened = time.perf_counter()
    end = time.time() + 1.2
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.004)
    finished = time.perf_counter()
    rates = {}
    for field in window.findChildren(ambient.AmbientWidget):
        if (not field.isVisible()
                or field.property("spacrPopupBackdrop")):
            continue
        every = seen.get(id(field), (field, []))[1]
        before = [t for t in every if t < opened]
        stamps = [t for t in every if t >= opened]
        if len(before) < 3 and not stamps:
            continue
        assert stamps, "a field that was animating stopped painting at the open"
        span = max(finished - (stamps[0] if len(before) < 3 else opened), 0.2)
        rates[id(field)] = len(stamps) / span
    assert rates, "no animated field was showing"
    assert min(rates.values()) >= FIELD_MIN_PAINTS_PER_S, rates


def test_a_folded_screens_field_has_its_first_frame_before_the_click(
        window, qapp, qtbot, monkeypatch):
    """The first frame is shaded while the screen idles, not at the click.

    Lesson 19: the folded screen's own field painted once in its first
    second because the click shaded its first frame on the GUI thread (a
    slow graphics lane took 0.6 s). Here the first shade of every engine
    sleeps 0.6 s; the click must not wait for it.
    """
    import time

    from spacr.qt.widgets import ambient

    original = ambient._DataArtEngine.shade

    def slow_first(self, *args, **kwargs):
        if not getattr(self, "_slow_first_done", False):
            self._slow_first_done = True
            time.sleep(0.6)
        return original(self, *args, **kwargs)

    window._on_nav_selected("make_masks")
    screen = window._screens["make_masks"]
    monkeypatch.setattr(ambient._DataArtEngine, "shade", slow_first)
    qtbot.waitUntil(lambda: "train_cellpose" in screen._fold_screens,
                    timeout=60000)
    qtbot.waitUntil(
        lambda: "train_cellpose" in getattr(screen, "_laid_out_folds", ()),
        timeout=60000)
    folded = screen._fold_screens["train_cellpose"]
    fields = [f for f in folded.findChildren(ambient.AmbientWidget)
              if not f.isVisible()]
    if not fields:
        pytest.skip("the folded screen has no animated field in this configuration")
    qtbot.waitUntil(
        lambda: all("_warmed_first_frame" in f.__dict__ for f in fields
                    if isinstance(f._engine, ambient._DataArtEngine)),
        timeout=60000)
    started = time.perf_counter()
    screen.open_folded("train_cellpose")
    took = time.perf_counter() - started
    assert took < 0.6 + 0.3, took
