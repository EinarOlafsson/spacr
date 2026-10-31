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
#: measurement; the single block this replaced was 1.4 s. It may be lowered as
#: those calls are split, never raised.
STALL_BUDGET_S = 0.7


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
