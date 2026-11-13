"""A module screen is built, re-themed, run and closed whatever else fails.

Building: contributing to a saved workspace or a theme helper is extra, so a
refusal there still gives a screen, and a screen that opts into tabs mounts
each category as a page. Re-theming: prose boxes are re-rendered on every
palette event, including on a screen mid-teardown. Running: the worker is
handed plain values resolved on the GUI thread, a grid that cannot mark the
run does not stop it, and a project locked read-only by another process is
refused without touching its inputs. Closing: every teardown step is tried
even when an earlier one finds its widget already gone.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
import types
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent                            # noqa: E402

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("the widget has been destroyed")


def _built(qtbot, key):
    screen = AppScreen(key)
    qtbot.addWidget(screen)
    return screen


# -- building a screen when an enrolment fails -------------------------------

def test_a_screen_is_built_even_when_an_extra_refuses(subtests, monkeypatch,
                                                      qtbot):
    """A refusing workspace registry or theme helper still yields a screen with
    a settings model, and a screen in SETTINGS_AS_TABS mounts its categories as
    titled tab pages (every page still a recorded section)."""
    from PySide6.QtWidgets import QTabWidget
    from spacr.qt import theme

    with subtests.test(case="workspace registry refuses"), \
            monkeypatch.context() as mp:
        mp.setattr(AppScreen, "register_workspace", _boom)
        assert _built(qtbot, "regression")._settings_model is not None
    with subtests.test(case="theme helper refuses"), \
            monkeypatch.context() as mp:
        mp.setattr(theme, "take_the_scroll_arrows_off", _boom)
        assert _built(qtbot, "regression")._settings_model is not None
    with subtests.test(case="settings as tabs"), monkeypatch.context() as mp:
        mp.setattr(AppScreen, "SETTINGS_AS_TABS", frozenset({"measure"}))
        built = _built(qtbot, "measure")
        tabs = built._settings_tabs
        assert isinstance(tabs, QTabWidget)
        assert tabs.count() > 2
        assert all(tabs.tabText(index) for index in range(tabs.count()))
        assert len(built._settings_sections) >= tabs.count()


# -- re-theming the prose boxes ------------------------------------------------

def _prose_none(screen, mp):
    mp.setattr(screen, "_section_explainers", {}, raising=False)
    screen._retheme_section_explainers()
    assert screen._section_explainers == {}


def _prose_palette_unresolved(screen, mp):
    from spacr.qt import theme

    attempts = []

    def refusing_palette():
        attempts.append(True)
        raise RuntimeError("the palette is unavailable")

    mp.setattr(screen, "_section_explainers",
               {"Paths": types.SimpleNamespace(setHtml=_boom)}, raising=False)
    mp.setattr(theme, "active_palette", refusing_palette)
    screen._retheme_section_explainers()
    assert attempts == [True]


def _prose_rendered(screen, mp, first):
    rendered = []
    mp.setattr(screen, "_section_explainers", {
        "Paths": first,
        "General": types.SimpleNamespace(
            setHtml=lambda html: rendered.append(html)),
    }, raising=False)
    screen._retheme_section_explainers()
    assert len(rendered) == 1


def _prose_removed_box_skipped(screen, mp):
    _prose_rendered(screen, mp, None)


def _prose_gone_box_spares_the_rest(screen, mp):
    _prose_rendered(screen, mp, types.SimpleNamespace(setHtml=_boom))


def _model_box_absent(screen, mp):
    mp.setattr(screen, "_model_explainer", None, raising=False)
    screen._refresh_model_explainer()
    assert screen._model_explainer is None


def _model_box_unreadable_setting(screen, mp):
    mp.setattr(screen._settings_model, "_read_widget", _boom)
    screen._refresh_model_explainer()
    assert screen._model_explainer.toHtml()


def test_prose_boxes_are_rethemed_box_by_box(subtests, monkeypatch, qtbot):
    """No boxes means nothing to do; an unresolvable palette leaves every box as
    it was; a removed or destroyed box is skipped without stopping the rest; the
    model box renders nothing when absent and falls back to defaults when a
    setting cannot be read."""
    screen = AppScreen("regression")
    try:
        cases = {
            "no prose boxes": _prose_none,
            "palette unresolved": _prose_palette_unresolved,
            "removed box": _prose_removed_box_skipped,
            "destroyed box": _prose_gone_box_spares_the_rest,
            "no model box": _model_box_absent,
            "unreadable setting": _model_box_unreadable_setting,
        }
        completed = []
        for name, case in cases.items():
            with subtests.test(case=name), monkeypatch.context() as mp:
                case(screen, mp)
                completed.append(name)
        assert completed == list(cases), "a case did not complete"
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


# -- closing a screen that is already half gone -------------------------------

def _close_flowview_already_gone(qtbot, mp):
    class Gone:
        asked = 0

        def shutdown(self):
            self.asked += 1
            raise RuntimeError("Internal C++ object already deleted.")

    screen = _built(qtbot, "mask")
    section = Gone()
    screen._flowview_section = section
    screen.close()
    assert section.asked == 1, "the section was never shut down"
    assert screen.isVisible() is False, "the screen did not close"


def _close_flowview_cleanly(qtbot, mp):
    closed = []
    screen = _built(qtbot, "mask")
    screen._flowview_section = types.SimpleNamespace(
        shutdown=lambda: closed.append(1))
    screen.close()
    assert closed == [1]


def _close_with_one_job_pool(qtbot, mp):
    screen = _built(qtbot, "regression")
    drained = []
    screen._usage_jobs = None
    screen._jobs = types.SimpleNamespace(shutdown=lambda: drained.append("jobs"))
    screen.closeEvent(QCloseEvent())
    assert drained == ["jobs"]


def _close_menus_not_retired(qtbot, mp):
    from spacr.qt import widget_cleanup

    screen = _built(qtbot, "regression")
    mp.setattr(widget_cleanup, "retire_pyqtgraph_menus", _boom)
    event = QCloseEvent()
    event.ignore()
    screen.closeEvent(event)
    assert event.isAccepted()


def _close_with_stand_ins_that_refuse(qtbot, mp):
    from PySide6.QtWidgets import QApplication

    screen = _built(qtbot, "regression")
    QApplication.processEvents()
    screen._usage_timer = None
    screen._figure_queue = object()          # no clear()
    screen._umap_explorer = object()         # no close()
    event = QCloseEvent()
    event.ignore()
    screen.closeEvent(event)
    assert event.isAccepted()
    assert screen._backdrops_ready is False


def test_closing_tries_every_teardown_step(subtests, monkeypatch, qtbot):
    """The FlowView section is shut down whether it closes cleanly or its C++
    half has gone, a job pool that was never built is stepped over while the
    other is drained, and a missing poll timer, a figure queue or explorer that
    cannot be cleared or closed, or menus that cannot be retired do not stop
    the close from being accepted."""
    cases = {
        "flowview already gone": _close_flowview_already_gone,
        "flowview closes cleanly": _close_flowview_cleanly,
        "one job pool": _close_with_one_job_pool,
        "stand-ins that refuse": _close_with_stand_ins_that_refuse,
        "menus not retired": _close_menus_not_retired,
    }
    completed = []
    for name, case in cases.items():
        with subtests.test(case=name), monkeypatch.context() as mp:
            case(qtbot, mp)
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


# -- what a run is handed ------------------------------------------------------

def test_a_run_gets_plain_settings_and_survives_a_grid_that_refuses(
        subtests, monkeypatch, qtbot, caplog):
    """A UMAP run is told the canvas colours as plain strings resolved on the
    GUI thread; a mask run carries the hashing preference, and when the figure
    grid cannot mark the run or the preference cannot be read both are logged,
    the run still starts and ``hash_inputs`` is simply left out."""
    from spacr.qt import preferences
    from spacr.qt.theme import active_palette

    caplog.set_level(logging.DEBUG, logger=aps.LOG.name)

    with subtests.test(case="umap colours"), monkeypatch.context() as mp:
        seen = {}
        mp.setattr(aps, "resolve_pipeline_entry",
                   lambda _key: lambda settings: seen.update(settings))
        screen = _built(qtbot, "umap")
        screen._on_run()
        qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=20000)
        palette = active_palette()
        assert seen["_plot_theme"] == {"background": palette["surface_alt"],
                                       "foreground": palette["fg"],
                                       "border": palette["fg"]}

    with subtests.test(case="grid refuses to mark"), \
            monkeypatch.context() as mp:
        seen = []
        mp.setattr(aps, "resolve_pipeline_entry",
                   lambda _key: lambda settings: seen.append(dict(settings)))
        screen = _built(qtbot, "mask")
        mp.setattr(preferences, "get_hash_inputs", lambda: True)
        screen._on_run()
        qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=20000)
        assert seen[0]["hash_inputs"] is True

        mp.setattr(screen._figure_queue, "mark_run", _boom)
        mp.setattr(preferences, "get_hash_inputs", _boom)
        screen._on_run()
        qtbot.waitUntil(lambda: len(seen) == 2, timeout=20000)
        qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=20000)
        assert "could not mark the run on the figure grid" in caplog.text
        assert "could not read the hashing preference" in caplog.text
        assert "hash_inputs" not in seen[1]


def test_run_refuses_a_real_read_only_project_without_changing_its_inputs(
        qtbot, tmp_path, monkeypatch):
    """A project whose lock another live process holds, opened read-only, makes
    Run say so (naming the folder) without creating a worker, and leaves the
    settings, the image and the lock file byte-for-byte as they were."""
    import numpy as np
    import tifffile
    from spacr.qt import crash_recovery

    screen = _built(qtbot, "mask")
    monkeypatch.setenv("SPACR_LOCK_DIR", str(tmp_path / "locks"))
    project = tmp_path / "project"
    project.mkdir()
    image = project / "field.tif"
    tifffile.imwrite(image, np.arange(72, dtype=np.uint16).reshape(8, 9))
    screen._settings_model.set_value_for_key("src", str(project))
    lock_path = Path(crash_recovery._lock_file_for(project))
    ready = tmp_path / "holder-ready"
    holder_code = textwrap.dedent("""
        import os, sys
        from pathlib import Path
        from PySide6.QtCore import QLockFile
        lock = QLockFile(sys.argv[1])
        lock.setStaleLockTime(0)
        assert lock.tryLock(5000)
        Path(sys.argv[2]).write_text(str(os.getpid()))
        sys.stdin.readline()
        lock.unlock()
    """)
    process = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(lock_path), str(ready)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True)
    try:
        qtbot.waitUntil(lambda: ready.exists() or process.poll() is not None,
                        timeout=7000)
        assert process.poll() is None and ready.read_text() == str(process.pid)
        lock_bytes = lock_path.read_bytes()
        image_bytes = image.read_bytes()
        settings = deepcopy(screen._settings_model.collect())
        holders, messages = [], []
        monkeypatch.setattr(
            crash_recovery, "_ask_about_a_locked_project",
            lambda _parent, _path, holder: holders.append(holder) or "read_only")
        monkeypatch.setattr(aps.QMessageBox, "information",
                            lambda _parent, _title, text: messages.append(text))
        monkeypatch.setattr(screen, "_wait_for_the_entry_import",
                            lambda _override: False)
        monkeypatch.setattr(aps, "resolve_pipeline_entry",
                            lambda _key: object())
        monkeypatch.setattr(
            aps, "make_thread",
            lambda *_a, **_k: pytest.fail("Read-only Run must not create a worker"))

        screen._on_run()

        assert len(holders) == 1 and holders[0]["pid"] == process.pid
        assert (len(messages) == 1 and "read-only" in messages[0]
                and str(project) in messages[0])
        assert screen._thread is None and getattr(screen, "_worker", None) is None
        assert screen._settings_model.collect() == settings
        assert image.read_bytes() == image_bytes
        assert lock_path.read_bytes() == lock_bytes
        assert process.poll() is None
    finally:
        try:
            process.communicate("\n", timeout=5)
        except (subprocess.TimeoutExpired, BrokenPipeError):
            process.kill()
            process.communicate(timeout=5)
        finally:
            crash_recovery._PROJECT_CHOICES.pop(str(lock_path), None)
