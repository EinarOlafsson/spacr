"""Pytest fixtures for the Qt GUI test suite.

Runs offscreen — no X server required. Skips cleanly if PySide6 or
pytest-qt is not installed so the rest of the suite still runs.
"""
from __future__ import annotations

from importlib.util import find_spec
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Skipping while this conftest is imported aborts collection of the entire
# repository on pytest 7, leaving pytest with exit code 5 ("no tests
# collected").  Ignore only this directory's test modules when an optional Qt
# test dependency is absent so the non-Qt suite can still run.
_QT_TEST_DEPENDENCIES = ("PySide6", "pytestqt")
collect_ignore_glob = (
    ["test_*.py"]
    if any(find_spec(module) is None for module in _QT_TEST_DEPENDENCIES)
    else []
)


@pytest.fixture(scope="session")
def qt_theme_applied(qapp):
    """Apply the spacr palette + QSS to the shared QApplication once."""
    from spacr.qt.theme import apply_qpalette, stylesheet
    apply_qpalette(qapp)
    qapp.setStyleSheet(stylesheet())
    return qapp


@pytest.fixture(scope="session", autouse=True)
def _font_scale_starts_at_one():
    """Start every session at scale 1.0, whatever the last one left behind.

    The font scale is persisted, so it is ambient state that survives the
    process. A run that ended with 1.5 set the starting scale for the next
    one, and for the one after that -- and CI, starting from nothing, ran at
    1.0. So the suite measured a different application locally than on the
    runners, silently, for as long as nobody reset it.

    It showed up as failures that read like real regressions and were not:
    ``scaled_px(320)`` returning 480, twenty modules "failing" the smoke test
    on clipped and elided labels, the search strip sitting 11px down. All of
    it was 1.5.

    Per-test restoration is below; this is the floor under it, so the first
    test of a session cannot inherit the last test of the previous one.
    """
    from spacr.qt import preferences

    try:
        preferences.set_font_scale(1.0)
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _restore_font_scale():
    """Put the font scale back the way the test found it.

    The same shape of leak as ``_restore_app_registry`` below, and it took
    ``test_settings_search``'s geometry test down the same way: passes on its
    own file, fails after any other qt file, and the blame lands on whichever
    test drew the short straw rather than on the one that leaked.

    ``test_preferences`` sets the scale ten times and restores it none;
    ``test_zoom_reaches_text`` sets it once and never puts it back;
    ``test_home_variants`` sets it five times and restores twice. Whoever ran
    last decided the scale for everything after them. At 1.5,
    ``test_the_strip_is_a_thin_band_above_the_form_not_over_it`` measured the
    search strip at ``y=11`` instead of ``0`` -- a real geometry difference at
    a scale no assertion in that file had asked for.

    Restored here rather than in each caller, so a file that starts changing
    the scale tomorrow is covered without anyone remembering to.
    """
    from spacr.qt import preferences

    try:
        original = preferences.get_font_scale()
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            preferences.set_font_scale(original)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _restore_app_registry():
    """Put ``spacr.qt.app``'s registry back the way the test found it.

    ``spacr.qt.register_self_registering_modules()`` mutates process-global
    state: ``APPS``, ``APP_FACTORIES``, ``APP_STAGE``, ``APP_META`` and,
    through ``app._META_TARGETS``, ``cli.INTERACTIVE_ONLY``,
    ``app_screen.APP_TITLES`` / ``APP_INTROS`` and
    ``settings_model._APP_API_MODULE``. Ten test files call it, and not one of
    them calls it wrongly — they need the registry a launched GUI has, which
    importing ``spacr.qt.app`` alone does not give them. What none of them did
    was put it back, and nine apps join the list when it runs.

    So every test that ran AFTER one of them saw a longer app list than it saw
    alone, and a whole family of failures was really one leak wearing
    different hats: ``test_control_chart_screen``'s two registration tests
    (``register()`` answering False on its first call), ``test_home_v2``'s
    alpha/beta lists, ``test_home_layout``'s staged-app count, the landing
    page's app count. Each passed on its own file and failed in the suite,
    which is the worst way for a test to fail — the blame lands on whichever
    test drew the short straw rather than on the call that leaked.

    Restoring here rather than in each caller means a file that starts calling
    it tomorrow is covered without anyone remembering. Driven off
    ``_META_TARGETS`` so a new side table is undone without this being edited.
    """
    import sys

    try:
        from spacr.qt import app as app_mod
    except Exception:
        yield
        return

    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    meta = dict(app_mod.APP_META)
    side = []
    for module_name, attribute, _field in app_mod._META_TARGETS:
        module = sys.modules.get(module_name)
        table = getattr(module, attribute, None) if module else None
        if isinstance(table, dict):
            side.append((table, dict(table)))
    try:
        yield
    finally:
        if list(app_mod.APPS) != apps:
            app_mod.APPS[:] = apps
            app_mod._refresh_sections()
        app_mod.APP_FACTORIES.clear()
        app_mod.APP_FACTORIES.update(factories)
        app_mod.APP_STAGE.clear()
        app_mod.APP_STAGE.update(stages)
        app_mod.APP_META.clear()
        app_mod.APP_META.update(meta)
        for table, saved in side:
            table.clear()
            table.update(saved)
        # A side table that was only imported DURING the test is restored on
        # the next test instead; the snapshot above cannot hold what did not
        # exist yet, and re-snapshotting every teardown would defeat the point.


@pytest.fixture(autouse=True)
def _skip_first_launch_tour():
    """The first-launch tour attaches a modal overlay to the MainWindow
    the first time it opens. Left alone, it steals focus + adds widgets
    that break test isolation. Mark it "seen" for every Qt test so
    MainWindow constructs without the overlay."""
    try:
        from spacr.qt.first_run import mark_tour_seen, reset_tour_state
        mark_tour_seen()
        yield
        # Leave the "seen" flag alone — tests that specifically want
        # the tour set force=True (see test_onboarding).
    except Exception:
        yield


@pytest.fixture(autouse=True)
def _drain_job_runners():
    """Stop every background JobRunner before its owner is destroyed.

    A screen polls resource usage on a 2-second QTimer and samples it on a
    JobRunner thread. `AppScreen.closeEvent` stops the timer and drains the
    runners -- but only if the screen is CLOSED, and a test that builds a
    screen and lets it fall out of scope never closes anything. So the
    sampler was still inside `psutil.virtual_memory()` on a worker thread
    while Qt was deleting the widget that owned it, and the process died:

        Fatal Python error: Segmentation fault
          app_screen.py:2987 in _sample_usage
          job_runner.py:60 in _capture
          bridge.py:1018 in run

    at 28% of the qt suite, which is why the run never reached its summary.
    `JobRunner` takes a parent but does not tie its threads to that parent's
    destruction, so nothing else was going to stop them.

    Draining here rather than in JobRunner itself is deliberate: the
    production path already stops cleanly on close, and reaching for
    `QObject.destroyed` would run Python during C++ teardown, which is how
    this codebase earned its other threading crash (INVARIANTS 4).
    """
    yield
    try:
        from PySide6.QtWidgets import QApplication

        from spacr.qt.job_runner import JobRunner
    except Exception:
        return
    app = QApplication.instance()
    if app is None:
        return
    seen = set()
    for widget in list(app.allWidgets()):
        for runner in widget.findChildren(JobRunner):
            if id(runner) in seen:
                continue
            seen.add(id(runner))
            try:
                runner.shutdown()
            except Exception:
                # Teardown is best-effort: a runner already gone is the
                # outcome we wanted, and raising here would fail a test
                # that had already passed.
                pass
