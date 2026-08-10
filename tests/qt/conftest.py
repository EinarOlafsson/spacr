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
def _widget_qss_registrars_loaded():
    """Fill ``theme._WIDGET_QSS`` before any test can snapshot it.

    ``theme.load_widget_qss_registrars()`` imports the ~37 modules that
    register a widget QSS block, and it LATCHES on
    ``theme._QSS_REGISTRARS_LOADED`` — it runs once per process and nothing
    resets the flag. Until it has run, the registry holds only the handful
    of blocks whose modules happened to be imported already (19, in a run
    that starts with ``test_field_fade``).

    That turns the ordinary save/restore fixture into a one-way shrink.
    ``test_field_fade``'s ``prefs_sandbox`` and ``test_registration_seams``'s
    ``qss_sandbox`` both do ``saved = dict(theme._WIDGET_QSS)`` at setup and
    ``clear() + update(saved)`` at teardown. Taken before the loader has run,
    that snapshot is 19 entries, and putting it back deletes the other 18
    permanently — the loader has already latched, so it can never refill
    them. Every ``stylesheet()`` built afterwards in that process is missing
    them, which is the black-box bug ``test_widget_qss_is_complete`` exists
    to catch, reported against whichever file drew the short straw
    (``SettingsSearchPane has no rule in a freshly built stylesheet``).

    Loading here, before the first test, means every such snapshot is taken
    of the full registry, so restoring one is a no-op instead of a deletion.
    Fixed at the session level rather than in the two fixtures because the
    shape — snapshot a lazily-filled global, restore it — is one anybody
    writing the next sandbox fixture would reproduce, and would have no
    reason to suspect.

    Safe with no ``QApplication`` yet: the loader guards every import
    individually and none of them needs one.
    """
    try:
        from spacr.qt import theme
        theme.load_widget_qss_registrars()
    except Exception:
        pass
    yield


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
def _restore_console_level_policy():
    """Put the in-app console's level gate back the way the test found it.

    ``verbose_logger`` keeps ONE ``_ConsoleForwarder`` for the process, and
    ``apply_console_levels`` gates it by mutating a ``LevelSetFilter`` on
    that handler rather than by swapping the handler out — deliberately, so
    the set can change while another thread is mid-log. The cost is that the
    gate is process-global and lives for the rest of the run.

    Anything that launches the GUI installs one.
    ``app.launch()`` -> ``logging_util.apply_level_policy(...)`` ->
    ``apply_console_levels(DEFAULT_CONSOLE_LEVELS)`` leaves the console
    showing ``{WARNING, ERROR, CRITICAL}`` and nothing takes it off again.
    Every later test that logs below WARNING then watches its records
    vanish at the handler: ``test_qt_worker_teardown``'s two console tests
    emit on ``spacr.trace`` at DEBUG, saw an empty console sink, and failed
    on "the console target received nothing at all" — never reaching the
    re-entrancy and reopen assertions they exist to make. They pass alone
    and fail after ``test_cov_qt_app::test_launch_drains_the_ai_consoles_on_quit``.

    Restored here rather than in those tests because the leak belongs to
    every caller of ``launch()``, and a test that logs is not obviously a
    test that depends on the console gate — the next one to be bitten would
    have the same day debugging it.

    The handler's own ``level`` goes with the filters: ``apply_console_levels``
    forces it to DEBUG so the filter is what decides, and that is just as
    much a change to the policy as the filter is.
    """
    try:
        from spacr.qt import verbose_logger as vl
    except Exception:
        yield
        return

    handler = vl._handler
    saved = ((list(handler.filters), handler.level)
             if handler is not None else None)
    try:
        yield
    finally:
        current = vl._handler
        if current is not None:
            # A handler that is not the one we measured — created during the
            # test, or swapped for a new one — carries a policy that is
            # entirely the test's, so "before" for it is no gate at all.
            # The handler object itself is left attached; detaching it is a
            # different concern and tests hold references to it.
            restorable = current is handler and saved is not None
            filters, level = saved if restorable else ([], 0)
            for existing in list(current.filters):
                current.removeFilter(existing)
            for existing in filters:
                current.addFilter(existing)
            current.setLevel(level)


@pytest.fixture(autouse=True)
def _invalidate_field_fade_cache():
    """Drop the cached field-fade preference so the next test re-reads it.

    ``field_fade._enabled`` is a module-global cache, and it is one on
    purpose: ``field_fade_enabled()`` is called on every paint event and
    building a ``QSettings`` per paint would put a file-format lookup in the
    render loop. It is dropped by ``invalidate_field_fade()``, which both
    setters call — so the cache and the store agree inside the application.

    A test that sandboxes ``QSettings`` breaks that agreement. The store is
    restored at teardown; the cache is not, so it goes on answering with the
    sandbox's value. ``test_spacr_mode``'s Extra-Performance tests turn the
    fade off, and every ``stylesheet()`` built after them carries an EMPTY
    FieldFade block — empty being the documented, load-bearing output of
    "off". ``test_widget_qss_is_complete`` then fails with "registered but
    rendering to nothing: ['FieldFade']", which reads like a broken QSS
    block and is really a stale bool.

    Dropped at teardown rather than restored to a saved value: there is
    nothing to restore. The cache is a copy of the preference, and the
    preference has already been put back by whatever sandboxed it, so the
    correct next value is "ask the store again".
    """
    try:
        from spacr.qt.widgets import field_fade
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            field_fade.invalidate_field_fade()
        except Exception:
            pass


@pytest.fixture
def deferred_deletions_flushed(qapp):
    """Deliver the ``deleteLater`` calls earlier tests already made.

    Opt-in, and the only fixture here that is: a test needs it when it
    measures something PER LIVE WIDGET, where a widget the previous test
    finished with is indistinguishable from one this test is responsible
    for.

    pytest-qt closes and ``deleteLater()``s the widgets it was given, but
    ``deleteLater`` only posts an event — the object dies on the next spin of
    the event loop, and a headless test file may never spin one. Eight
    ``AppScreen``s survive ``test_preferences_gear`` that way: not visible,
    not parented, referenced by nothing but their own bound-method cycles,
    already asked to die. They still answer ``PaletteChange``, so
    ``apply_preferences_to_app`` pays for a wallpaper lookup on each of them
    and ``test_space_theme``'s cost assertion counted 17 instead of 1.

    This is NOT the "reach across every live widget at teardown" fixture
    that was removed on 2026-08-08 for segfaulting the run three ways. It
    deletes nothing on its own initiative — it delivers deletions their
    owners already requested — and it runs at SETUP, when the previous
    test's teardown is complete and pytest-qt is not part-way through
    closing anything.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    yield qapp


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


# REMOVED 2026-08-08: `_drain_job_runners`.
#
# It was added to fix a segfault in the resource-usage sampler at 28% of
# the suite, and it caused more than it fixed. Measured, by running the
# suite with and without it:
#
#     WITH     segfault at 28% (dead widgets), 33% (event loop pumped at
#              teardown), 35% -- three separate crashes, two of them in
#              this fixture itself
#     WITHOUT  reached 66% with zero crashes and zero failures, and only
#              stopped because the harness timeout expired
#
# Three bugs came out of one fixture: a 3000ms-per-runner budget that took
# the run from "crashes at 28%" to "still going at 35% after 80 minutes";
# a crash on widgets whose C++ side was already gone; and a crash on live
# ones, because shutdown() pumps the event loop and that runs pending
# deleteLater calls on widgets pytest-qt was about to close itself.
#
# The shape was the problem. A teardown fixture that reaches across every
# widget in the application, during Qt teardown, is dangerous in a way no
# amount of guarding fixes -- each guard I added revealed the next crash.
#
# If the original 28% sampler segfault returns, fix it where it lives
# (AppScreen stops its timer and drains its runners in closeEvent already;
# a test that never closes the screen is the real gap) rather than by
# reaching across the application again.


@pytest.fixture(autouse=True)
def _issue_prompt_does_not_block():
    """File issues without a prompt, unless a test says otherwise.

    `_on_file_issue` asks before filing, and a QMessageBox in a headless run
    has nobody to answer it -- which is the hang that cost this suite its
    entire run once already (instruction 47). Rather than leave that trap
    for the next test that touches the reporter, the default here is
    `always`: the prompt is skipped, and the tests that exercise the FILING
    path get to exercise it.

    A test about the prompt itself sets the mode it wants; this restores
    whatever was there afterwards, so it cannot leak either way.
    """
    try:
        from spacr.qt import preferences
    except Exception:
        yield
        return
    try:
        original = preferences.get_issue_prompt_mode()
        preferences.set_issue_prompt_mode(preferences.ISSUE_PROMPT_ALWAYS)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            preferences.set_issue_prompt_mode(original)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _no_unguarded_modals(monkeypatch):
    """Turn any unguarded modal dialog into a fast failure, not a hang.

    THREE separate hangs in this suite have been the same thing: a
    `.exec()` on a dialog in a headless run, with nobody to click it, and
    the whole suite sitting there until someone kills it.

        _on_stop        -> shutdown.ask_how_to_quit   (QMessageBox)
        _on_file_issue  -> the report prompt          (QMessageBox)
        drag-and-drop   -> dnd_handlers:730           (QDialog)

    Each was found by a different multi-hour run, and each was fixed by
    adding a guard to ONE test file. That does not scale -- the next
    `.exec()` anyone adds hangs the suite again, and the cost of finding it
    is another wasted run.

    So the default is now global: exec() raises. A test that genuinely
    wants to drive a dialog patches it itself, and because monkeypatch in
    the test body runs AFTER this fixture, that patch wins.

    The message names the fix, because the failure it produces is otherwise
    mystifying to whoever meets it first.
    """
    from PySide6.QtWidgets import QDialog, QMessageBox

    def _refuse(self, *args, **kwargs):
        raise AssertionError(
            f"{type(self).__name__}.exec() was called in a headless test. "
            "A modal has nobody to answer it here and hangs the run. Stub "
            "the call that opens it -- e.g. monkeypatch "
            "spacr.qt.shutdown.ask_how_to_quit -- or patch this dialog's "
            "exec in the test that wants it.")

    for cls in (QDialog, QMessageBox):
        for name in ("exec", "exec_"):
            monkeypatch.setattr(cls, name, _refuse, raising=False)
