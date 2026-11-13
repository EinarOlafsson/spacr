"""A decorative backdrop must never hold up, or take down, the module screen.

Opening a module once froze the GUI thread for 3148 ms because the GPU
backdrop's constructor waited on HEAVY_IMPORT_LOCK, held by the startup
preloader for a whole module import. The rule since: the module is what the
user waits for, so the backdrop is what waits. These tests assert the DECISION
(install now, or come back later) rather than a duration.

The other half is re-theming and teardown: a palette event reaches every
screen, including ones mid-teardown, so retheming, removal and the page's own
fill must tolerate backdrops and preferences that no longer answer.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer                                # noqa: E402

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


@pytest.fixture
def regression_screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


# -- the peek at the heavy-import lock ---------------------------------------

def _peek_free(scr, mp):
    assert scr._heavy_lock_is_free() is True


def _peek_held(scr, mp):
    from spacr.qt.app import HEAVY_IMPORT_LOCK

    with HEAVY_IMPORT_LOCK:
        assert scr._heavy_lock_is_free() is False
    assert scr._heavy_lock_is_free() is True


def _peek_releases(scr, mp):
    from spacr.qt.app import HEAVY_IMPORT_LOCK

    assert scr._heavy_lock_is_free() is True
    got = HEAVY_IMPORT_LOCK.acquire(blocking=False)
    try:
        assert got is True, "the peek kept the lock it was only borrowing"
    finally:
        if got:
            HEAVY_IMPORT_LOCK.release()


def _peek_no_lock_module(scr, mp):
    import spacr.qt.widgets.fractal_travel as ft

    mp.setattr(ft, "_heavy_import_lock", lambda: None)
    assert scr._heavy_lock_is_free() is True


def _peek_lock_unreachable(scr, mp):
    import spacr.qt.widgets.fractal_travel as ft

    mp.setattr(ft, "_heavy_import_lock", _boom)
    assert scr._heavy_lock_is_free() is True


def test_the_heavy_lock_peek_reports_without_ever_holding_it(
        subtests, monkeypatch, qtbot):
    """`_heavy_lock_is_free` is True for a free, absent or unreachable lock,
    False only while someone holds it, and never keeps the lock it peeked."""
    scr = _built(qtbot, "mask")
    cases = {
        "free": _peek_free,
        "held": _peek_held,
        "peek releases": _peek_releases,
        "no lock to ask": _peek_no_lock_module,
        "lock unreachable": _peek_lock_unreachable,
    }
    completed = []
    for name, case in cases.items():
        with subtests.test(case=name), monkeypatch.context() as mp:
            case(scr, mp)
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


# -- whether the ambient install goes ahead -----------------------------------

INSTALL_CASES = {
    # name: (_backdrops_ready, has ambient already, lock free,
    #        expect deferred, expect lock asked, expect ambient None after)
    "busy lock defers": (True, False, False, True, True, True),
    "free lock installs now": (True, False, True, False, True, None),
    "already decorated": (True, True, False, False, False, False),
    "not ready for backdrops": (False, False, False, False, False, True),
}


def test_the_ambient_install_waits_for_the_module_not_the_reverse(
        subtests, monkeypatch, qtbot):
    """A busy lock defers the install with a positive-delay retry of itself and
    leaves the screen undecorated; a free lock installs immediately; a screen
    already decorated or not ready never peeks at the lock or queues a retry,
    and an existing backdrop is not rebuilt even if the preferences raise."""
    from spacr.qt import preferences

    shared = _built(qtbot, "mask")
    for name, (ready, has_one, free, deferred, asked_lock,
               none_after) in INSTALL_CASES.items():
        # The free-lock arm really installs, so it gets a screen of its own.
        scr = _built(qtbot, "mask") if free else shared
        with subtests.test(case=name), monkeypatch.context() as mp:
            existing = object() if has_one else None
            mp.setattr(scr, "_ambient", existing)
            mp.setattr(scr, "_backdrops_ready", ready)
            asked = []
            mp.setattr(scr, "_heavy_lock_is_free",
                       lambda: asked.append(1) or free)
            if has_one:
                mp.setattr(preferences, "get_ambient_enabled", _boom)
            scheduled = []
            mp.setattr(QTimer, "singleShot",
                       staticmethod(lambda ms, fn: scheduled.append((ms, fn))))

            scr._install_ambient()

            assert bool(scheduled) is deferred
            assert bool(asked) is asked_lock
            if deferred:
                delay, retry = scheduled[0]
                assert delay > 0
                assert retry == scr._install_ambient
            if none_after is True:
                assert scr._ambient is None, "it built the backdrop anyway"
            if has_one:
                assert scr._ambient is existing


def test_a_rain_that_will_not_install_leaves_the_screen_without_one(
        subtests, monkeypatch, qtbot):
    """``map_barcodes`` (the DNA_RAIN_APPS screen) takes the rain the installer
    returns, and an installer that raises leaves ``_dna_rain`` explicitly None
    rather than costing the module."""
    from spacr.qt.widgets import dna_rain

    assert "map_barcodes" in aps.DNA_RAIN_APPS
    installed = object()
    for name, installer, expected in (
            ("installer answers", lambda *a, **k: installed, installed),
            ("installer raises", _boom, None)):
        with subtests.test(case=name), monkeypatch.context() as mp:
            mp.setattr(dna_rain, "install_dna_rain", installer)
            screen = _built(qtbot, "map_barcodes")
            assert screen._dna_rain is expected


# -- removing and re-theming backdrops on a screen being taken apart ---------

def _remove_refusing_backdrop(screen, mp, qtbot):
    widget = types.SimpleNamespace(set_animating=_boom)
    discarded = []
    mp.setattr(aps, "_discard_widget", discarded.append)
    mp.setattr(screen, "_ambient", widget)
    screen._remove_ambient()
    assert discarded == [widget]
    assert screen._ambient is not widget


def _retheme_unresolved_theme(screen, mp, qtbot):
    from spacr.qt import preferences

    pushed = []
    mp.setattr(screen, "_ambient", types.SimpleNamespace(
        set_background_color=pushed.append))
    mp.setattr(preferences, "resolve_effective_theme", _boom)
    screen._retheme_backdrops()
    assert pushed == []


def _retheme_one_refuses(screen, mp, qtbot):
    taken = []
    mp.setattr(screen, "_dna_rain", types.SimpleNamespace(
        set_background_color=_boom, set_backdrop=_boom))
    mp.setattr(screen, "_ambient", types.SimpleNamespace(
        set_background_color=taken.append))
    mp.setattr(screen, "_backdrop_applied", None, raising=False)
    screen._retheme_backdrops()
    assert len(taken) == 1


def _refresh_records_only_what_was_taken(screen, mp, qtbot):
    from spacr.qt import preferences

    def prefer(theme, palette):
        mp.setattr(preferences, "get_ambient_enabled", lambda: True)
        mp.setattr(preferences, "get_ambient_theme", lambda: theme)
        mp.setattr(preferences, "get_ambient_palette", lambda: palette)

    assert aps.uses_ambient_background(screen.app_key)
    prefer("blobs", "cool")
    applied = []
    mp.setattr(screen, "_ambient", types.SimpleNamespace(
        set_theme=applied.append, set_palette=applied.append))
    mp.setattr(screen, "_ambient_applied", None, raising=False)
    screen.refresh_ambient_background()
    assert applied == ["blobs", "cool"]
    assert screen._ambient_applied == ("blobs", "cool")

    prefer("rain", "warm")
    mp.setattr(screen, "_ambient", types.SimpleNamespace(
        set_theme=_boom, set_palette=_boom))
    screen.refresh_ambient_background()
    assert screen._ambient_applied == ("blobs", "cool")


def _orphan_backdrop_taken_away(screen, mp, qtbot):
    from spacr.qt.widgets.ambient import AmbientWidget

    orphan = AmbientWidget(screen)
    qtbot.addWidget(orphan)
    mp.setattr(orphan, "set_animating", _boom)
    discarded = []
    mp.setattr(aps, "_discard_widget", discarded.append)
    screen._discard_orphan_ambient()
    assert orphan in discarded


def test_backdrops_are_rethemed_and_removed_whatever_refuses(
        subtests, monkeypatch, qtbot, regression_screen):
    """A backdrop that will not stop animating is still discarded (owned or
    orphaned), an unresolvable theme leaves the current fill alone, one backdrop
    refusing the fill does not stop the other, and a theme the backdrop refused
    is never recorded as applied."""
    cases = {
        "remove refusing backdrop": _remove_refusing_backdrop,
        "retheme unresolved theme": _retheme_unresolved_theme,
        "retheme one refuses": _retheme_one_refuses,
        "refresh records only accepted theme": _refresh_records_only_what_was_taken,
        # Last: it leaves an extra child widget on the screen.
        "orphan backdrop": _orphan_backdrop_taken_away,
    }
    completed = []
    for name, case in cases.items():
        with subtests.test(case=name), monkeypatch.context() as mp:
            case(regression_screen, mp, qtbot)
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


# -- the page's own fill -----------------------------------------------------

PAGE_FILL_CASES = {
    # name: (ambient, theme resolver, expect a valid colour)
    "animating backdrop": (object(), None, False),
    "image theme": (None, lambda: "cell", False),
    "flat theme": (None, lambda: "dark", True),
    "unresolvable theme": (None, _boom, False),
}


def test_the_page_fill_is_a_colour_only_for_a_flat_theme(
        subtests, monkeypatch, regression_screen):
    """``page_fill`` is None when something animates behind the page, when an
    image theme should show through, when the theme cannot be resolved or the
    module was loaded outside its package; only a flat theme paints a colour."""
    from PySide6.QtWidgets import QApplication
    from spacr.qt import preferences

    screen = regression_screen
    for name, (ambient, resolver, coloured) in PAGE_FILL_CASES.items():
        with subtests.test(case=name), monkeypatch.context() as mp:
            mp.setattr(screen, "_ambient", ambient)
            mp.setattr(screen, "_dna_rain", None)
            if resolver is not None:
                mp.setattr(preferences, "resolve_effective_theme", resolver)
            colour = screen.page_fill()
            if coloured:
                assert colour is not None and colour.isValid()
            else:
                assert colour is None
    with subtests.test(case="outside its package"), monkeypatch.context() as mp:
        mp.setattr(screen, "_ambient", None)
        mp.setattr(screen, "_dna_rain", None)
        with monkeypatch.context() as broken:
            broken.setattr(aps, "__package__", "not_a_package.qt.screens")
            broken.setattr(aps, "__spec__", None)
            assert screen.page_fill() is None
            assert screen._page_fill_for_a_frame() is None
            screen._sync_page_palette()
            screen.close()
        QApplication.processEvents()
        assert screen.isVisible() is False
