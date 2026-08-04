"""The Animation preference's ``None`` entry, and what it costs: nothing.

"No animation" is easy to claim and easy to get wrong. The three failure
modes that look identical from the outside are an engine that paints an
empty frame, a widget that is hidden but still ticking, and a timer that
was never stopped because a flag said it did not need to be — all of them
report "off" to anything that asks a boolean, and all of them keep waking
a laptop's CPU sixty times a second.

So the claim is measured the way the activity spinner's is: by counting
painted frames over a real window of time (see
``test_activity_spinner.py::test_an_idle_spinner_paints_nothing_at_all``),
with a control case in the same test proving the counter would have moved
if anything had painted.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings


IDLE_WINDOW_MS = 700


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    """A real QSettings backed by a temp INI, never the developer's own."""
    from spacr.qt import preferences as prefs
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    from spacr.qt.theme import apply_qpalette, stylesheet
    apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(stylesheet())
    return store


def _ambient():
    return pytest.importorskip("spacr.qt.widgets.ambient")


def _screen(qtbot, app_key="measure"):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    screen.resize(600, 400)
    return screen


# ---------------------------------------------------------------------------
# The choice itself
# ---------------------------------------------------------------------------

def test_none_is_offered_alongside_the_six_animations():
    ambient = _ambient()
    assert ambient.NO_ANIMATION == "none"
    assert ambient.ANIMATION_CHOICES == (ambient.NO_ANIMATION,) + \
        ("blobs", "aurora", "ripple", "drift", "bokeh", "cells")
    assert ambient.AMBIENT_THEMES == ("blobs", "aurora", "ripple", "drift",
                                      "bokeh", "cells"), (
        "None must not join the paintable themes: make_engine, "
        "_require_theme and every engine test mean 'can be drawn' by it")


def test_none_has_a_label_and_a_note_that_states_the_cost():
    ambient = _ambient()
    assert ambient.animation_label(ambient.NO_ANIMATION) == "None"
    note = ambient.animation_note(ambient.NO_ANIMATION)
    assert "no animation timer" in note.lower()
    for key in ambient.AMBIENT_THEMES:
        assert ambient.animation_label(key) == ambient.theme_label(key)
        assert ambient.animation_note(key) == ambient.theme_note(key)


def test_none_is_not_a_paintable_theme():
    ambient = _ambient()
    assert ambient.is_animation_choice(ambient.NO_ANIMATION)
    assert not ambient.is_valid_theme(ambient.NO_ANIMATION)
    with pytest.raises(ValueError):
        ambient.make_engine(ambient.NO_ANIMATION, "spacr", "#000000")


def test_the_preference_round_trips_none():
    from spacr.qt import preferences as prefs
    ambient = _ambient()

    prefs.set_ambient_animation("cells")
    assert prefs.get_ambient_animation() == "cells"
    assert prefs.get_ambient_enabled() is True

    prefs.set_ambient_animation(ambient.NO_ANIMATION)
    assert prefs.get_ambient_animation() == ambient.NO_ANIMATION
    assert prefs.get_ambient_enabled() is False
    # And the getter callers hand to make_engine still answers something
    # paintable, so a caller that ignores `get_ambient_enabled` fails
    # safe rather than raising deep inside a widget constructor.
    assert prefs.get_ambient_theme() in ambient.AMBIENT_THEMES

    prefs.set_ambient_animation("cells")
    assert prefs.get_ambient_enabled() is True


def test_choosing_none_does_not_lose_the_animation_you_had():
    from spacr.qt import preferences as prefs
    ambient = _ambient()
    prefs.set_ambient_animation("bokeh")
    palette = prefs.get_ambient_palette()
    prefs.set_ambient_animation(ambient.NO_ANIMATION)
    prefs.set_ambient_animation("bokeh")
    assert prefs.get_ambient_palette() == palette


def test_an_unknown_animation_is_refused_rather_than_stored():
    from spacr.qt import preferences as prefs
    with pytest.raises(ValueError):
        prefs.set_ambient_animation("nonexistent")


# ---------------------------------------------------------------------------
# The cost, counted rather than asserted from a flag
# ---------------------------------------------------------------------------

def test_none_installs_no_widget_and_paints_no_frames(qtbot):
    """The claim, measured. Control case first, so the counter is trusted.

    Both halves matter. The control proves an animation really does paint
    frames through this path, which is what makes the zero in the second
    half evidence rather than an artefact of a screen nobody showed.
    """
    from spacr.qt import preferences as prefs
    ambient = _ambient()

    # -- control: a real animation paints, and the counter sees it.
    prefs.set_ambient_animation("blobs")
    lively = _screen(qtbot)
    lively.show()
    qtbot.waitExposed(lively)
    qtbot.wait(120)
    assert isinstance(lively._ambient, ambient.AmbientWidget)
    assert lively._ambient.is_running()
    before = ambient.total_frames_painted()
    qtbot.wait(IDLE_WINDOW_MS)
    painted = ambient.total_frames_painted() - before
    assert painted > 0, "the control case painted nothing; the counter is dead"
    # Hidden, not deleted: qtbot owns it until teardown, and a backdrop on
    # a hidden screen stops ticking, which is what makes the second half's
    # count a statement about None rather than about visibility.
    lively.hide()
    qtbot.wait(50)

    # -- and with None: nothing is constructed, so nothing can tick.
    prefs.set_ambient_animation(ambient.NO_ANIMATION)
    quiet = _screen(qtbot)
    quiet.show()
    qtbot.waitExposed(quiet)
    qtbot.wait(120)
    assert quiet._ambient is None, (
        "None built a backdrop widget; not building it is the whole point")
    assert not any(isinstance(child, ambient.AmbientWidget)
                   for child in quiet.children()), (
        "an ambient widget was parented to the screen anyway")

    before = ambient.total_frames_painted()
    qtbot.wait(IDLE_WINDOW_MS)
    assert ambient.total_frames_painted() == before, (
        f"{ambient.total_frames_painted() - before} ambient frames were "
        f"painted in {IDLE_WINDOW_MS} ms with the animation set to None")


def test_no_ambient_timer_is_left_running_anywhere_under_none(qtbot):
    """Not one timer in the process, not just not on this screen.

    A backdrop on a screen the user is not looking at is exactly the thing
    that would keep ticking unnoticed, so the sweep is over every widget
    the application owns rather than over the one in hand.
    """
    from PySide6.QtWidgets import QApplication
    from spacr.qt import preferences as prefs
    ambient = _ambient()

    prefs.set_ambient_animation("aurora")
    screens = [_screen(qtbot, key) for key in ("measure", "classify")]
    for screen in screens:
        screen.show()
        qtbot.waitExposed(screen)
    qtbot.wait(120)
    assert any(isinstance(w, ambient.AmbientWidget) and w.is_running()
               for w in QApplication.instance().allWidgets())

    prefs.set_ambient_animation(ambient.NO_ANIMATION)
    prefs.apply_ambient_preferences()
    for screen in screens:
        screen.refresh_ambient_background()
    qtbot.wait(50)

    still = [w for w in QApplication.instance().allWidgets()
             if isinstance(w, ambient.AmbientWidget) and w.is_running()]
    assert not still, f"{len(still)} ambient timer(s) still ticking under None"

    before = ambient.total_frames_painted()
    qtbot.wait(IDLE_WINDOW_MS)
    assert ambient.total_frames_painted() == before


def test_turning_it_back_on_gives_the_backdrop_back(qtbot):
    """The off switch has to be reversible, or it is a trap."""
    from spacr.qt import preferences as prefs
    ambient = _ambient()

    prefs.set_ambient_animation(ambient.NO_ANIMATION)
    screen = _screen(qtbot)
    screen.show()
    qtbot.waitExposed(screen)
    assert screen._ambient is None

    prefs.set_ambient_animation("ripple")
    screen.refresh_ambient_background()
    qtbot.wait(120)
    assert isinstance(screen._ambient, ambient.AmbientWidget)
    assert screen._ambient.is_running()

    before = ambient.total_frames_painted()
    qtbot.wait(300)
    assert ambient.total_frames_painted() > before


def test_the_frame_counter_counts_this_widgets_own_frames(qtbot):
    """The per-widget counter, since the process-wide one is a sum."""
    ambient = _ambient()
    widget = ambient.AmbientWidget(theme="blobs", palette="spacr")
    qtbot.addWidget(widget)
    widget.resize(200, 150)
    assert widget.frames_painted == 0
    widget.show()
    qtbot.waitExposed(widget)
    qtbot.wait(250)
    assert widget.frames_painted > 0

    widget.set_animating(False)
    qtbot.wait(50)
    before = widget.frames_painted
    qtbot.wait(IDLE_WINDOW_MS)
    assert widget.frames_painted == before, "a paused backdrop kept painting"
