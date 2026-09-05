"""378 -- hold Z, turn the wheel, and the text resizes while you turn it.

The request carried a condition -- "only if possible to do fast without
lag" -- and the measurement that answered it is what these tests pin.

Two of them are measurements of **Qt**, not of spaCR, and they are here on
purpose. The instruction's proposed design was to rewrite the stylesheet's
49 ``font-size`` declarations in ``em`` so a single
``QApplication.setFont`` moved the whole interface. Qt does not implement a
relative unit for ``font-size``, so that design does not exist, and the one
that replaced it rests on the other fact: an explicit ``QWidget.setFont``
beats the application sheet's pixel size. If either ever stops being true,
these two fail and the design is revisited rather than quietly wrong.

Everything else is measured as rendered geometry -- ``QFontMetrics`` on the
widget's own resolved font -- because a scale stored in an attribute proves
only that a number was written down.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QFont, QFontMetrics, QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QVBoxLayout, QWidget

from spacr.qt import live_zoom
from spacr.qt import preferences as prefs

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _leave_the_application_as_we_found_it(qapp):
    """Repolish the shared QApplication once this file is done with it.

    The settle hands every font it borrowed back to the stylesheet, and
    these tests stub out the rebuild that would re-resolve it -- so the last
    gesture of the file leaves the process's long-lived widgets inheriting
    rather than styled. One rebuild at the end costs a second and puts them
    back; one per test would cost twenty.
    """
    yield
    from spacr.qt.theme import stylesheet
    qapp.setStyleSheet(stylesheet())
    QApplication.processEvents()


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never write the developer's real font scale. 378 persists on release."""
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write real preferences")
    prefs.set_font_scale(1.0)
    return store


@pytest.fixture(autouse=True)
def _never_restyle_the_shared_application(monkeypatch):
    """Keep the settle's stylesheet rebuild out of the shared QApplication.

    ``apply_preferences_to_app`` restyles every live widget in the process,
    which for a session-scoped ``qapp`` means every later test measuring a
    pixel. The tests that care about the settle assert on what it *asks*
    for; one of them un-patches this deliberately.
    """
    applied = []
    monkeypatch.setattr(prefs, "apply_preferences_to_app",
                        lambda app=None: applied.append(app))
    return applied


@pytest.fixture
def zoom(qt_theme_applied):
    """A filter that always ends its gesture, whatever the test did.

    A live gesture holds every visible widget in the process, this one
    included -- leaving one running would hand the next test a screen full
    of fonts this one had grown.
    """
    live = live_zoom.LiveZoomFilter()
    yield live
    live.settle()


@pytest.fixture
def screen(qt_theme_applied):
    """A shown widget tree styled by the real application stylesheet."""
    root = QWidget()
    layout = QVBoxLayout(root)
    title = QLabel("Plate 3")
    title.setObjectName("ScreenTitle")
    caption = QLabel("well A01")
    field = QLineEdit("nuclei")
    for widget in (title, caption, field):
        layout.addWidget(widget)
    root.show()
    QApplication.processEvents()
    yield root, title, caption, field
    root.hide()
    root.deleteLater()
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key(kind, key=Qt.Key_Z, modifiers=Qt.NoModifier, autorepeat=False):
    event = QKeyEvent(kind, key, modifiers, "z", autorepeat)
    return event


def _wheel(notches: int = 1, modifiers=Qt.NoModifier):
    """One wheel event carrying ``notches`` detents, Qt's own 120ths."""
    return QWheelEvent(
        QPointF(10, 10), QPointF(10, 10), QPoint(0, 0),
        QPoint(0, int(notches * 120)), Qt.NoButton, modifiers,
        Qt.NoScrollPhase, False)


def _hold(zoom, widget):
    """Press Z the way the user does, and confirm the gesture armed."""
    assert zoom.eventFilter(widget, _key(QEvent.KeyPress)) is False, (
        "the gesture must never swallow the key itself -- Z is a letter")
    assert zoom._held


def _height(widget) -> int:
    """Rendered line height of the widget's resolved font, in pixels."""
    return QFontMetrics(widget.font()).height()


# ---------------------------------------------------------------------------
# The two measurements the design rests on
# ---------------------------------------------------------------------------

def test_qt_still_has_no_relative_font_size(qapp):
    """``em`` and ``%`` are dropped, so the sheet cannot follow one setFont.

    This is the measurement that killed part 2 of instruction 378. Qt's CSS
    parser accepts ``pt``, ``px`` and the size keywords for ``font-size``
    and silently discards anything else, so a sheet written in ``em`` styles
    nothing at all -- which is worse than the slow path, not faster.

    The day this fails, Qt has gained the unit and the far simpler design
    (one ``QApplication.setFont``, 14 ms, no per-widget pass) is available.
    """
    previous = qapp.styleSheet()
    root = QWidget()
    layout = QVBoxLayout(root)
    relative, percent, absolute = QLabel("a"), QLabel("b"), QLabel("c")
    relative.setObjectName("Em")
    percent.setObjectName("Percent")
    absolute.setObjectName("Px")
    for label in (relative, percent, absolute):
        layout.addWidget(label)
    try:
        qapp.setStyleSheet(
            "QWidget { font-size: 13px; }"
            "QLabel#Em { font-size: 2em; }"
            "QLabel#Percent { font-size: 200%; }"
            "QLabel#Px { font-size: 26px; }")
        root.show()
        QApplication.processEvents()

        assert absolute.font().pixelSize() == 26, "px is the unit that works"
        assert relative.font().pixelSize() == 13, "em was applied after all"
        assert percent.font().pixelSize() == 13, "% was applied after all"
    finally:
        root.hide()
        root.deleteLater()
        qapp.setStyleSheet(previous)
        QApplication.processEvents()


def test_an_explicit_font_beats_the_sheets_pixel_size(screen):
    """The escape hatch the live pass uses, and the reason it is per-widget.

    A QSS ``font-size`` outranks the inherited application font, which is
    why ``QApplication.setFont`` moves nothing. It does NOT outrank a font
    set on the widget itself, so the live pass can move a widget the sheet
    has pinned without touching the sheet.
    """
    _root, title, _caption, _field = screen
    pinned = title.font().pixelSize()
    assert pinned > 0, "the stylesheet did not reach the label"

    font = QFont(title.font())
    font.setPixelSize(pinned * 2)
    title.setFont(font)
    QApplication.processEvents()

    assert title.font().pixelSize() == pinned * 2
    assert QFontMetrics(title.font()).height() > pinned


# ---------------------------------------------------------------------------
# The gesture
# ---------------------------------------------------------------------------

def test_a_notch_grows_the_rendered_text(zoom, screen):
    """Measured on the widget's own font, not on the number that was stored."""
    root, title, caption, _field = screen
    before = {w: _height(w) for w in (title, caption)}

    _hold(zoom, root)
    for _ in range(4):                       # 4 x 5 % -- past every rounding
        zoom.eventFilter(root, _wheel(+1))
    QApplication.processEvents()

    for widget, was in before.items():
        assert _height(widget) > was, (
            "the wheel did not reach a widget the stylesheet had pinned")


def test_the_roles_keep_their_proportion(zoom, screen):
    """A title stays bigger than a caption: each is scaled from its OWN font.

    Scaling everything to one size would be simpler and would flatten the
    typography into a single ransom-note size the moment the wheel moved.
    """
    root, title, caption, _field = screen
    font = QFont(title.font())
    font.setPixelSize(caption.font().pixelSize() * 2)
    title.setFont(font)
    QApplication.processEvents()

    _hold(zoom, root)
    for _ in range(6):
        zoom.eventFilter(root, _wheel(+1))
    QApplication.processEvents()

    assert title.font().pixelSize() > caption.font().pixelSize()
    ratio = title.font().pixelSize() / caption.font().pixelSize()
    assert 1.8 < ratio < 2.2, f"the role hierarchy drifted to {ratio:.2f}"


def test_the_text_shrinks_again_on_the_way_back(zoom, screen):
    """Down is not a separate path, and it returns to where it started."""
    root, title, _caption, _field = screen
    _hold(zoom, root)
    for _ in range(4):
        zoom.eventFilter(root, _wheel(+1))
    grown = title.font().pixelSize()
    for _ in range(4):
        zoom.eventFilter(root, _wheel(-1))
    QApplication.processEvents()

    assert grown > title.font().pixelSize()
    assert zoom._live_scale == pytest.approx(1.0)


def test_a_notch_is_the_sliders_own_step(zoom, screen):
    """5 % a notch, so the gesture and the Preferences slider agree."""
    root, *_ = screen
    _hold(zoom, root)
    zoom.eventFilter(root, _wheel(+1))
    assert zoom._live_scale == pytest.approx(1.0 + live_zoom.FONT_SCALE_STEP)
    zoom.eventFilter(root, _wheel(-2))
    assert zoom._live_scale == pytest.approx(1.0 - live_zoom.FONT_SCALE_STEP)


def test_the_scale_is_clamped_to_the_preference_bounds(zoom, screen):
    """The same bounds the slider has; the wheel cannot reach past them."""
    root, *_ = screen
    _hold(zoom, root)
    for _ in range(60):
        zoom.eventFilter(root, _wheel(+1))
    assert zoom._live_scale == pytest.approx(prefs.FONT_SCALE_MAX)
    for _ in range(120):
        zoom.eventFilter(root, _wheel(-1))
    assert zoom._live_scale == pytest.approx(prefs.FONT_SCALE_MIN)


def test_the_wheel_is_taken_from_the_list_underneath(zoom, screen):
    """Accepted and consumed, or the settings list scrolls as the text grows."""
    root, *_ = screen
    _hold(zoom, root)
    event = _wheel(+1)
    assert zoom.eventFilter(root, event) is True
    assert event.isAccepted()


def test_a_wheel_at_the_bound_is_still_consumed(zoom, screen):
    """Reaching 200 % must not hand the wheel back to a scroll area.

    Otherwise the list under the pointer starts scrolling mid-gesture, at
    the exact moment the text stops responding, which reads as the gesture
    breaking rather than as a limit.
    """
    root, *_ = screen
    _hold(zoom, root)
    for _ in range(60):
        zoom.eventFilter(root, _wheel(+1))
    event = _wheel(+1)
    assert zoom.eventFilter(root, event) is True


def test_the_wheel_is_left_alone_when_z_is_not_held(zoom, screen):
    """Ctrl+wheel is canvas zoom and a bare wheel scrolls; both stay theirs."""
    root, title, *_ = screen
    was = _height(title)

    assert zoom.eventFilter(root, _wheel(+1)) is False
    assert zoom.eventFilter(root, _wheel(+1, Qt.ControlModifier)) is False
    QApplication.processEvents()

    assert _height(title) == was
    assert zoom._baseline == []


def test_ctrl_z_does_not_arm_the_gesture(zoom, screen):
    """Ctrl+Z is undo. A Z carrying a command modifier is somebody else's."""
    root, *_ = screen
    zoom.eventFilter(root, _key(QEvent.KeyPress, modifiers=Qt.ControlModifier))
    assert not zoom._held
    assert zoom.eventFilter(root, _wheel(+1)) is False


def test_another_letter_does_not_arm_the_gesture(zoom, screen):
    root, *_ = screen
    zoom.eventFilter(root, _key(QEvent.KeyPress, key=Qt.Key_X))
    assert not zoom._held


def test_the_gesture_works_with_the_focus_in_a_text_field(zoom, screen):
    """A settings form is mostly fields, and one of them always has focus.

    The first version of this refused to arm while a text field had the
    focus, on the theory that somebody typing a plate name is pressing the
    same key. That disabled the gesture on exactly the screens it is for.
    The key is never consumed, so the letter is still typed either way, and
    a misfire needs the user to hold Z down AND turn the wheel.
    """
    root, title, _caption, field = screen
    field.setFocus()
    QApplication.processEvents()
    was = _height(title)

    _hold(zoom, field)
    for _ in range(4):
        zoom.eventFilter(field, _wheel(+1))
    QApplication.processEvents()

    assert _height(title) > was


def test_an_autorepeat_release_does_not_end_the_gesture(zoom, screen):
    """X11 sends a release/press PAIR for every auto-repeat tick.

    Trusting the first KeyRelease disarms the gesture a few hundred
    milliseconds into the hold, and the rest of the scroll goes to the list.
    """
    root, *_ = screen
    _hold(zoom, root)
    zoom.eventFilter(root, _key(QEvent.KeyRelease, autorepeat=True))
    assert zoom._held
    zoom.eventFilter(root, _key(QEvent.KeyPress, autorepeat=True))
    assert zoom._held

    zoom.eventFilter(root, _key(QEvent.KeyRelease))
    assert not zoom._held


def test_an_unrelated_event_costs_two_comparisons(zoom, screen, monkeypatch):
    """The filter sees every event in the process; it may not act on them.

    315's warning about application-wide filters is about cost. Nothing
    below the two type checks may run for a mouse move, which is the event
    the application delivers most of.
    """
    root, *_ = screen
    monkeypatch.setattr(
        zoom, "_begin",
        lambda *a: pytest.fail("a mouse move started a zoom gesture"))
    assert zoom.eventFilter(root, QEvent(QEvent.MouseMove)) is False
    _hold(zoom, root)
    assert zoom.eventFilter(root, QEvent(QEvent.MouseMove)) is False


# ---------------------------------------------------------------------------
# The settle -- the half that is deliberately not live
# ---------------------------------------------------------------------------

def test_nothing_is_written_to_settings_per_notch(zoom, screen, monkeypatch):
    """Twenty QSettings writes a second is not free. Persist on release."""
    root, *_ = screen
    writes = []
    monkeypatch.setattr(prefs, "set_font_scale",
                        lambda scale: writes.append(scale))

    _hold(zoom, root)
    for _ in range(5):
        zoom.eventFilter(root, _wheel(+1))
    assert writes == [], "the wheel wrote the preference while it turned"

    zoom.eventFilter(root, _key(QEvent.KeyRelease))
    assert writes == [pytest.approx(1.25)], "release did not persist once"


def test_the_release_rebuilds_the_spacing(zoom, screen,
                                          _never_restyle_the_shared_application):
    """"Text live, spacing on release" -- the compromise, made explicit.

    Row heights, column widths, icon and tile sizes all come from
    :func:`spacr.qt.preferences.scaled_px`, which only moves when the
    stylesheet is rebuilt. That rebuild is the 587 ms the gesture exists to
    avoid paying per notch, so it happens exactly once, here.
    """
    root, *_ = screen
    _hold(zoom, root)
    for _ in range(2):
        zoom.eventFilter(root, _wheel(+1))
    assert _never_restyle_the_shared_application == [], (
        "the expensive rebuild ran while the wheel was still turning")

    zoom.eventFilter(root, _key(QEvent.KeyRelease))

    assert len(_never_restyle_the_shared_application) == 1
    assert prefs.get_font_scale() == pytest.approx(1.10)


def test_the_wheel_going_quiet_settles_without_the_key_coming_up(zoom, screen):
    """The spacing catches up when the wheel stops, not only on release."""
    root, *_ = screen
    _hold(zoom, root)
    zoom.eventFilter(root, _wheel(+1))
    assert zoom._settle_timer.isActive()
    assert zoom._settle_timer.interval() == live_zoom._SETTLE_MS

    zoom._settle_timer.timeout.emit()
    assert zoom._baseline == []
    assert prefs.get_font_scale() == pytest.approx(1.05)


def test_a_pause_mid_gesture_does_not_disarm_the_key(zoom, screen):
    """The wheel going quiet ends the gesture; it does not end the hold.

    A user reads the new size, decides it is not enough, and scrolls again
    without ever letting go of Z. Clearing the hold on the idle timer would
    send that second scroll to the list under the pointer.
    """
    root, title, *_ = screen
    _hold(zoom, root)
    zoom.eventFilter(root, _wheel(+1))
    zoom.settle(released=False)
    assert zoom._held

    was = _height(title)
    for _ in range(4):
        zoom.eventFilter(root, _wheel(+1))
    QApplication.processEvents()

    assert _height(title) > was, "the second half of the gesture was lost"
    assert zoom._live_scale == pytest.approx(1.25), (
        "the second gesture did not resume from the scale the first saved")


def test_leaving_the_window_settles_a_held_gesture(zoom, screen):
    """Alt-tab delivers the KeyRelease to somebody else's application."""
    root, *_ = screen
    _hold(zoom, root)
    zoom.eventFilter(root, _wheel(+1))
    zoom.eventFilter(root, QEvent(QEvent.WindowDeactivate))

    assert not zoom._held
    assert prefs.get_font_scale() == pytest.approx(1.05)


def test_a_gesture_that_moved_nothing_writes_nothing(zoom, screen,
                                                     monkeypatch):
    """Z pressed and released with no wheel is not a preference change."""
    root, *_ = screen
    writes = []
    monkeypatch.setattr(prefs, "set_font_scale",
                        lambda scale: writes.append(scale))
    _hold(zoom, root)
    zoom.eventFilter(root, _key(QEvent.KeyRelease))
    assert writes == []


def test_the_saved_scale_survives_being_read_back_as_a_percent(zoom, screen):
    """Preferences reads the scale with ``int(scale * 100)``, which truncates.

    Four notches down from 1.0 is 0.7999999999999998 in binary floating
    point, and the slider then opens on 79 % -- the gesture and the control
    disagreeing by a percent for no reason the user could discover.
    """
    root, *_ = screen
    _hold(zoom, root)
    for _ in range(4):
        zoom.eventFilter(root, _wheel(-1))
    zoom.eventFilter(root, _key(QEvent.KeyRelease))

    assert int(prefs.get_font_scale() * 100) == 80


def test_the_settle_hands_the_font_back_to_the_stylesheet(zoom, screen):
    """A widget the sheet dressed must not end the gesture wearing its own.

    ``setFont`` sets ``WA_SetFont``, and a widget carrying that attribute
    keeps its font against everything the next stylesheet says unless a rule
    names it. Leaving it set would pin whatever the gesture last rendered
    for the rest of the session -- the gesture's own size outliving the
    preference it was setting.
    """
    _root, title, *_ = screen
    assert not title.testAttribute(Qt.WA_SetFont), (
        "the stylesheet's font is not the widget's own; the test needs one "
        "that is not")

    _hold(zoom, _root)
    for _ in range(4):
        zoom.eventFilter(_root, _wheel(+1))
    assert title.testAttribute(Qt.WA_SetFont), "the live pass did not run"

    zoom.eventFilter(_root, _key(QEvent.KeyRelease))
    assert not title.testAttribute(Qt.WA_SetFont)


def test_the_settle_gives_back_a_font_the_widget_really_owned(zoom, screen):
    """The console's monospace and the AI toggle's size are not the sheet's.

    Those widgets set their own font on purpose, so the settle has to return
    that font rather than clearing it and letting the sheet take over.
    """
    _root, _title, caption, _field = screen
    mine = QFont("Courier")
    mine.setPixelSize(9)
    caption.setFont(mine)
    QApplication.processEvents()

    _hold(zoom, _root)
    for _ in range(4):
        zoom.eventFilter(_root, _wheel(+1))
    zoom.eventFilter(_root, _key(QEvent.KeyRelease))

    assert caption.testAttribute(Qt.WA_SetFont)
    assert caption.font().family() == "Courier"
    assert caption.font().pixelSize() == 9


def test_a_widget_deleted_mid_gesture_does_not_end_it(zoom, screen):
    """A dialog closed while Z is down is ordinary, not exceptional."""
    root, title, caption, _field = screen
    _hold(zoom, root)
    zoom.eventFilter(root, _wheel(+1))

    import shiboken6
    shiboken6.delete(caption)

    zoom.eventFilter(root, _wheel(+1))
    zoom.eventFilter(root, _key(QEvent.KeyRelease))
    assert prefs.get_font_scale() == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def test_installing_twice_leaves_one_filter(qapp):
    """Every event in the process pays for each filter on the list."""
    previous = getattr(qapp, live_zoom._FILTER_ATTRIBUTE, None)
    try:
        if previous is not None:
            qapp.removeEventFilter(previous)
            delattr(qapp, live_zoom._FILTER_ATTRIBUTE)
        first = live_zoom.install_live_zoom(qapp)
        second = live_zoom.install_live_zoom(qapp)
        assert first is second
    finally:
        installed = getattr(qapp, live_zoom._FILTER_ATTRIBUTE, None)
        if installed is not None and installed is not previous:
            qapp.removeEventFilter(installed)
            delattr(qapp, live_zoom._FILTER_ATTRIBUTE)
        if previous is not None:
            setattr(qapp, live_zoom._FILTER_ATTRIBUTE, previous)
            qapp.installEventFilter(previous)


def test_the_launch_installs_the_gesture():
    """An application-wide gesture that nothing installs works nowhere."""
    import inspect

    from spacr.qt import app as qt_app

    source = inspect.getsource(qt_app.launch)
    assert "install_live_zoom(app)" in source


def test_a_round_trip_still_restyles(qtbot, qt_theme_applied, monkeypatch):
    """Scroll up, scroll down, release -- and the repair still happens.

    THE REGRESSION THIS PINS. `settle()` puts a QSS-dressed widget back by
    clearing its font with `setFont(QFont())`, which leaves it INHERITING
    rather than styled: the sheet's `font-size` does not come back until
    something re-polishes. It then returned early whenever the wheel had
    ended on the scale it began at -- on the reasoning that there was
    nothing to persist, which is true, and nothing to repair, which is not.

    Measured before the fix: one notch up, one notch down, release, and the
    visible widgets went from 13 px to no styled size at all, with nothing
    scheduled to put it back.

    Asserted on the CALL rather than on a font size, because the repair is a
    re-polish that Qt performs on its own schedule -- a size read straight
    afterwards is testing Qt's timing, not this code. What must be true here
    is that the re-polish is asked for.
    """
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    from spacr.qt import live_zoom as module

    applied = []
    monkeypatch.setattr(module, "LOG", module.LOG)
    real = __import__("spacr.qt.preferences", fromlist=["x"])
    monkeypatch.setattr(real, "apply_preferences_to_app",
                        lambda app: applied.append(app))

    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    layout.addWidget(QLabel("hello"))
    host.show()

    zoom = module.LiveZoomFilter()
    zoom._held = True
    zoom._begin(host)
    zoom._live_scale = zoom._base_scale * 1.10
    zoom._apply()
    zoom._live_scale = zoom._base_scale        # back where it started
    zoom._apply()
    zoom.settle(released=True)

    assert applied, (
        "settle() returned without re-styling because the scale was "
        "unchanged -- every widget it cleared is left with no styled font")
