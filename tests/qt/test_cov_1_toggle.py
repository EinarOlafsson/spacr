"""A switch only answers to the left button, and never strands its knob.

``Toggle`` takes over the mouse handling a QCheckBox would normally do, so
every event it declines to handle has to be handed back to Qt rather than
swallowed. The cases below are the ones a user produces by accident: a
right-click, a pointer passing over the switch, a release that belongs to a
press somewhere else, and a drag that lets go on the side it started from.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.qt.widgets.toggle import Toggle


@contextmanager
def _at_design_scale():
    """Keep saved and active zoom local to designed-coordinate checks."""
    from spacr.qt import gui_scale, preferences

    settings = preferences._settings()
    keys = ("prefs/font_scale", "prefs/gui_scale")
    previous = {key: settings.value(key) for key in keys
                if settings.contains(key)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", 1.0)
        try:
            preferences.set_font_scale(1.0)
            preferences.set_gui_scale(1.0)
            yield
        finally:
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)


@pytest.fixture(autouse=True)
def _geometry_at_design_scale(qapp):
    with _at_design_scale():
        yield


@pytest.fixture
def toggle(qtbot):
    """A shown switch, wide enough for the track and a drag."""
    widget = Toggle()
    qtbot.addWidget(widget)
    widget.resize(60, 24)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


def _knob_centre(widget, x):
    return QPoint(int(x), widget.height() // 2)


def test_a_right_click_does_not_flip_the_switch(toggle):
    """The right button belongs to the context menu, not to the switch."""
    start = toggle.isChecked()

    QTest.mouseClick(toggle, Qt.RightButton,
                     pos=_knob_centre(toggle, toggle._minimum_knob_x() + 6))

    assert toggle.isChecked() is start
    assert toggle._mouse_pressed is False, (
        "a right press must not arm the drag state the left button uses")


def test_a_pointer_passing_over_the_switch_does_not_drag_the_knob(toggle):
    """A move with no button held is a hover, and hovering moves nothing."""
    resting = toggle._knob_pos

    QTest.mouseMove(toggle, pos=_knob_centre(toggle, toggle._maximum_knob_x()))

    assert toggle._knob_pos == resting
    assert toggle._dragging is False


def test_a_release_from_a_press_elsewhere_is_ignored(toggle):
    """A left release the switch never saw the press for changes nothing.

    Releasing over a widget that was not the one pressed is ordinary mouse
    behaviour -- press on a neighbouring control, slide across, let go. It
    must not count as a tap on the switch.
    """
    start = toggle.isChecked()

    QTest.mouseRelease(toggle, Qt.LeftButton,
                       pos=_knob_centre(toggle, toggle._maximum_knob_x()))

    assert toggle.isChecked() is start
    assert toggle._knob_pos == float(toggle._minimum_knob_x())


def test_a_drag_that_lets_go_on_the_same_side_snaps_the_knob_back(toggle,
                                                                  qtbot):
    """A short drag that does not cross the middle returns the knob home.

    The state does not change, so nothing else would move the knob: without
    an explicit re-animation it would sit wherever the finger let go, showing
    a half-open switch for a setting that is still off.
    """
    assert toggle.isChecked() is False
    y = toggle.height() // 2
    start_x = toggle._minimum_knob_x() + toggle._knob_d // 2

    QTest.mousePress(toggle, Qt.LeftButton, pos=QPoint(start_x, y))
    QTest.mouseMove(toggle, pos=QPoint(start_x + 5, y))
    assert toggle._dragging is True
    assert toggle._knob_pos > toggle._minimum_knob_x(), (
        "the drag has to have moved the knob for the snap-back to mean "
        "anything")

    QTest.mouseRelease(toggle, Qt.LeftButton, pos=QPoint(start_x + 5, y))

    assert toggle.isChecked() is False
    qtbot.waitUntil(
        lambda: toggle._knob_pos == float(toggle._minimum_knob_x()),
        timeout=2000)


def test_a_disabled_switch_is_drawn_dimmer_than_a_live_one(qtbot):
    """A switch that cannot be changed says so by fading, label and all.

    The state colour and the track fill are both painted at reduced alpha when
    the widget is disabled. Drawn at full strength it would look like a
    setting the user can still reach.
    """
    live, dead = Toggle("Timelapse"), Toggle("Timelapse")
    for widget in (live, dead):
        qtbot.addWidget(widget)
        widget.resize(120, 24)
        widget.setChecked(True)
    dead.setEnabled(False)

    knob_x = live._maximum_knob_x() + live._knob_d // 2
    y = live.height() // 2
    live_pixel = live.grab().toImage().pixelColor(knob_x, y)
    dead_pixel = dead.grab().toImage().pixelColor(knob_x, y)

    assert live_pixel != dead_pixel, (
        "the disabled switch painted its knob at full strength")


def test_a_labelled_switch_asks_for_room_for_the_track_and_the_text(qtbot):
    """The size hint covers the switch as well as the words beside it.

    A QCheckBox sizes itself for its indicator, which this widget does not
    draw; without the track's own width the label would be laid out over the
    switch.
    """
    from PySide6.QtWidgets import QCheckBox

    toggle = Toggle("Timelapse")
    qtbot.addWidget(toggle)
    plain = QCheckBox("Timelapse")
    plain.setObjectName(toggle.objectName())
    plain.setStyleSheet(toggle.styleSheet())
    qtbot.addWidget(plain)

    assert toggle.sizeHint().width() >= (
        plain.sizeHint().width() + toggle._track_x + toggle._track_w
        + toggle._label_gap)
    assert toggle.sizeHint().height() >= toggle._track_h


def test_a_compact_native_font_does_not_shrink_the_switch_track_hint(qtbot):
    """A short native text hint must still reserve the painted track."""
    from PySide6.QtWidgets import QCheckBox

    toggle = Toggle("Timelapse")
    toggle.setStyleSheet(
        toggle.styleSheet()
        + "QCheckBox#SpacrToggle { font-size: 8px; }")
    qtbot.addWidget(toggle)
    toggle.ensurePolished()

    native_hint = QCheckBox.sizeHint(toggle)
    assert toggle.sizeHint().height() >= native_hint.height()
    assert toggle.sizeHint().height() >= toggle._track_h + 2


def test_preferences_modules_at_ninety_percent_reserve_the_painted_track(
        qtbot, qt_theme_applied, monkeypatch):
    from PySide6.QtWidgets import QApplication, QTabWidget, QWidget
    from spacr.qt import gui_scale, preferences

    gui_scale.install_scaling_layer()
    with monkeypatch.context() as patch:
        patch.setattr(gui_scale, '_SCALE', .9)
        preferences.set_gui_scale(.9)
        preferences.set_font_scale(.9)
        dialog = preferences.PreferencesDialog()
        qtbot.addWidget(dialog)
        tabs = dialog.findChild(QTabWidget, 'PreferencesTabs')
        page = dialog.findChild(QWidget, 'PreferencesTabModules')
        tabs.setCurrentIndex(tabs.indexOf(page.parentWidget().parentWidget()))
        dialog.resize(900, 650)
        dialog.show()
        QApplication.processEvents()
        captions = {'Enable verbose logging',
                    'Include redacted log excerpts in issue previews',
                    'Allow editing in the Database Browser'}
        toggles = [toggle for toggle in page.findChildren(Toggle) if toggle.text() in captions]
        assert len(toggles) == 3
        for toggle in toggles:
            assert gui_scale._actual_minimum_height(toggle) >= toggle._track_h + 2
            assert toggle.minimumSizeHint().height() >= toggle._track_h + 2
            top = (toggle.height() - toggle._track_h) // 2
            assert top >= .75
            assert top + toggle._track_h + .75 <= toggle.height()
            assert toggle.isVisible()


def test_existing_plain_and_wrapped_tracks_fit_across_live_scale_changes(
        qtbot, qt_theme_applied, monkeypatch):
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
    from spacr.qt import gui_scale, preferences

    gui_scale.install_scaling_layer()
    with monkeypatch.context() as patch:
        patch.setattr(gui_scale, '_SCALE', .9)
        preferences.set_gui_scale(.9)
        preferences.set_font_scale(.9)
        panel = QWidget()
        qtbot.addWidget(panel)
        layout = QVBoxLayout(panel)
        toggles = [Toggle('A switch caption', word_wrap=wrapped) for wrapped in (False, True)]
        for toggle in toggles:
            layout.addWidget(toggle)
        panel.show()
        for scale in (.9, .75, 2., .1, .9):
            gui_scale.set_gui_scale_live(scale)
            QApplication.processEvents()
            for toggle in toggles:
                floor = toggle._track_h + 2
                assert gui_scale._actual_minimum_height(toggle) >= floor
                assert gui_scale._actual_minimum_height(toggle) <= max(floor, gui_scale.scale_int(floor, scale))
                top = (toggle.height() - toggle._track_h) // 2
                assert top >= .75
                assert top + toggle._track_h + .75 <= toggle.height()
