"""Optional field waves use GUI popup centres and bounded worker-owned state."""
import math

import pytest
from PySide6.QtWidgets import QDialog, QDoubleSpinBox, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient
from tests.qt.test_preferences_apply import _answer, _apply, _dialog
from tests.qt.test_preferences_apply import private_preferences as private_preferences
from tests.qt.test_requested_appearance_defaults import empty_store as empty_store


def engine(frequency=0):
    made = ambient.make_engine('data_art_impulse_lens', 'spacr', '#101010',
                               seed=12, popup_wave_frequency=frequency)
    made.set_gravity_radius(0)
    return made


def test_zero_frequency_keeps_discrete_waves_without_recurring(qapp, empty_store):
    assert prefs._field_popup_wave_frequency() == 5
    prefs._set_field_popup_wave_frequency(0)
    made = engine()
    made._set_popup_wave_origin((0.3, 0.7))
    assert made._popup_waves == [(0, (0.3, 0.7))]
    made.advance(1)
    assert len(made._popup_waves) == 1
    made._set_popup_wave_origin(None)
    assert made._popup_waves[-1] == (made.time, (0.3, 0.7))
    made.advance(120)
    assert made._popup_waves == []
    made.set_popup_wave_frequency(60)
    made._set_popup_wave_origin((0.3, 0.7))
    made.advance(1)
    assert len(made._popup_waves) == 2
    active = list(made._popup_waves)
    made.set_popup_wave_frequency(0)
    assert made._popup_waves == active
    made.advance(1)
    assert made._popup_waves == active
    made.advance(120)
    assert made._popup_waves == []


def test_frequency_changes_keep_in_flight_click_and_container_feedback(qapp):
    made = engine(30)
    made._add_ripple((0.2, 0.3))
    made._set_popup_wave_origin((0.6, 0.7), popup_id=1)
    active = list(made._popup_waves)
    made.advance(0.5)
    made.set_popup_wave_frequency(0)
    assert made._popup_waves == active
    assert made._popup_wave_elapsed == 0
    made.advance(1)
    assert made._popup_waves == active
    made._set_popup_wave_origin(None)
    assert made._popup_waves[-1][1] == (0.6, 0.7)
    assert len(made._popup_waves) == 3


def test_frequency_tracks_elapsed_time_and_popup_centre_without_mouse_gravity(qapp):
    made = engine(30)
    made._set_popup_wave_origin((0.3, 0.7))
    assert made._popup_waves == [(0, (0.3, 0.7))]
    made.advance(1.9)
    assert len(made._popup_waves) == 1
    made.advance(0.1)
    assert len(made._popup_waves) == 2
    assert made._popup_waves[0][1] == (0.3, 0.7)
    made._set_popup_wave_origin((0.6, 0.4))
    made.advance(2)
    assert [row[1] for row in made._popup_waves] == [
        (0.3, 0.7), (0.3, 0.7), (0.6, 0.4)]
    assert made.gravity_radius == 0
    made._set_popup_wave_origin(None)
    made.advance(5)
    assert made._popup_waves == []
    made._set_popup_wave_origin((math.nan, 0.4))
    made.advance(5)
    assert made._popup_waves == []


def test_waves_change_owned_pixels_and_remain_bounded(qapp):
    reference = engine()
    made = engine(60)
    made._set_popup_wave_origin((0.4, 0.6))
    made.advance(1)
    made.advance(0.8)
    reference.advance(1.8)
    first = made.shade(640, 480)
    saved = bytes(first.constBits())
    assert saved != bytes(reference.shade(640, 480).constBits())
    for _ in range(100):
        made.advance(1)
        made.shade(640, 480)
        assert len(made._popup_waves) <= 6
    assert bytes(first.constBits()) == saved
    material = next(value for key, value in made._material_cache.items()
                    if key[0] == 'impulse_lens')
    assert len(material[3]) <= 6


def test_queue_retains_popup_origin_even_when_gravity_is_disabled(qapp):
    made = engine(60)
    queue = ambient._QueuedArtInput()
    queue._offer(1, None, (), popup_origin=(0.25, 0.75))
    queue._consume(made)
    assert len(made._popup_waves) == 2
    assert made._popup_waves[0][1] == (0.25, 0.75)
    queue._consume(made)
    assert len(made._popup_waves) == 2
    queue._consume(made, discard_clicks=True)
    assert made._popup_wave_origin is None


def test_each_new_popup_emits_once_and_moving_the_same_popup_does_not(qapp):
    made = engine(5)
    queue = ambient._QueuedArtInput()
    queue._offer(.04, None, (), popup_origin=(.25, .75), popup_id=1)
    queue._consume(made)
    assert made._popup_waves == [(0, (.25, .75))]
    queue._consume(made)
    queue._offer(.04, None, (), popup_origin=(.3, .7), popup_id=1)
    queue._consume(made)
    assert len(made._popup_waves) == 1
    queue._offer(.04, None, (), popup_origin=(.3, .7), popup_id=2)
    queue._consume(made)
    assert len(made._popup_waves) == 2
    queue._offer(.04, None, (), popup_origin=None)
    queue._consume(made)
    assert len(made._popup_waves) == 3
    queue._offer(.04, None, (), popup_origin=(.3, .7), popup_id=2)
    queue._consume(made)
    assert len(made._popup_waves) == 4


@pytest.mark.parametrize("enabled", [False, True])
def test_real_dialog_open_close_is_discrete_with_gravity_and_frequency_zero(
        qtbot, monkeypatch, enabled):
    monkeypatch.setattr(ambient.AmbientWidget, '_start_producer', lambda self: None)
    owner = QWidget()
    qtbot.addWidget(owner)
    owner.resize(400, 300)
    field = ambient.AmbientWidget(
        owner, theme='data_art_impulse_lens', palette='spacr',
        popup_wave_frequency=0, gravity_radius=0, ripples_enabled=enabled)
    field.setGeometry(owner.rect())
    owner.show()
    field.show()
    field._timer.stop()
    popup = QDialog(owner)
    qtbot.addWidget(popup)
    popup.resize(120, 80)
    popup.move(owner.mapToGlobal(owner.rect().center()) - popup.rect().center())
    popup.open()
    qtbot.waitUntil(lambda: ambient.QApplication.activeModalWidget() is popup)
    field._on_tick()
    field._timer.stop()
    assert len(field.engine._popup_waves) == int(enabled)
    if enabled:
        assert len(field.engine._popup_waves[0][1]) == 4
    first = list(field.engine._popup_waves)
    field.engine.advance(1)
    field._on_tick()
    assert field.engine._popup_waves == first
    popup.move(popup.pos().x() + 10, popup.pos().y() + 10)
    field._on_tick()
    assert field.engine._popup_waves == first
    last_origin = field.engine._popup_wave_origin
    popup.reject()
    qtbot.waitUntil(lambda: not popup.isVisible())
    field._on_tick()
    assert len(field.engine._popup_waves) == 2 * int(enabled)
    if enabled:
        assert field.engine._popup_waves[-1][1] == last_origin
        assert len(last_origin) == 4
    assert field.engine.gravity_radius == 0
    assert field.engine._gravity_impulses == []
    closed = list(field.engine._popup_waves)
    field._on_tick()
    assert field.engine._popup_waves == closed
    field.stop()


def test_origin_resolves_the_visible_popup_in_this_window(qtbot, monkeypatch):
    owner = QWidget()
    owner.resize(640, 480)
    qtbot.addWidget(owner)
    owner.show()
    widget = ambient.AmbientWidget(owner, theme='data_art_impulse_lens', palette='spacr',
                                  popup_wave_frequency=20, gravity_radius=0)
    widget.setGeometry(owner.rect())
    widget.show()
    widget._timer.stop()
    popup = QDialog(owner)
    qtbot.addWidget(popup)
    popup.resize(100, 80)
    popup.move(owner.mapToGlobal(owner.rect().center()) - popup.rect().center())
    popup.show()
    monkeypatch.setattr(ambient.QApplication, 'activeModalWidget', lambda: popup)
    monkeypatch.setattr(ambient.QApplication, 'activePopupWidget', lambda: None)
    perimeter = widget._popup_wave_origin_for_tick()
    assert perimeter is not None and len(perimeter) == 4
    assert perimeter[0][0] == perimeter[1][0] == perimeter[2][0]
    assert perimeter[0][2] == perimeter[1][2] == perimeter[3][0]
    assert perimeter[0][1] == perimeter[2][1] == perimeter[3][1]
    assert perimeter[1][1] == perimeter[2][3] == perimeter[3][3]
    assert (perimeter[0][0] + perimeter[0][2]) / 2 == pytest.approx(0.5, abs=0.02)
    assert (perimeter[0][1] + perimeter[1][1]) / 2 == pytest.approx(0.5, abs=0.02)
    popup.move(owner.mapToGlobal(owner.rect().bottomRight()) + popup.rect().bottomRight())
    assert widget._popup_wave_origin_for_tick() is None
    popup.hide()
    assert widget._popup_wave_origin_for_tick() is None
    other = QDialog()
    qtbot.addWidget(other)
    other.show()
    monkeypatch.setattr(ambient.QApplication, 'activeModalWidget', lambda: other)
    assert widget._popup_wave_origin_for_tick() is None
    widget.set_animating(False)


def test_widget_wave_controls_keep_other_controls_and_skip_equal_mutations(qtbot, monkeypatch):
    widget = ambient.AmbientWidget(theme='data_art_impulse_lens', blink_percent=2,
                                  popup_wave_frequency=20, density=0.5, gravity_radius=0)
    qtbot.addWidget(widget)
    widget.stop()
    calls = []
    original = widget._mutate_engine

    def mutate(change):
        calls.append(True)
        original(change)

    monkeypatch.setattr(widget, '_mutate_engine', mutate)
    widget.set_popup_wave_frequency(20)
    assert not calls and widget.popup_wave_frequency() == 20
    widget.set_popup_wave_frequency(120)
    assert calls == [True] and widget.popup_wave_frequency() == 60
    assert widget.engine.popup_wave_frequency == 60
    widget.set_popup_wave_frequency(math.nan)
    assert len(calls) == 2 and widget.popup_wave_frequency() == 0
    assert widget.engine.popup_wave_frequency == 0
    assert widget.blink_percent() == 2 and widget.density() == 0.5
    assert widget.gravity_radius() == 0


def test_popup_publication_race_never_applies_future_origin_to_old_tick(qapp):
    class InterleavedQueue(ambient._QueuedArtInput):
        armed = False

        def __getattribute__(self, name):
            value = super().__getattribute__(name)
            if name == '_snapshot' and self.armed:
                self.armed = False
                self._offer(1, None, (), popup_origin=(0.7, 0.3), popup_id=2)
            return value

    made = engine(5)
    queue = InterleavedQueue()
    queue._offer(1, None, (), popup_origin=(0.2, 0.8), popup_id=1)
    queue.armed = True
    queue._consume(made)
    assert made._popup_wave_origin is None and made._popup_waves == []
    assert made.time == 1
    queue._consume(made)
    assert made.time == 2 and made._popup_wave_origin == (0.7, 0.3)
    assert made._popup_waves == [(1, (0.7, 0.3))]


def test_rewinding_and_expiring_popup_waves_clear_material_without_changing_owned_pixels(qapp):
    made = engine(5)
    made.set_time(10)
    made._set_popup_wave_origin((0.4, 0.6))
    made.advance(0.5)
    published = made.shade(320, 240)
    saved = bytes(published.constBits())
    made.set_time(0)
    reference = engine(5)
    assert bytes(made.shade(320, 240).constBits()) == bytes(reference.shade(320, 240).constBits())
    material = next(value for key, value in made._material_cache.items() if key[0] == 'impulse_lens')
    assert material[3] == {}
    made._set_popup_wave_origin(None)
    made.set_time(20)
    reference.set_time(20)
    assert bytes(made.shade(320, 240).constBits()) == bytes(reference.shade(320, 240).constBits())
    assert bytes(published.constBits()) == saved


def test_frequency_apply_revert_keeps_preferences_open(private_preferences, qtbot):
    prefs.set_ambient_animation('data_art_impulse_lens')
    dialog, _, _ = _dialog(qtbot)
    value = dialog.findChild(QDoubleSpinBox, 'FieldPopupWaveFrequencyValue')
    assert value.value() == 5
    value.setValue(12.3)
    question = _apply(dialog, qtbot)
    assert prefs._field_popup_wave_frequency() == 12.3
    _answer(question, 'Revert', qtbot)
    assert prefs._field_popup_wave_frequency() == 5
    assert value.value() == 12.3 and dialog.isVisible()
    question = _apply(dialog, qtbot)
    _answer(question, 'Keep', qtbot)
    assert prefs._field_popup_wave_frequency() == 12.3
