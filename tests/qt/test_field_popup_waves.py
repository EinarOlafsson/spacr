"""Optional field waves use GUI popup centres and bounded worker-owned state."""
import math

import pytest
from PySide6.QtWidgets import QDialog, QDoubleSpinBox, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient
from tests.qt.test_requested_appearance_defaults import empty_store
from tests.qt.test_preferences_apply import private_preferences, _dialog, _apply, _answer


def engine(frequency=0):
    made = ambient.make_engine('data_art_impulse_lens', 'spacr', '#101010',
                               seed=12, popup_wave_frequency=frequency)
    made.set_gravity_radius(0)
    return made


def test_default_and_disabled_frequency_emit_no_waves(qapp, empty_store):
    assert prefs._field_popup_wave_frequency() == 5
    prefs._set_field_popup_wave_frequency(0)
    made = engine()
    made._set_popup_wave_origin((0.3, 0.7))
    made.advance(120)
    assert made._popup_waves == []
    made.set_popup_wave_frequency(60)
    made.advance(1)
    assert len(made._popup_waves) == 1
    made.set_popup_wave_frequency(0)
    assert made._popup_waves == []
    made.advance(120)
    assert made._popup_waves == []


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
    queue._offer(.04, None, (), popup_origin=(.3, .7), popup_id=2)
    queue._consume(made)
    assert len(made._popup_waves) == 3


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
    point = widget._popup_wave_origin_for_tick()
    assert point is not None
    assert point[0] == pytest.approx(0.5, abs=0.01)
    assert point[1] == pytest.approx(0.5, abs=0.01)
    popup.hide()
    assert widget._popup_wave_origin_for_tick() is None
    other = QDialog()
    qtbot.addWidget(other)
    other.show()
    monkeypatch.setattr(ambient.QApplication, 'activeModalWidget', lambda: other)
    assert widget._popup_wave_origin_for_tick() is None
    widget.set_animating(False)


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
