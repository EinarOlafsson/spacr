"""Dot blinking has a useful tiny range and normal Apply/Keep/Revert behavior."""
import numpy as np
import pytest

from PySide6.QtWidgets import QDoubleSpinBox, QSlider

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient
from tests.qt.test_preferences_apply import private_preferences, _dialog, _apply, _answer
from tests.qt.test_requested_appearance_defaults import empty_store


@pytest.mark.parametrize('value,expected', [
    (0, 0), (0.00001, 0.00001), (0.02, 0.02), (10, 10),
    (11, 10), (float('nan'), 0), (float('inf'), 0),
    ('bad', 0), (None, 0),
])
def test_blink_percentage_is_persisted_and_validated(empty_store, value, expected):
    prefs._set_ambient_blink_percent(value)
    assert prefs._ambient_blink_percent() == expected
    assert float(prefs._settings().value(prefs._KEY_AMBIENT_BLINK_PERCENT)) == expected


@pytest.mark.parametrize('theme', ['data_art_impulse_lens', 'data_art_genetic_advection',
                                    'data_art_point_atlas', 'drift'])
def test_dot_themes_use_the_control_without_changing_geometry(qapp, theme):
    density = 1.0 if theme == 'drift' else 0.1
    engine = ambient.make_engine(theme, 'spacr', '#101010', seed=19, density=density)
    def render():
        from PySide6.QtGui import QImage, QPainter
        if hasattr(engine, "shade"):
            return engine.shade(640, 480)
        image = QImage(640, 480, QImage.Format_RGB32)
        image.fill(0xff101010)
        painter = QPainter(image)
        try:
            engine.paint(painter, 640, 480)
        finally:
            painter.end()
        return image
    small = render()
    count = engine._blink_selection[0][1]
    assert engine._blink_selection[1].size == 0
    first_bytes = bytes(small.constBits())
    engine.set_blink_percent(10)
    large = render()
    assert engine._blink_selection[0][1] == count
    assert abs(engine._blink_selection[1].size - count / 10) <= 1
    assert bytes(large.constBits()) != first_bytes
    assert bytes(small.constBits()) == first_bytes
    assert engine.density == density


def test_tiny_counts_are_not_rounded_up_to_one_dot(qapp):
    engine = ambient.make_engine('data_art_impulse_lens', 'spacr', '#101010', seed=19)
    assert engine._blinking_indices(100).size == 0
    engine.set_blink_percent(0.1)
    counts = []
    for tick in range(1000):
        engine.time = tick / 4
        selected = engine._blinking_indices(100)
        assert len(selected) == len(np.unique(selected))
        counts.append(len(selected))
    assert set(counts) == {0, 1}
    assert 0.07 < np.mean(counts) < 0.13


def test_random_advection_blinks_across_all_trail_coordinates(qapp):
    engine = ambient.make_engine('data_art_genetic_advection', 'random',
                                 '#101010', seed=19, density=.25)
    engine.set_blink_percent(10)
    image = engine.shade(640, 360)
    flat = np.frombuffer(image.constBits(), dtype=np.uint32)
    assert np.count_nonzero(flat == 0xffffffff) > 0
    assert engine._blink_selection[0][1] > 12
    before = bytes(image.constBits())
    engine.advance(.25)
    next_image = engine.shade(640, 360)
    assert bytes(next_image.constBits()) != before
    assert bytes(image.constBits()) == before


def test_slider_spans_all_six_decades_and_allows_exact_input(private_preferences, qtbot):
    dialog, _, _ = _dialog(qtbot)
    slider = dialog.findChild(QSlider, 'AmbientBlinkPercent')
    number = dialog.findChild(QDoubleSpinBox, 'AmbientBlinkPercentValue')
    assert (number.minimum(), number.maximum(), number.decimals()) == (0.0, 10, 5)
    slider.setValue(slider.maximum())
    assert number.value() == 10
    slider.setValue(slider.minimum())
    assert number.value() == 0
    slider.setValue(1)
    assert number.value() == 0.00001
    slider.setValue(301)
    assert number.value() == 0.01
    number.setValue(0.12345)
    assert 400 < slider.value() < 420
    assert number.value() == 0.12345


def test_blink_apply_keep_revert_and_reopen(private_preferences, qtbot):
    prefs._set_ambient_blink_percent(0.02)
    widget = ambient.AmbientWidget(theme='data_art_impulse_lens', palette='spacr')
    qtbot.addWidget(widget)
    widget.set_animating(False)
    prefs.set_ambient_animation('data_art_impulse_lens')
    dialog, _, _ = _dialog(qtbot)
    value = dialog.findChild(QDoubleSpinBox, 'AmbientBlinkPercentValue')
    value.setValue(0.12345)
    question = _apply(dialog, qtbot)
    assert prefs._ambient_blink_percent() == 0.12345
    assert widget.blink_percent() == 0.12345
    _answer(question, 'Revert', qtbot)
    assert dialog.isVisible() and value.value() == 0.12345
    assert prefs._ambient_blink_percent() == 0.02
    assert widget.blink_percent() == 0.02
    question = _apply(dialog, qtbot)
    _answer(question, 'Keep', qtbot)
    assert prefs._ambient_blink_percent() == 0.12345
    new, _, _ = _dialog(qtbot)
    assert new.findChild(QDoubleSpinBox, 'AmbientBlinkPercentValue').value() == 0.12345
    widget.set_animating(False)
