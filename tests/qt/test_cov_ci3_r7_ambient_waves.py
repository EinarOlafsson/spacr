"""Column waves refuse unusable columns; expired waves are still; deleted widgets never ripple."""

import math

import pytest

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog, QWidget  # noqa: E402

from spacr.qt.widgets import ambient  # noqa: E402


def _engine(theme="data_art_impulse_lens"):
    engine = ambient.make_engine(theme, "spacr", "#191919", seed=684, density=1,
                                 blink_percent=0, popup_wave_frequency=0)
    engine.set_time(10.0)
    return engine


def test_a_valid_column_wave_is_clamped_and_sorted():
    engine = _engine()
    engine._add_column_wave((0.8, -0.5, 1.7))
    assert engine._column_waves == [(10.0, (0.0, 0.8, 1.0), 1.0)]


@pytest.mark.parametrize("column", [
    None, (0.1, 0.5), ("left", 0.5, 0.5), (0.1, object(), 0.5), 7,
    (math.nan, 0.5, 0.5), (0.1, math.inf, 0.5), (0.1, 0.5, -math.inf),
    (0.4, 0.4, 0.5), (1.2, 1.5, 0.5), (-0.3, -0.1, 0.2),
])
def test_an_unusable_column_sends_no_wave(column):
    engine = _engine()
    engine._add_column_wave(column)
    assert engine._column_waves == []


def test_column_waves_need_ripples_and_the_impulse_lens():
    engine = _engine()
    engine.set_ripples_enabled(False)
    engine._add_column_wave((0.1, 0.5, 0.5))
    assert engine._column_waves == []
    other = _engine("data_art_tissue_facets")
    other._add_column_wave((0.1, 0.5, 0.5))
    assert other._column_waves == []


@pytest.mark.parametrize("age", [-0.01, ambient.RIPPLE_LIFETIME_S,
                                 ambient.RIPPLE_LIFETIME_S + 3.0])
def test_a_wave_outside_its_lifetime_is_still(age):
    engine = _engine()
    assert engine._ripple_wave(0.2, age) == (0.0, 0.0)
    shift, glow = engine._ripple_wave(0.2, 0.75)
    assert abs(float(shift)) + abs(float(glow)) > 0.0


def _deleted(widget):
    import shiboken6

    shiboken6.delete(widget)
    assert not shiboken6.isValid(widget)
    return widget


def test_a_deleted_widget_is_not_an_explicit_ripple_window(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.setProperty(ambient.EXPLICIT_RIPPLES_PROPERTY, True)
    child = QWidget(window)
    assert ambient._ripples_are_explicit(child) is True
    orphan = _deleted(QWidget())
    assert ambient._ripples_are_explicit(orphan) is False


def test_a_deleted_popup_sends_no_wave(qtbot, monkeypatch):
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(400, 300)
    field = ambient.AmbientWidget(host, theme="data_art_impulse_lens", gravity_radius=0,
                                  popup_wave_frequency=0, blink_percent=0, density=1,
                                  seed=19)
    field.resize(host.size())
    host.show()
    field._timer.stop()
    before = ambient._CONTAINER_RIPPLES[0]
    dialog = _deleted(QDialog(host))
    assert ambient._field_ripple_for_popup(dialog) is None
    assert ambient._CONTAINER_RIPPLES[0] == before + 1
    assert field.engine._popup_waves == []
