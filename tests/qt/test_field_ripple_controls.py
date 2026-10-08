"""Field ripples remain independent of gravity, with one persistent switch."""

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QDialogButtonBox, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient
from spacr.qt.widgets.toggle import Toggle


@pytest.fixture(autouse=True)
def ripple_store(tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "ripples.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)
    return store


def _field(qtbot, gravity_radius=0):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 180)
    field = ambient.AmbientWidget(
        host, theme="data_art_impulse_lens", gravity_radius=gravity_radius,
        popup_wave_frequency=0, blink_percent=0, density=1, seed=19)
    field.resize(host.size())
    host.show()
    field.show()
    field._timer.stop()
    return host, field


@pytest.mark.parametrize("radius", [0, float("nan"), float("inf"), float("-inf")])
def test_actual_click_creates_visible_ripples_with_gravity_disabled(qtbot, radius):
    host, field = _field(qtbot, radius)
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
    field._on_tick()
    field._timer.stop()
    assert field.engine.gravity_radius == 0
    assert not field.engine._gravity_impulses
    assert len(field.engine._popup_waves) == 1
    field.engine.advance(0.3)
    animated = field.engine.shade(320, 180).copy()
    plain = ambient.make_engine("data_art_impulse_lens", "spacr", field.engine.identity,
                                seed=19, density=1, blink_percent=0,
                                popup_wave_frequency=0)
    plain.set_time(field.engine.time)
    reference = plain.shade(320, 180)
    assert not np.array_equal(np.asarray(animated.bits()), np.asarray(reference.bits()))


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_switch_blocks_click_and_edge_ripples_without_changing_gravity(qtbot, edge):
    host, field = _field(qtbot)
    field._ripple_from_edge(edge)
    assert field.engine._popup_waves
    field.set_ripples_enabled(False)
    assert not field.engine._popup_waves
    field._ripple_from_edge(edge)
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
    field._on_tick()
    field._timer.stop()
    assert not field.engine._popup_waves
    assert field.engine.gravity_radius == 0
    field.set_ripples_enabled(True)
    assert not field.engine._popup_waves
    field._ripple_from_edge(edge)
    assert len(field.engine._popup_waves) == 3


def test_disabled_queue_events_cannot_reappear_when_reenabled():
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                                 popup_wave_frequency=60)
    queue = ambient._QueuedArtInput()
    engine.set_ripples_enabled(False)
    queue._offer_boundary_waves(((0.0, 0.5),))
    queue._offer(0.1, None, ((0.4, 0.5),), popup_origin=(0.3, 0.2))
    queue._consume(engine)
    engine.advance(1.0)
    assert not engine._popup_waves
    engine.set_popup_wave_frequency(0)
    engine.set_ripples_enabled(True)
    queue._consume(engine)
    assert not engine._popup_waves
    queue._offer_boundary_waves(((1.0, 0.5),))
    queue._consume(engine)
    assert engine._popup_waves == [(engine.time, (1.0, 0.5))]


def test_switch_preserves_gravity_and_drag_but_survives_engine_rebuild(qtbot):
    _host, field = _field(qtbot)
    field.set_gravity_radius(0.4)
    field._field_grab = ((0.5, 0.5), (0.1, 0.0))
    field._offer_field_grab()
    field.set_ripples_enabled(False)
    assert field.engine.gravity_radius == 0.4
    assert field.engine._field_grab_target == (0.1, 0.0)
    field.set_theme("data_art_genetic_advection")
    field.set_theme("data_art_impulse_lens")
    assert not field.engine.ripples_enabled
    assert field.engine.gravity_radius == 0.4


@pytest.mark.parametrize("staged", [False, True])
def test_enabling_consumes_disabled_clicks_under_the_old_switch(qtbot, staged):
    host, field = _field(qtbot, gravity_radius=0.4)
    field.set_ripples_enabled(False)
    if staged:
        field._art_input._offer(0.0, None, ((0.4, 0.5),))
    else:
        qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
        assert field._pending_art_impulses
    field.set_ripples_enabled(True)
    assert field.engine._gravity_impulses
    assert not field.engine._popup_waves
    assert not field._pending_art_impulses
    field._on_tick()
    field._timer.stop()
    assert not field.engine._popup_waves


def test_preferences_cancel_and_save_update_existing_fields(
        ripple_store, qtbot, qt_theme_applied):
    prefs.set_ambient_animation("data_art_impulse_lens")
    _host, field = _field(qtbot)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    toggle = dialog.findChild(Toggle, "FieldRipplesEnabled")
    assert toggle.isChecked()
    toggle.setChecked(False)
    dialog.reject()
    assert prefs._field_ripples_enabled()
    assert field.engine.ripples_enabled
    accepted = prefs.PreferencesDialog()
    qtbot.addWidget(accepted)
    accepted.findChild(Toggle, "FieldRipplesEnabled").setChecked(False)
    accepted.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert not prefs._field_ripples_enabled()
    assert not field.engine.ripples_enabled
    assert field.engine.gravity_radius == prefs._ambient_gravity_radius()


@pytest.mark.parametrize("previously_enabled", [False, True])
def test_reenabling_cannot_revive_old_gravity_wave_pixels(previously_enabled):
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                                popup_wave_frequency=0, blink_percent=0, seed=19)
    engine.set_gravity_radius(0.4)
    engine.set_ripples_enabled(previously_enabled)
    engine._add_impulse((0.5, 0.5))
    engine.set_time(0.3)
    engine.set_ripples_enabled(False)
    before = engine.shade(320, 180).copy()
    engine.set_ripples_enabled(True)
    after = engine.shade(320, 180).copy()
    assert engine._gravity_impulses
    assert np.array_equal(np.asarray(before.bits()), np.asarray(after.bits()))
    engine._add_impulse((0.5, 0.5))
    fresh = engine.shade(320, 180).copy()
    assert not np.array_equal(np.asarray(after.bits()), np.asarray(fresh.bits()))


def test_gravity_wave_bookkeeping_is_bounded_and_clears_with_gravity():
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418")
    engine.set_gravity_radius(0.4)
    for _ in range(100):
        engine._add_impulse((0.5, 0.5))
    assert len(engine._gravity_impulses) == 24
    assert len(engine._gravity_ripple_impulses) == 24
    engine.set_gravity_radius(0)
    assert not engine._gravity_ripple_impulses
