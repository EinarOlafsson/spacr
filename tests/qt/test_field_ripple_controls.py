"""Field ripples remain independent of gravity, with one persistent switch."""

import numpy as np
import pytest
from PySide6.QtCore import QRect, QSettings, Qt
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
    assert len(field.engine._popup_waves) == 1
    segment = field.engine._popup_waves[0][1]
    assert len(segment) == 1
    assert segment[0] in ((0.0, 0.0, 0.0, 1.0),
                          (1.0, 0.0, 1.0, 1.0),
                          (0.0, 0.0, 1.0, 0.0),
                          (0.0, 1.0, 1.0, 1.0))


def test_one_edge_wave_reaches_both_ends_of_the_field_material():
    kwargs = dict(seed=19, density=1, blink_percent=0, popup_wave_frequency=0)
    wave = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418", **kwargs)
    idle = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418", **kwargs)
    boundary = ((0.0, 0.0, 0.0, 1.0),)
    wave.set_gravity_radius(0)
    wave._add_ripple(boundary)
    wave.set_time(0.3)
    idle.set_time(0.3)
    painted_image = wave.shade(320, 180)
    reference_image = idle.shade(320, 180)
    painted = np.frombuffer(painted_image.bits(), dtype=np.uint32,
                            count=320 * 180).reshape(180, 320)
    reference = np.frombuffer(reference_image.bits(), dtype=np.uint32,
                              count=320 * 180).reshape(180, 320)
    changed = np.flatnonzero(np.any(painted != reference, axis=1))
    assert np.any(changed < 45) and np.any(changed > 135)

    key = ("impulse_lens", 320, 180, wave.size, wave.density)
    x, y, _, fields = wave._material_cache[key]
    ex, ey, distance = fields[boundary]
    inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    assert np.allclose(ey[inside], 0)
    assert np.allclose(ex[inside], x[inside])
    assert np.allclose(distance[inside],
                       np.sqrt((x[inside] * 320 / 180) ** 2 + 1e-6))


def test_one_popup_wave_keeps_all_four_edges_and_nearest_segment_material():
    made = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                               seed=19, density=1, blink_percent=0,
                               popup_wave_frequency=60)
    perimeter = ((0.2, 0.2, 0.8, 0.2), (0.2, 0.8, 0.8, 0.8),
                 (0.2, 0.2, 0.2, 0.8), (0.8, 0.2, 0.8, 0.8))
    made._set_popup_wave_origin(perimeter, popup_id=12)
    made.advance(1)
    assert len(made._popup_waves) == 2
    assert all(origin == perimeter for _age, origin in made._popup_waves)
    made.shade(320, 180)
    key = ("impulse_lens", 320, 180, made.size, made.density)
    x, y, _, fields = made._material_cache[key]
    ex, ey, distance = fields[perimeter]
    for target_x, target_y, expected_axis in ((0.5, 0.16, "y"),
                                               (0.5, 0.84, "y"),
                                               (0.16, 0.5, "x"),
                                               (0.84, 0.5, "x")):
        index = np.argmin((x - target_x) ** 2 + (y - target_y) ** 2)
        assert np.isfinite(distance.flat[index])
        if expected_axis == "x":
            assert abs(ey.flat[index]) < 0.02
        else:
            assert abs(ex.flat[index]) < 0.02
    for _ in range(7):
        made._add_ripple(perimeter)
    assert len(made._popup_waves) == 6
    assert all(origin == perimeter for _age, origin in made._popup_waves)


def test_invalid_edges_cannot_publish_partial_or_nonfinite_waves():
    made = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                               popup_wave_frequency=0)
    invalid = ((), ((0.0, 0.0, 1.0, 0.0),) * 5,
               ((0.0, 0.0, 0.5, 0.5),),
               ((float("nan"), 0.0, float("nan"), 1.0),),
               ("not a segment",))
    for origin in invalid:
        made._set_popup_wave_origin(origin)
        made._add_ripple(origin)
    made._add_ripple((float("nan"), 0.5))
    assert made._popup_wave_origin is None
    assert not made._popup_waves


def test_clipped_widget_perimeter_keeps_only_its_real_visible_edges(qtbot):
    host, field = _field(qtbot)
    child = QWidget(host)
    child.setGeometry(25, 20, 100, 80)
    child.show()
    child.setGeometry(35, 30, 120, 90)
    ambient.field_ripple_for_widget(child, edge="bottom")
    assert field.engine._popup_waves[-1][1] == (
        (35 / 320, 120 / 180, 155 / 320, 120 / 180),)

    field._ripple_from_rect(QRect(-40, 30, 140, 120))
    segments = field.engine._popup_waves[-1][1]
    assert len(segments) == 3
    assert not any(x0 == x1 == 0 for x0, _y0, x1, _y1 in segments)
    assert segments == ((0.0, 30 / 180, 100 / 320, 30 / 180),
                        (0.0, 150 / 180, 100 / 320, 150 / 180),
                        (100 / 320, 30 / 180, 100 / 320, 150 / 180))
    assert len(field._art_input._boundary_waves) == 2

    field._ripple_from_rect(QRect(30, 40, 100, 60), edge="top")
    assert field.engine._popup_waves[-1][1] == (
        (30 / 320, 40 / 180, 130 / 320, 40 / 180),)
    ambient.field_ripple_for_widget(host, edge="right")
    assert field.engine._popup_waves[-1][1] == ((1.0, 0.0, 1.0, 1.0),)
    before = len(field.engine._popup_waves)
    field._ripple_from_rect(QRect(-200, 30, 100, 80))
    field._ripple_from_rect(QRect(20, 30, 0, 80))
    assert len(field.engine._popup_waves) == before


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


def test_container_header_click_has_one_final_edge_and_no_point_wave(qtbot):
    from spacr.qt.widgets.section import Section

    host, field = _field(qtbot)
    section = Section("Settings", parent=host)
    section.setGeometry(30, 20, 140, 100)
    section.show()
    for expanded in (True, False):
        field.engine._popup_waves.clear()
        qtbot.mouseClick(section.header(), Qt.LeftButton)
        qtbot.waitUntil(lambda: bool(field.engine._popup_waves))
        field._on_tick()
        field._timer.stop()
        assert section._expanded is expanded
        assert not field._pending_art_impulses
        assert len(field.engine._popup_waves) == 1
        origin = field.engine._popup_waves[0][1]
        bottom = (section.y() + section.height()) / field.height()
        assert origin == ((section.x() / field.width(), bottom,
                           (section.x() + section.width()) / field.width(), bottom),)


@pytest.mark.parametrize("value,expected", [(0, 0), (2, 2), (9, 2),
    (-1, 0), (float("nan"), 1), (float("inf"), 1), ("invalid", 1)])
def test_ripple_intensity_settings_are_finite_and_bounded(ripple_store, value, expected):
    ripple_store.setValue(prefs._KEY_FIELD_RIPPLE_INTENSITY, value)
    assert prefs._field_ripple_intensity() == expected


def test_intensity_changes_wave_pixels_and_survives_theme_rebuild(qtbot):
    _host, field = _field(qtbot)
    field.engine._add_ripple((0.5, 0.5))
    field.engine.set_time(0.4)
    field._set_ripple_intensity(0)
    quiet = field.engine.shade(320, 180).copy()
    plain = ambient.make_engine("data_art_impulse_lens", "spacr", field.engine.identity,
                                seed=19, density=1, blink_percent=0,
                                popup_wave_frequency=0)
    plain.set_time(0.4)
    assert np.array_equal(np.asarray(quiet.bits()), np.asarray(plain.shade(320, 180).bits()))
    field._set_ripple_intensity(2)
    strong = field.engine.shade(320, 180).copy()
    assert not np.array_equal(np.asarray(quiet.bits()), np.asarray(strong.bits()))
    field.set_theme("data_art_genetic_advection")
    field.set_theme("data_art_impulse_lens")
    assert field.engine._ripple_intensity == 2
    assert field.engine.gravity_radius == 0


def test_intensity_preferences_save_cancel_and_reload(qtbot, ripple_store, qt_theme_applied):
    from PySide6.QtWidgets import QDoubleSpinBox

    prefs.set_ambient_animation("data_art_impulse_lens")
    _host, field = _field(qtbot)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    value = dialog.findChild(QDoubleSpinBox, "FieldRippleIntensity")
    assert value.value() == 100
    value.setValue(180)
    dialog.reject()
    assert prefs._field_ripple_intensity() == 1
    accepted = prefs.PreferencesDialog()
    qtbot.addWidget(accepted)
    accepted.findChild(QDoubleSpinBox, "FieldRippleIntensity").setValue(180)
    accepted.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert prefs._field_ripple_intensity() == 1.8
    assert field.engine._ripple_intensity == 1.8
    restored = prefs.PreferencesDialog()
    qtbot.addWidget(restored)
    assert restored.findChild(QDoubleSpinBox, "FieldRippleIntensity").value() == 180
