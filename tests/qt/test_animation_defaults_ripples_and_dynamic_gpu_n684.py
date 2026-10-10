"""N684: animation defaults, ripple controls, click-ripple suppression and Dynamic animation."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QDialogButtonBox, QDoubleSpinBox,
                               QSlider, QWidget)

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient
from spacr.qt.widgets.toggle import Toggle


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "n684.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)
    return settings


class _Handle:
    def __init__(self, app_key, user_visible=True):
        self.app_key = app_key
        self.user_visible = user_visible


class _Registry(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.handles = []

    def active(self):
        return list(self.handles)

    def start(self, key, **kwargs):
        handle = _Handle(key, **kwargs)
        self.handles.append(handle)
        self.changed.emit()
        return handle

    def finish(self, handle):
        self.handles.remove(handle)
        self.changed.emit()


@pytest.fixture
def jobs(monkeypatch, qapp):
    from spacr.qt import bridge

    fake = _Registry()
    monkeypatch.setattr(bridge, "registry", lambda: fake)
    monkeypatch.setattr(ambient, "_analysis_gpu_present", lambda: True)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    policy = ambient._animation_resource_policy()
    yield fake, policy
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)


def _field(qtbot, **kwargs):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 180)
    options = dict(theme="data_art_impulse_lens", gravity_radius=0,
                   popup_wave_frequency=0, blink_percent=0, density=1, seed=19)
    options.update(kwargs)
    field = ambient.AmbientWidget(host, **options)
    field.resize(host.size())
    host.show()
    field.show()
    field._timer.stop()
    return host, field


def _save(dialog):
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()


def test_fresh_defaults_field_density_25_and_dark_background_191919(store, monkeypatch):
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "dark")
    assert prefs.get_ambient_theme() == "data_art_impulse_lens"
    assert prefs.get_ambient_density() == pytest.approx(0.25)
    for theme in ("blobs", "drift", "data_art_genetic_advection", "data_art_point_atlas"):
        assert ambient._default_density_for(theme) == ambient.DEFAULT_DENSITY
    assert prefs._ambient_background_choice() == "#191919"
    assert prefs._effective_ambient_background() == QColor(25, 25, 25)
    assert prefs._ambient_gravity_radius() == pytest.approx(0.15)


def test_saved_density_and_background_are_never_overwritten(store, monkeypatch):
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "dark")
    store.setValue(prefs._KEY_AMBIENT_DENSITY, 0.4)
    store.setValue(prefs._KEY_AMBIENT_BACKGROUND, "#202830")
    assert prefs.get_ambient_density() == pytest.approx(0.4)
    assert prefs._ambient_background_choice() == "#202830"
    prefs._set_ambient_background_choice(None)
    assert prefs._ambient_background_choice() is None
    assert store.value(prefs._KEY_AMBIENT_BACKGROUND) == "theme"


def test_untouched_density_follows_each_theme_default(store):
    prefs.set_ambient_density(0.25)
    assert not store.contains(prefs._KEY_AMBIENT_DENSITY)
    prefs.set_ambient_animation("blobs")
    assert prefs.get_ambient_density() == pytest.approx(ambient.DEFAULT_DENSITY)
    prefs.set_ambient_density(0.6)
    assert store.contains(prefs._KEY_AMBIENT_DENSITY)
    prefs.set_ambient_animation("data_art_impulse_lens")
    assert prefs.get_ambient_density() == pytest.approx(0.6)


def test_light_theme_keeps_its_own_page_when_no_background_was_chosen(store, monkeypatch):
    from spacr.qt.theme import active_page_colour

    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "light")
    assert prefs._effective_ambient_background() == QColor(active_page_colour())


def test_dialog_density_slider_follows_theme_until_chosen(store, qtbot, qt_theme_applied):
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    density = dialog.findChild(QSlider, "AmbientDensity")
    combo = dialog.findChild(QComboBox, "AmbientTheme") or next(
        box for box in dialog.findChildren(QComboBox)
        if box.findData("data_art_impulse_lens") >= 0 and box.findData("blobs") >= 0)
    assert density.value() == 25
    combo.setCurrentIndex(combo.findData("blobs"))
    assert density.value() == 10
    combo.setCurrentIndex(combo.findData("data_art_impulse_lens"))
    assert density.value() == 25
    _save(dialog)
    assert not store.contains(prefs._KEY_AMBIENT_DENSITY)


@pytest.mark.parametrize("old,expected", [(False, 0.0), (True, 1.0), ("false", 0.0)])
def test_old_ripple_switch_migrates_once_into_the_slider(store, old, expected):
    store.setValue(prefs._KEY_FIELD_RIPPLES, old)
    assert prefs._field_ripple_intensity() == expected
    assert prefs._field_ripples_enabled() is (expected > 0)
    assert not store.contains(prefs._KEY_FIELD_RIPPLES)
    prefs._set_field_ripple_intensity(1.5)
    store.setValue(prefs._KEY_FIELD_RIPPLES, False)
    assert prefs._field_ripple_intensity() == 1.5


def test_ripple_slider_zero_is_off_and_click_ripples_save_and_cancel(
        store, qtbot, qt_theme_applied):
    prefs.set_ambient_animation("data_art_impulse_lens")
    _host, field = _field(qtbot)
    assert prefs._field_click_ripples() is True
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    slider = dialog.findChild(QSlider, "FieldRipples")
    value = dialog.findChild(QDoubleSpinBox, "FieldRippleIntensity")
    click = dialog.findChild(Toggle, "FieldClickRipples")
    assert dialog.findChild(Toggle, "FieldRipplesEnabled") is None
    assert slider.value() == 100 and click.isChecked()
    slider.setValue(0)
    assert value.value() == 0 and value.text() == "Off"
    click.setChecked(False)
    dialog.reject()
    assert prefs._field_ripple_intensity() == 1.0 and prefs._field_click_ripples()
    accepted = prefs.PreferencesDialog()
    qtbot.addWidget(accepted)
    accepted.findChild(QSlider, "FieldRipples").setValue(0)
    accepted.findChild(Toggle, "FieldClickRipples").setChecked(False)
    _save(accepted)
    assert not prefs._field_ripples_enabled()
    assert not prefs._field_click_ripples()
    assert not field.engine.ripples_enabled
    assert field._click_ripples is False
    reopened = prefs.PreferencesDialog()
    qtbot.addWidget(reopened)
    assert reopened.findChild(QSlider, "FieldRipples").value() == 0
    assert not reopened.findChild(Toggle, "FieldClickRipples").isChecked()


def test_plain_click_ripples_at_its_position_after_settling(qtbot):
    host, field = _field(qtbot)
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
    assert not field.engine._popup_waves
    qtbot.waitUntil(lambda: bool(field.engine._popup_waves))
    origin = field.engine._popup_waves[-1][1]
    assert origin == pytest.approx((0.5, 0.5), abs=0.01)


@pytest.mark.parametrize("setup", ["click_off", "slider_zero"])
def test_disabled_click_ripples_or_zero_slider_send_nothing(qtbot, setup):
    host, field = _field(qtbot)
    if setup == "click_off":
        field.set_click_ripples(False)
    else:
        field._set_ripple_intensity(0)
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
    qtbot.wait(ambient.CLICK_RIPPLE_SETTLE_MS * 3)
    assert not field.engine._popup_waves
    field._ripple_from_edge("left")
    assert bool(field.engine._popup_waves) is (setup == "click_off")


def test_click_that_closes_a_container_sends_only_the_container_ripple(qtbot):
    host, field = _field(qtbot)
    panel = QWidget(host)
    panel.setGeometry(40, 30, 120, 80)
    panel.show()

    class Closer(QObject):
        def eventFilter(self, obj, event):
            if obj is host and event.type() == event.Type.MouseButtonRelease:
                panel.hide()
                ambient.field_ripple_for_widget(panel, edge="bottom")
            return False

    closer = Closer(host)
    host.installEventFilter(closer)
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().bottomRight() - host.rect().center() / 4)
    qtbot.wait(ambient.CLICK_RIPPLE_SETTLE_MS * 3)
    origins = [origin for _when, origin in field.engine._popup_waves]
    assert len(origins) == 1
    assert isinstance(origins[0][0], tuple)
    host.removeEventFilter(closer)
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
    qtbot.waitUntil(lambda: len(field.engine._popup_waves) == 2)
    assert not isinstance(field.engine._popup_waves[-1][1][0], tuple)


def test_a_popup_closing_during_the_click_also_suppresses_it(qtbot):
    host, field = _field(qtbot)
    field._popup_was_open = True
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
    field._on_tick()
    field._timer.stop()
    qtbot.wait(ambient.CLICK_RIPPLE_SETTLE_MS * 3)
    assert not any(not isinstance(origin[0], tuple)
                   for _when, origin in field.engine._popup_waves)


def test_a_drag_of_the_field_is_not_a_click(qtbot):
    host, field = _field(qtbot)
    qtbot.mousePress(host, Qt.LeftButton, pos=host.rect().center())
    qtbot.mouseRelease(host, Qt.LeftButton, pos=host.rect().center() + host.rect().center() / 2)
    qtbot.wait(ambient.CLICK_RIPPLE_SETTLE_MS * 3)
    assert not field.engine._popup_waves


def test_gravity_clicks_keep_their_burst_without_a_second_wave(qtbot):
    host, field = _field(qtbot, gravity_radius=0.4)
    qtbot.mouseClick(host, Qt.LeftButton, pos=host.rect().center())
    field._on_tick()
    field._timer.stop()
    assert field.engine._gravity_impulses
    assert not field.engine._gravity_ripple_impulses
    qtbot.waitUntil(lambda: len(field.engine._popup_waves) == 1)


def test_gpu_backend_and_dynamic_animation_preferences(store, qtbot, qt_theme_applied):
    assert prefs._flow_graphics_backend() == "gpu"
    assert prefs._dynamic_animation_enabled() is True
    prefs._set_flow_graphics_backend("bogus")
    assert prefs._flow_graphics_backend() == "gpu"
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    gpu = dialog.findChild(QSlider, "AnimationGpu")
    dynamic = dialog.findChild(Toggle, "DynamicAnimation")
    assert (gpu.minimum(), gpu.maximum(), gpu.value()) == (0, 2, 2)
    assert dynamic.isChecked()
    gpu.setValue(0)
    dynamic.setChecked(False)
    dialog.reject()
    assert prefs._flow_graphics_backend() == "gpu"
    accepted = prefs.PreferencesDialog()
    qtbot.addWidget(accepted)
    accepted.findChild(QSlider, "AnimationGpu").setValue(0)
    accepted.findChild(Toggle, "DynamicAnimation").setChecked(False)
    _save(accepted)
    assert prefs._flow_graphics_backend() == "cpu"
    assert prefs._dynamic_animation_enabled() is False


@pytest.mark.parametrize("key,expected", [
    ("measure", "cpu"), ("mask", "gpu"), ("train_cellpose", "gpu"),
    ("umap", "cpu"), ("loading", None), ("", None)])
def test_job_classification(key, expected):
    assert ambient._job_resource(key, True) == expected


def test_gpu_jobs_without_a_gpu_count_as_cpu_work():
    assert ambient._job_resource("mask", False) == "cpu"


def test_cpu_job_pauses_and_restores_animation(jobs, qtbot):
    registry, policy = jobs
    _host, field = _field(qtbot)
    assert field._should_run()
    job = registry.start("measure")
    assert policy.pauses_animation() and policy.blocks_gpu()
    assert not field._should_run() and not field.is_running()
    registry.finish(job)
    assert field._should_run() and field.is_running()
    assert not ambient._decorative_gpu_blocked()


def test_gpu_job_keeps_cpu_animation_but_blocks_decorative_gpu(jobs, qtbot):
    registry, policy = jobs
    _host, field = _field(qtbot)
    job = registry.start("mask")
    assert field._should_run()
    assert ambient._decorative_gpu_blocked()
    registry.finish(job)
    assert not ambient._decorative_gpu_blocked()


def test_overlapping_jobs_release_only_when_the_last_ends(jobs, qtbot):
    registry, policy = jobs
    _host, field = _field(qtbot)
    gpu = registry.start("mask")
    cpu = registry.start("measure")
    assert policy.pauses_animation()
    registry.finish(gpu)
    assert policy.pauses_animation()
    second = registry.start("measure")
    registry.finish(cpu)
    assert policy.pauses_animation()
    registry.finish(second)
    assert not policy.blocks_gpu() and field._should_run()


def test_failed_or_cancelled_jobs_release_through_unregistering(jobs):
    registry, policy = jobs
    failed = registry.start("measure")
    cancelled = registry.start("mask")
    registry.finish(failed)
    assert policy.blocks_gpu() and not policy.pauses_animation()
    registry.finish(cancelled)
    assert not policy.blocks_gpu()


def test_housekeeping_and_unknown_jobs_never_pause(jobs):
    registry, policy = jobs
    registry.start("measure", user_visible=False)
    registry.start("loading")
    assert not policy.blocks_gpu()


def test_policy_off_disables_workload_rules_and_saves_nothing(jobs, qtbot, store):
    registry, policy = jobs
    _host, field = _field(qtbot)
    before = {key: store.value(key) for key in store.allKeys()}
    registry.start("measure")
    policy.set_enabled(False)
    assert field._should_run() and not ambient._decorative_gpu_blocked()
    assert not ambient._dynamic_animation_on()
    policy.set_enabled(True)
    assert not field._should_run()
    after = {key: store.value(key) for key in store.allKeys()}
    assert after == before


def test_policy_pauses_only_fractals_it_must_and_resumes_only_those(jobs, monkeypatch):
    registry, policy = jobs

    class Fractal:
        def __init__(self, backend, paused=False):
            self.backend_name = backend
            self.paused = paused

        def is_paused(self):
            return self.paused

        def pause(self):
            self.paused = True
            return True

        def resume(self):
            self.paused = False
            return True

    gpu, cpu, already = Fractal("gpu"), Fractal("cpu"), Fractal("gpu", paused=True)
    monkeypatch.setattr(ambient, "_live_spaceout_fractals", lambda: [gpu, cpu, already])
    job = registry.start("mask")
    assert gpu.paused and not cpu.paused and already.paused
    registry.finish(job)
    assert not gpu.paused and already.paused
    job = registry.start("measure")
    assert gpu.paused and cpu.paused
    registry.finish(job)
    assert not gpu.paused and not cpu.paused and already.paused


def test_apply_preferences_reaches_the_policy(jobs, store, qapp):
    _registry, policy = jobs
    prefs._set_dynamic_animation_enabled(False)
    prefs.apply_ambient_preferences(qapp)
    assert policy.enabled is False
    prefs._set_dynamic_animation_enabled(True)
    prefs.apply_ambient_preferences(qapp)
    assert policy.enabled is True
