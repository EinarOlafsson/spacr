"""Unchanged controls and missing optional state preserve real ready pixels."""

import weakref

import pytest

from spacr.qt import preferences
from spacr.qt.widgets import ambient


def _pixels(engine):
    image = engine.shade(240, 160)
    return image.constBits().tobytes()


@pytest.mark.parametrize("setting,hook,changed", [
    ("blur", "_reblur", 1.5),
    ("resolution", "_reresolve", 0.75),
    ("density", "_redensify", 0.5),
    ("size", "_resize", 1.5),
])
def test_equal_engine_control_preserves_cached_frame_then_changed_control_invalidates(
        qapp, monkeypatch, setting, hook, changed):
    engine = ambient.make_engine("blobs", "spacr", "#101418", seed=7)
    engine.set_time(4)
    before = _pixels(engine)
    cached = engine._buffer
    calls = []
    original = getattr(engine, hook)

    def invalidate():
        calls.append(True)
        original()

    monkeypatch.setattr(engine, hook, invalidate)
    setter = getattr(engine, "set_" + setting)
    initial = getattr(engine, setting)
    setter(initial)
    assert not calls and engine._buffer is cached
    assert _pixels(engine) == before
    setter(changed)
    assert calls == [True] and getattr(engine, setting) == changed
    reference = ambient.make_engine("blobs", "spacr", "#101418", seed=7)
    reference.set_time(4)
    getattr(reference, "set_" + setting)(changed)
    assert _pixels(engine) == (
        _pixels(reference))


def test_retained_resonance_engine_drops_dimension_and_size_dependent_plate_caches(qapp):
    engine = ambient.ResonanceEngine(ambient.PALETTE_SETS["spacr"].colors,
                                     "#101418", seed=7)
    engine.set_time(4)
    engine.shade(240, 160)
    assert engine._floor is not None and engine._canvas is not None
    assert engine._surface is not None
    engine.set_resolution(0.5)
    assert engine._floor is None and engine._canvas is None and engine._surface is None
    assert engine._buffer is None
    engine.shade(240, 160)
    first_floor = engine._floor
    assert first_floor is not None
    engine.set_size(1.5)
    assert engine._floor is None
    actual = _pixels(engine)
    reference = ambient.ResonanceEngine(ambient.PALETTE_SETS["spacr"].colors,
                                        "#101418", seed=7, resolution=0.5, size=1.5)
    reference.set_time(4)
    assert actual == _pixels(reference)
    assert engine._floor is not first_floor


def test_unreadable_optional_gravity_preference_disables_mouse_capture(qtbot, monkeypatch):
    def unreadable():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(preferences, "_ambient_gravity_radius", unreadable)
    assert ambient._preferred_gravity_radius() == 0
    widget = ambient.AmbientWidget(theme="data_art_impulse_lens")
    qtbot.addWidget(widget)
    assert widget.gravity_radius() == 0
    assert widget._interaction_app is None
    monkeypatch.setattr(preferences, "_ambient_gravity_radius", lambda: 0.37)
    assert ambient._preferred_gravity_radius() == 0.37


def test_optional_gravity_on_noninteractive_engine_keeps_pixels_and_has_no_input_queue(qtbot):
    widget = ambient.AmbientWidget(theme="blobs", seed=7, gravity_radius=0)
    qtbot.addWidget(widget)
    widget.set_time(4)
    before = _pixels(widget.engine)
    assert widget._art_input is None and not hasattr(widget.engine, "set_gravity_radius")
    widget.set_gravity_radius(0.5)
    assert widget.gravity_radius() == 0.5 and widget._interaction_app is None
    assert _pixels(widget.engine) == before
    widget.set_gravity_radius(0)
    assert widget.gravity_radius() == 0 and not widget._pending_art_impulses


def test_dead_application_reference_cleanup_discards_unconsumed_clicks(qtbot):
    widget = ambient.AmbientWidget(theme="data_art_impulse_lens", gravity_radius=0.5)
    qtbot.addWidget(widget)
    widget.resize(240, 160)
    widget.show()
    assert widget._interaction_app is not None and widget._interaction_app() is not None
    widget.stop()
    assert widget._interaction_app is None

    class FormerApplication:
        pass

    former = FormerApplication()
    expired = weakref.ref(former)
    del former
    assert expired() is None
    widget._interaction_app = expired
    widget._pending_art_impulses.append((0.5, 0.5))
    widget._sync_interaction_filter()
    assert widget._interaction_app is None and not widget._pending_art_impulses
    assert not widget.is_running()


def test_stopping_legacy_animation_discards_pending_input_while_worker_is_busy(qtbot):
    import threading

    widget = ambient.AmbientWidget(theme="blobs", seed=7, gravity_radius=0)
    qtbot.addWidget(widget)
    widget.stop()
    owned = _pixels(widget.engine)
    locked, release = threading.Event(), threading.Event()

    def shade_in_progress():
        with widget._engine_lock:
            locked.set()
            release.wait(5)

    worker = threading.Thread(target=shade_in_progress)
    worker.start()
    try:
        assert locked.wait(2)
        widget._legacy_input._offer(0.5, None, ())
        widget._pending_dt = 0.5
        widget.stop()
        assert not release.is_set()
        assert widget.engine.time == 0
        assert widget._pending_dt == 0
        assert widget._legacy_input._snapshot[1] == 0
        assert widget._last_frame is None and not widget.is_running()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert _pixels(widget.engine) == owned
