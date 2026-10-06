"""Flow backdrops draw bounded moving filaments and poll only local pointers."""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt.widgets import ambient as amb  # noqa: E402


FAMILIES = ("cytoplasm", "synapse", "wind", "atlas", "helix",
            "chromatin", "nebula", "silk")
DARK = "#101418"
LIGHT = "#f6f7f9"


def _engine(family: str, *, mouse: bool = False, background: str = DARK):
    """Construct one stable flow family with the ordinary shipped controls."""
    suffix = "_mouse" if mouse else ""
    return amb.make_engine(f"flow_{family}{suffix}", "ocean", background,
                           seed=27)


def _frame_bytes(image: QImage) -> bytes:
    """Read an owned QImage without relying on its Python object identity."""
    return bytes(image.constBits())


def test_all_sixteen_keys_have_a_real_engine_and_supported_palettes():
    """Every menu choice resolves and shares its family palette contract."""
    for family in FAMILIES:
        base = f"flow_{family}"
        mouse = f"{base}_mouse"
        assert base in amb.AMBIENT_THEMES and mouse in amb.AMBIENT_THEMES
        assert amb.palettes_for(base) == amb.palettes_for(mouse)
        assert {"spacr", "ocean", "lowsun", "midnight", "dusk",
                "deepwater"}.issubset(amb.palettes_for(base))
        for key in (base, mouse):
            engine = amb.make_engine(key, "ocean", DARK, seed=27)
            assert engine.name == key
            assert engine.mouse_enabled == key.endswith("_mouse")


def test_eight_families_have_distinct_bounded_geometry_and_shaded_frames():
    """The eight choices are drawings, not recolours of one producer."""
    geometry = {}
    frames = {}
    for family in FAMILIES:
        engine = _engine(family)
        engine.set_time(4.5)
        assert engine.buffer_size(3840, 2160) == (960, 540)
        assert engine.buffer_size(7680, 4320) == (960, 540)
        paths = engine.geometry(320, 180)
        assert 16 <= len(paths) <= 96
        assert all(4 <= len(path) <= 98 and len(path) % 2 == 0
                   for path in paths)
        assert all(math.isfinite(value) for path in paths for value in path)
        geometry[family] = paths
        first = engine.shade(640, 360)
        assert first is not None and first.size() == engine._buffer.size()
        frames[family] = _frame_bytes(first)
        assert len(set(frames[family])) > 16
    assert len({repr(paths) for paths in geometry.values()}) == 8
    assert len(set(frames.values())) == 8


@pytest.mark.parametrize("family", FAMILIES)
def test_clock_and_seed_reproduce_each_flow_and_advance_its_trace(family):
    """A frame is a stable clock function with genuine moving light."""
    stepped = _engine(family)
    jumped = _engine(family)
    for _ in range(12):
        stepped.advance(0.25)
    jumped.set_time(3.0)
    assert stepped.geometry(320, 180) == jumped.geometry(320, 180)
    assert _frame_bytes(stepped.shade(320, 180)) == _frame_bytes(
        jumped.shade(320, 180))
    stepped.set_time(5.0)
    assert _frame_bytes(stepped.shade(320, 180)) != _frame_bytes(
        jumped.shade(320, 180))


@pytest.mark.parametrize("family", FAMILIES)
def test_pointer_bends_only_its_mouse_variant_at_a_fixed_clock(family):
    """A cursor changes the local field without steering ordinary themes."""
    base = _engine(family)
    mouse = _engine(family, mouse=True)
    base.set_time(2.0)
    mouse.set_time(2.0)
    original = base.geometry(320, 180)
    assert mouse.geometry(320, 180) == original
    base.set_pointer((0.5, 0.5))
    mouse.set_pointer((0.5, 0.5))
    assert base.geometry(320, 180) == original
    bent = mouse.geometry(320, 180)
    assert bent != original
    assert any(abs(x - y) > 0.5 for path_a, path_b in zip(original, bent)
               for x, y in zip(path_a, path_b))
    mouse.set_pointer(None)
    assert mouse.geometry(320, 180) == original


@pytest.mark.parametrize("background", (DARK, LIGHT))
def test_flow_filaments_are_visible_over_dark_and_light_pages(background):
    """The compositing mode leaves a soft but visible field on either page."""
    for family in FAMILIES:
        image = _engine(family, background=background).shade(320, 180)
        assert image is not None
        identity = QColor("#000000" if background == DARK else "#ffffff")
        assert any(image.pixelColor(x, y) != identity
                   for y in range(0, image.height(), 5)
                   for x in range(0, image.width(), 5)), family


def test_local_cursor_polling_requires_the_active_window_and_widget(qtbot,
                                                                     monkeypatch):
    """A dialog or another window under the pointer cannot steer the flow."""
    host = QWidget()
    host.resize(320, 200)
    qtbot.addWidget(host)
    flow = amb.AmbientWidget(host, theme="flow_wind_mouse", palette="ocean",
                             background=DARK, seed=27)
    flow.setGeometry(host.rect())
    host.show()
    qtbot.waitExposed(host)
    flow.show()
    flow._animating = True
    centre = flow.mapToGlobal(QPoint(160, 100))
    state = {"active": host, "hovered": flow, "cursor": centre}
    monkeypatch.setattr(amb, "QApplication", SimpleNamespace(
        activeWindow=lambda: state["active"],
        widgetAt=lambda _point: state["hovered"]))
    monkeypatch.setattr(amb, "QCursor", SimpleNamespace(
        pos=lambda: state["cursor"]))
    assert flow._flow_pointer_for_tick() == pytest.approx(
        ((160.5 / 320), (100.5 / 200)))
    other = QWidget()
    qtbot.addWidget(other)
    state["active"] = other
    assert flow._flow_pointer_for_tick() is None
    state["active"] = host
    state["hovered"] = other
    assert flow._flow_pointer_for_tick() is None
    state["hovered"] = flow
    state["cursor"] = flow.mapToGlobal(QPoint(-1, 100))
    assert flow._flow_pointer_for_tick() is None
    state["cursor"] = centre
    flow._animating = False
    assert flow._flow_pointer_for_tick() is None
    flow.stop()


def test_tick_skips_cursor_for_ordinary_flow_and_never_waits_for_shading(
        qtbot, monkeypatch):
    """Cursor work is opt-in and a contended producer lock defers the tick."""
    flow = amb.AmbientWidget(theme="flow_wind", palette="ocean",
                             background=DARK, seed=27)
    qtbot.addWidget(flow)
    flow.stop()
    monkeypatch.setattr(flow, "_flow_pointer_for_tick", lambda: pytest.fail(
        "ordinary flow polled the cursor"))
    flow._on_tick()

    flow.set_theme("flow_wind_mouse")
    monkeypatch.setattr(flow, "_flow_pointer_for_tick", lambda: (0.5, 0.5))
    held = threading.Event()
    release = threading.Event()

    def occupy_lock():
        """Hold the producer lock in another thread for exactly one tick."""
        with flow._engine_lock:
            held.set()
            release.wait(5.0)

    thread = threading.Thread(target=occupy_lock)
    thread.start()
    try:
        assert held.wait(2.0)
        previous = flow.time()
        flow._on_tick()
        assert flow.time() == previous
        assert flow._engine.pointer is None
    finally:
        release.set()
        thread.join(2.0)
    assert not thread.is_alive()
    flow._on_tick()
    assert flow.time() > previous
    assert flow._engine.pointer == (0.5, 0.5)
