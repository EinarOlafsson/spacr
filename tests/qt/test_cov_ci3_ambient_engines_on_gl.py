"""N684: each GPU-capable theme hands its own frame data to the renderer.

``moderngl`` is the recording stand-in from
``test_cov_ci3_ambient_flow_renderer``; nothing touches a graphics driver.
The engines shade on a thread named like the producer, as they do in the
application, and the recorded draw calls are compared with the data the
engine computes for the same frame.
"""
import sys
import threading

import numpy as np
import pytest
from PySide6.QtGui import QImage

from spacr.qt.widgets import ambient
from tests.qt.test_cov_ci3_ambient_flow_renderer import _fake_moderngl


@pytest.fixture(autouse=True)
def _a_mac_gpu_is_probed(monkeypatch):
    """On macOS only a Radeon is probed for hardware OpenGL, so on an Apple
    Silicon Mac these real-context tests found no context at all. They are
    about the renderer, not that policy; an Apple M2 runs them."""
    from spacr.qt.widgets import ambient

    monkeypatch.setattr(ambient, "_mac_radeon_graphics", lambda: True)
    monkeypatch.setattr(ambient, "_FLOW_GRAPHICS_PROBE", None, raising=False)
    # Its fakes model the OpenGL 4.3 storage buffers, not macOS's 4.1 tables.
    monkeypatch.setattr(ambient, "_graphics_context_options",
                        lambda: {"standalone": True, "require": 430})


@pytest.fixture
def gpu(monkeypatch):
    module = _fake_moderngl()
    monkeypatch.setitem(sys.modules, "moderngl", module)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda stop_event=None: True)
    return module


def _make(theme, **kwargs):
    engine = ambient.make_engine(theme, ambient.default_palette_for(theme), "#191919",
                                 seed=684, **kwargs)
    engine.set_max_pixels(320 * 200)
    engine.set_time(9.5)
    engine._graphics_backend = "gpu"
    return engine


def _on_producer(function):
    box = {}

    def run():
        try:
            box["value"] = function()
        except BaseException as error:
            box["error"] = error

    thread = threading.Thread(target=run, name="spacr-ambient-shade")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _shade_and_record(engine, extra=None):
    def run():
        image = engine.shade(320, 200)
        renderer = engine._point_graphics
        context = renderer._context
        renders = list(context.renders)
        names = _programs(renderer, renders)
        detail = extra() if extra else None
        engine._release_point_graphics()
        return image, names, context, renders, detail
    return _on_producer(run)


def _programs(renderer, renders):
    names = {id(program): name for name, program in renderer.__dict__.get("_programs", {}).items()}
    names[id(renderer._program)] = "points"
    return [names[id(render.program)] for render in renders]


def test_blobs_send_one_instance_per_blob_and_release_on_the_producer(gpu, qapp):
    engine = _make("blobs")
    image, names, context, renders, rows = _shade_and_record(
        engine, lambda: engine._blob_rows(*engine.buffer_size(320, 200)))
    assert image is not None and not image.isNull()
    render, = renders
    assert names == ["blobs"]
    assert render.instances == len(rows) > 0
    sent = np.frombuffer(render.data, np.float32).reshape(-1, 7)
    assert sent == pytest.approx(np.asarray(rows, np.float32))
    assert context.context_released
    assert engine._point_graphics is None and not engine._graphics_failed


def test_stratified_stars_come_back_as_a_transparent_layer(gpu, qapp):
    engine = _make("drift")
    image, names, _context, renders, _ = _shade_and_record(engine)
    assert image.format() == QImage.Format_ARGB32_Premultiplied
    assert image.pixelColor(0, 0).alpha() == 0
    assert set(names) == {"discs"}
    assert sum(render.instances for render in renders) > 0


def test_aurora_generates_one_texture_per_curtain(gpu, qapp):
    engine = _make("aurora")
    layers = engine._aurora_layers(*engine.buffer_size(320, 200))
    image, names, _context, renders, _ = _shade_and_record(engine)
    assert image is not None
    assert names.count("aurora_texture") == names.count("strips") == len(layers) > 0


def test_growth_strokes_and_tips_reach_the_renderer(gpu, qapp):
    engine = _make("data_art_fungal_growth")
    engine.set_time(41.5)
    image, names, _context, renders, _ = _shade_and_record(engine)
    assert image is not None
    assert names[-1] == "discs"
    assert {"hairlines", "capsules"} & set(names)


def test_spinning_paper_tiles_are_placed_with_their_rotation(gpu, qapp):
    engine = _make("data_art_tissue_facets")
    engine.pointer = (0.5, 0.5)
    engine.gravity_radius = 0.6

    def two_frames():
        engine.shade(320, 200)
        renderer = engine._point_graphics
        first = len(renderer._context.renders)
        engine.set_time(9.6)
        image = engine.shade(320, 200)
        render = renderer._context.renders[-1]
        names = _programs(renderer, renderer._context.renders)
        engine._release_point_graphics()
        return first, image, render, names

    first, image, render, names = _on_producer(two_frames)
    assert first == 1 and names == ["tiles", "tiles"]
    rows = np.frombuffer(render.data, np.float32).reshape(-1, 9)
    assert (rows[:, 2] != 0.0).any() and (rows[:, 2] == 0.0).any()
    assert image is not None


def test_cpu_paper_paints_rotated_tiles_differently_from_resting_ones(qapp):
    def frames(active):
        engine = _make("data_art_tissue_facets")
        engine._graphics_backend = "cpu"
        engine.pointer = (0.5, 0.5) if active else None
        engine.gravity_radius = 0.6
        engine.shade(320, 200)
        engine.set_time(9.6)
        layout = engine._tissue_layout(*engine.buffer_size(320, 200))
        engine.set_time(9.7)
        return engine.shade(320, 200), layout

    spun, (_material, placements) = frames(True)
    still, _layout = frames(False)
    assert any(angle != 0.0 for *_rest, angle in placements)
    assert spun != still


def test_point_material_frames_still_flash_their_blinking_dots(gpu, qapp):
    engine = _make("data_art_impulse_lens", blink_percent=100.0, density=0.25)
    image, names, _context, renders, _ = _shade_and_record(engine)
    render, = renders
    assert names == ["subpixel_grains"]
    assert render.mode == "TRIANGLE_STRIP" and render.instances > 0
    points = np.floor(np.frombuffer(render.data, np.float32).reshape(-1, 3)).astype(np.int64)
    width, height = engine.buffer_size(320, 200)
    inside = points[(points[:, 0] >= 0) & (points[:, 0] < width)
                    & (points[:, 1] >= 0) & (points[:, 1] < height)]
    assert len(inside)
    assert (width, height) == (image.width(), image.height())
    pixels = np.frombuffer(image.constBits(), np.uint32,
                           count=width * height).reshape(height, width) & 0xFFFFFF
    lit = np.argwhere(pixels != 0)
    assert len(lit), "the black renderer frame must carry the white flashes"
    near = np.zeros((height, width), bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            xs, ys = inside[:, 0] + dx, inside[:, 1] + dy
            keep = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            near[ys[keep], xs[keep]] = True
    assert near[lit[:, 0], lit[:, 1]].all()
