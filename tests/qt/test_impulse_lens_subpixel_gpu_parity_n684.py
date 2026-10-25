"""N684: the spaCR field's GPU frames match its CPU frames with ripples in flight.

The 2026-10-12 workstation parity run (nightly 36f4f5fd2) failed six
impulse-lens rows, all with ripples in flight: dark background at density
0.25 (mean 0.68-0.72, p99 33-34) and 1.0 (mean 2.72-2.86, p99 70-71),
light at 1.0 (mean 0.89-0.94, p99 19-20). The ripple constants were not
the cause: displacement and brightness are computed once on the CPU for
both paths. The cause was the stamp. Since 35a44a19a the CPU stamps each
field grain at its fractional centre with a constant-energy 4x4 footprint
(``_scatter_subpixel_grains``), while the renderer still drew the
whole-pixel 3x3 grain at the floored centre. Grains at rest sit on pixel
centres, where both stamps agree; a ripple moves them between pixels.

The renderer now draws the same footprint (``_draw_subpixel``, the
``subpixel_grains`` shader). The first tests use the recording moderngl
stand-in (no driver); the parity test runs the real renderer on Mesa's
software device (llvmpipe, CPU only) in a child process, through the
parity tool's producer-thread route, and is skipped without moderngl or
Mesa EGL. The workstation repeats it on the hardware GPU.
"""
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QImage

from spacr.qt.widgets import ambient
from tests.qt.test_cov_ci3_ambient_flow_renderer import _fake_moderngl
from tests.qt.test_growth_gpu_on_software_gl_n684 import MESA, _software_device


@pytest.fixture(autouse=True)
def _the_opengl_43_path(monkeypatch):
    """These fakes model the OpenGL 4.3 storage-buffer path. On macOS the
    context options ask for 4.1 (lookup textures instead) and only Radeon
    is probed, so without this pin a Mac ran a path the fakes do not have."""
    from spacr.qt.widgets import ambient

    monkeypatch.setattr(ambient, "_graphics_context_options",
                        lambda: {"standalone": True, "require": 430})
    monkeypatch.setattr(ambient, "_mac_radeon_graphics", lambda: True)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.setitem(sys.modules, "moderngl", _fake_moderngl())
    made = ambient._FlowPointGraphics()
    yield made
    made._close()


@pytest.mark.parametrize("dark,equation,clear", [(True, "MAX", 0.0), (False, "MIN", 1.0)])
def test_subpixel_grains_draw_one_footprint_quad_per_grain(renderer, dark, equation, clear):
    context = renderer._context
    lookup = np.arange(256, dtype=np.uint32) | np.uint32(0xFF000000)
    xs = np.array([3.5, 4.25, -1.5], dtype=np.float32)
    ys = np.array([2.5, 1.75, 0.5], dtype=np.float32)
    values = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    image = renderer._draw_subpixel(8, 4, xs, ys, values, lookup, dark)
    assert (image.width(), image.height(), image.format()) == (8, 4, QImage.Format_RGB32)
    assert renderer._framebuffer.colour == (clear, clear, clear, 1.0)
    render = context.renders[-1]
    assert (render.mode, render.vertices, render.instances) == ("TRIANGLE_STRIP", 4, 3)
    assert render.blend[0] == equation and render.blending
    assert render.layout == ("3f/i", "grain")
    sent = np.frombuffer(render.data, np.float32).reshape(-1, 3)
    assert sent.tolist() == np.stack((xs, ys, values), axis=1).tolist()
    program = renderer._programs["subpixel_grains"]
    assert render.program is program and program["canvas"].value == (8.0, 4.0)
    assert np.frombuffer(context.bindings[2].data, np.uint32).tolist() == lookup.tolist()


def test_an_empty_frame_is_a_cleared_target(renderer):
    empty = np.zeros(0, dtype=np.float32)
    lookup = np.full(256, 0xFFFFFFFF, dtype=np.uint32)
    image = renderer._draw_subpixel(4, 2, empty, empty, empty, lookup, False)
    assert renderer._context.renders[-1].instances == 0
    assert image.pixel(0, 0) & 0xFFFFFF == 0xFFFFFF


def test_the_shader_weighs_grains_with_the_cpu_profile():
    vertex, fragment = ambient._GL_SHADERS["subpixel_grains"]
    for source in (vertex, fragment):
        assert ambient._GL_GRAIN_WEIGHT in source
    assert "4.68" in vertex and f"{ambient._GRAIN_ENERGY:.2f}" == "4.68"
    knots = ("1.0 - 0.32 * r", "0.68 + (r - 1.0) * (0.24 - 0.68)",
             "0.24 + (r - root2) * (0.0 - 0.24)", "r < 2.0")
    assert all(knot in ambient._GL_GRAIN_WEIGHT for knot in knots)
    assert ambient._GRAIN_KNOTS[1:3] == (1.0, 2 ** 0.5)
    assert ambient._GRAIN_WEIGHTS == (1.0, 0.68, 0.24, 0.0)


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "moderngl", _fake_moderngl())
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: True)


def _shade_on_producer(engine):
    box = {}

    def run():
        box["image"] = engine.shade(320, 200)
        renderer = engine._point_graphics
        box["renders"] = list(renderer._context.renders)
        box["programs"] = {id(program): name for name, program
                           in renderer.__dict__.get("_programs", {}).items()}
        box["programs"][id(renderer._program)] = "points"
        engine._release_point_graphics()

    thread = threading.Thread(target=run, name="spacr-ambient-shade")
    thread.start()
    thread.join()
    return box


def test_whole_pixel_point_themes_keep_the_point_program(fake_gpu, qapp):
    engine = ambient.make_engine("data_art_point_atlas", "spacr", "#191919", seed=684)
    engine.set_max_pixels(320 * 200)
    engine._graphics_backend = "gpu"
    engine.set_time(9.5)
    box = _shade_on_producer(engine)
    names = [box["programs"][id(render.program)] for render in box["renders"]]
    assert names == ["points"] and box["renders"][0].mode == "POINTS"


def test_the_field_sends_fractional_grains_to_the_renderer(fake_gpu, qapp):
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#191919",
                                 seed=684, density=0.25)
    engine.set_max_pixels(320 * 200)
    engine._graphics_backend = "gpu"
    engine.set_time(9.1)
    engine._add_ripple((0.5, 0.5))
    engine.set_time(9.5)
    box = _shade_on_producer(engine)
    render, = box["renders"]
    assert box["programs"][id(render.program)] == "subpixel_grains"
    assert render.instances > 0 and not engine._graphics_failed
    grains = np.frombuffer(render.data, np.float32).reshape(-1, 3)
    width, height = engine.buffer_size(320, 200)
    assert (grains[:, 0] > -2).all() and (grains[:, 0] < width + 2).all()
    assert (grains[:, 1] > -2).all() and (grains[:, 1] < height + 2).all()
    centres = np.abs(grains[:, :2] - np.floor(grains[:, :2]) - 0.5)
    assert (centres > 1e-3).any(), "a ripple in flight moves grains off pixel centres"
    assert box["image"] is not None and box["image"].width() == width


def test_a_refused_renderer_keeps_the_cpu_subpixel_frame(monkeypatch, qapp):
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#f2f2f2",
                                 seed=684, density=0.25)
    engine.set_max_pixels(320 * 200)
    engine.set_time(9.5)
    expected = engine.shade(320, 200).copy()
    engine._graphics_backend = "gpu"
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: False)
    monkeypatch.setattr(threading, "current_thread",
                        lambda: SimpleNamespace(name="spacr-ambient-shade"))
    assert engine.shade(320, 200) == expected
    assert engine._graphics_failed and "probe failed" in engine._graphics_refusal


CHILD = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {root!r})
    sys.path.insert(0, {tools!r})
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    from spacr.qt.widgets import ambient
    import verify_animation_gpu_parity as tool
    options = ambient._graphics_context_options()
    options['device_index'] = {device}
    ambient._graphics_context_options = lambda: dict(options)
    ambient._flow_graphics_preflight = lambda: True
    ambient._flow_compute_idle = lambda: True
    ambient._RESOURCE_POLICY = None
    tool._on_producer = tool._Producer()
    size = (640, 360)
    out = []
    for background in ('#191919', '#f2f2f2'):
        for density in (0.25, 1.0):
            cpu, gpu = tool._engines(ambient, 'data_art_impulse_lens', background,
                                     density, size)
            still, _unused = tool._engines(ambient, 'data_art_impulse_lens', background,
                                           density, size)
            for moment in (0.0, 7.25, 41.5):
                still.set_time(moment)
                for engine in (cpu, gpu):
                    engine.set_time(moment)
                    tool._start_ripples(engine, moment)
                cpu_image, _, _ = tool._frame(cpu, size)
                gpu_image, _, used = tool._frame(gpu, size)
                still_image, _, _ = tool._frame(still, size)
                subpixel = tool._on_producer(
                    lambda: 'subpixel_grains' in gpu._point_graphics._programs) if used else False
                pair = tool._compare(cpu_image, gpu_image)
                pair.pop('diff', None)
                moved = tool._compare(cpu_image, still_image)
                out.append(dict(background=background, density=density, moment=moment,
                                used=used, failed=gpu._graphics_failed,
                                refusal=gpu._graphics_refusal, subpixel=subpixel,
                                ripple_mean=moved['mean'], **pair))
            tool._on_producer(gpu._release_point_graphics)
    tool._on_producer.stop()
    print(json.dumps(out))
""")


def test_rippling_field_matches_on_the_software_device():
    device = _software_device()
    if device is None:
        pytest.skip("needs moderngl and Mesa's software EGL device")
    env = dict(os.environ, __EGL_VENDOR_LIBRARY_FILENAMES=MESA, CUDA_VISIBLE_DEVICES="",
               LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe", LP_NUM_THREADS="4",
               QT_QPA_PLATFORM="offscreen", PYTHONPATH=os.pathsep.join(sys.path))
    script = CHILD.format(root=str(ROOT), tools=str(ROOT / "tools"), device=device)
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    rows = json.loads(result.stdout.strip().splitlines()[-1])
    assert {(row["background"], row["density"]) for row in rows} == {
        ("#191919", 0.25), ("#191919", 1.0), ("#f2f2f2", 0.25), ("#f2f2f2", 1.0)}
    for row in rows:
        assert row["used"] and row["subpixel"] and not row["failed"], row
        assert row["refusal"] is None, row
        assert row["ripple_mean"] > 0.0, row
        assert row["mean"] <= 1.0 and row["p99"] <= 12 and row["over_48"] <= 0.005, row
