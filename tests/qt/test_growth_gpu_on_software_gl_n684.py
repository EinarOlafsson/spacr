"""N684: growth's GPU path on a real (software) context, and visible refusals.

The 2026-10-10 workstation parity run (NVIDIA EGL) refused every growth
frame. The cause is not reproducible here: growth passes on Mesa llvmpipe
with both numpy 1.26/PySide6 6.10 and numpy 2.5/PySide6 6.11, moderngl
5.13. So every refusal now records and logs its reason, including the
failing function and line of a draw error, and the parity receipt carries
it per row. Vertex arrays also bind only the inputs a linker kept, so a
driver that drops the capsule shader's unused ``caps`` input cannot break
the draw (checked by making Mesa drop it).

The end-to-end check runs growth through the parity tool's producer-thread
route on Mesa's software device (never a hardware GPU) in a child process.
It is skipped where moderngl or Mesa EGL is unavailable.
"""
import json
import logging
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr.qt.widgets import ambient


@pytest.fixture(autouse=True)
def _a_mac_gpu_is_probed(monkeypatch):
    """Each refusal is named by the probe; on an Apple Silicon Mac the
    Radeon-only macOS policy answered first, with its own sentence."""
    from spacr.qt.widgets import ambient

    monkeypatch.setattr(ambient, "_mac_radeon_graphics", lambda: True)

ROOT = Path(__file__).resolve().parents[2]
MESA = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"


class _Program:
    def __init__(self, *names):
        self.names = names

    def __iter__(self):
        return iter(self.names)


def test_inputs_the_linker_removed_become_padding():
    layout, names = ambient._FlowPointGraphics._active_layout(
        _Program("segment", "half_width"), "4f 1f 1f/i",
        ("segment", "half_width", "caps"))
    assert (layout, names) == ("4f 1f 4x/i", ("segment", "half_width"))
    layout, names = ambient._FlowPointGraphics._active_layout(
        _Program("vertex"), "4f", ("vertex",))
    assert (layout, names) == ("4f", ("vertex",))


def test_vertex_array_binds_only_active_inputs():
    calls = []
    renderer = ambient._FlowPointGraphics.__new__(ambient._FlowPointGraphics)
    renderer._resources = []
    renderer._context = SimpleNamespace(
        vertex_array=lambda program, content: calls.append(content) or object())
    renderer._array(_Program("segment", "half_width"), object(), "4f 1f 1f/i",
                    "segment", "half_width", "caps")
    (buffer, layout, *names), = calls[0]
    assert layout == "4f 1f 4x/i" and names == ["segment", "half_width"]


@pytest.fixture
def producer_thread(monkeypatch):
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    monkeypatch.setattr(threading, "current_thread",
                        lambda: SimpleNamespace(name="spacr-ambient-shade"))


def test_a_draw_error_is_recorded_and_logged_once(producer_thread, monkeypatch, caplog, qapp):
    class Broken:
        def __init__(self):
            self._owner = threading.get_ident()

        def _draw_filaments(self, *args):
            raise KeyError("caps")

        def _close(self):
            pass

    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: True)
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)
    monkeypatch.setattr(ambient, "_FlowPointGraphics", Broken)
    monkeypatch.setattr(ambient, "_GRAPHICS_REFUSALS_LOGGED", set())
    engine = ambient.make_engine("data_art_fungal_growth", "spacr", "#191919", seed=3)
    engine.set_time(20.0)
    engine._graphics_backend = "gpu"
    with caplog.at_level(logging.WARNING, logger=ambient.LOG.name):
        assert engine._graphics_frame("_draw_filaments", 64, 64, build=lambda: ()) is None
        assert engine._graphics_frame("_draw_filaments", 64, 64, build=lambda: ()) is None
    assert engine._graphics_refusal.startswith("_draw_filaments failed in _draw_filaments line ")
    assert engine._graphics_refusal.endswith("KeyError: 'caps'")
    warnings = [r for r in caplog.records if "stays on the CPU" in r.getMessage()]
    assert len(warnings) == 1 and "KeyError" in warnings[0].getMessage()


@pytest.mark.parametrize("reason,patch", [
    ("probe failed", {"_flow_graphics_preflight": lambda: False}),
    ("another process", {"_flow_graphics_preflight": lambda: True,
                         "_flow_compute_idle": lambda: False}),
    ("Dynamic animation", {"_decorative_gpu_blocked": lambda: True}),
])
def test_each_refusal_names_its_reason(reason, patch, producer_thread, monkeypatch, qapp):
    for name, value in patch.items():
        monkeypatch.setattr(ambient, name, value)
    engine = ambient.make_engine("data_art_fungal_growth", "spacr", "#191919", seed=3)
    engine._graphics_backend = "gpu"
    assert engine._graphics_renderer(64, 64) is None
    assert reason in engine._graphics_refusal


def _software_device():
    """Index of Mesa's software EGL device, probed in children, or None."""
    if not os.path.exists(MESA):
        return None
    try:
        import moderngl  # noqa: F401
    except ImportError:
        return None
    env = dict(os.environ, __EGL_VENDOR_LIBRARY_FILENAMES=MESA, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(sys.path))
    for index in range(4):
        script = ("import moderngl; c = moderngl.create_context(standalone=True, "
                  f"require=430, backend='egl', device_index={index}); "
                  "print(c.info['GL_RENDERER'])")
        try:
            result = subprocess.run([sys.executable, "-c", script], env=env,
                                    capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0 and "llvmpipe" in result.stdout.lower():
            return index
    return None


CHILD = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {root!r})
    sys.path.insert(0, {tools!r})
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    from spacr.qt.widgets import ambient
    import verify_animation_gpu_parity as tool
    if {drop_caps!r}:
        vertex, fragment = ambient._GL_SHADERS['capsules']
        vertex = vertex.replace('capped = caps;', 'capped = 0.0;')
        assert 'capped = 0.0;' in vertex
        ambient._GL_SHADERS['capsules'] = (vertex, fragment)
    options = ambient._graphics_context_options()
    options['device_index'] = {device}
    ambient._graphics_context_options = lambda: dict(options)
    ambient._flow_graphics_preflight = lambda: True
    ambient._flow_compute_idle = lambda: True
    ambient._RESOURCE_POLICY = None
    tool._on_producer = tool._Producer()
    out = []
    for background in ('#191919', '#f2f2f2'):
        size = (640, 360)
        cpu, gpu = tool._engines(ambient, 'data_art_fungal_growth', background, 1.0, size)
        for engine in (cpu, gpu):
            engine.set_time(41.5)
        cpu_image, _, _ = tool._frame(cpu, size)
        gpu_image, _, used = tool._frame(gpu, size)
        dropped = tool._on_producer(
            lambda: 'caps' not in gpu._point_graphics._shader('capsules')) if used else None
        pair = tool._compare(cpu_image, gpu_image)
        pair.pop('diff', None)
        out.append(dict(used=used, failed=gpu._graphics_failed,
                        refusal=gpu._graphics_refusal, dropped=dropped, **pair))
        tool._on_producer(gpu._release_point_graphics)
    tool._on_producer.stop()
    print(json.dumps(out))
""")


@pytest.mark.parametrize("drop_caps", [False, True])
def test_growth_draws_on_software_gl_through_the_producer_route(drop_caps):
    device = _software_device()
    if device is None:
        pytest.skip("needs moderngl and Mesa's software EGL device")
    env = dict(os.environ, __EGL_VENDOR_LIBRARY_FILENAMES=MESA, CUDA_VISIBLE_DEVICES="",
               LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe", LP_NUM_THREADS="4",
               QT_QPA_PLATFORM="offscreen", PYTHONPATH=os.pathsep.join(sys.path))
    script = CHILD.format(root=str(ROOT), tools=str(ROOT / "tools"), device=device,
                          drop_caps=drop_caps)
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    rows = json.loads(result.stdout.strip().splitlines()[-1])
    for row in rows:
        assert row["used"] and not row["failed"] and row["refusal"] is None, row
        assert row["dropped"] is drop_caps
        assert row["mean"] <= 1.0 and row["p99"] <= 12 and row["over_48"] <= 0.005, row
