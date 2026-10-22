"""N673 wgpu follow-up (2026-10-16): Vulkan, Metal and Direct3D 12 for True 3D.

The maintainer added wgpu to spaCR proper, so the True 3D view can draw
through Vulkan, Metal or Direct3D 12 on as many computers as possible. A
default install brings wgpu (Python 3.10+, x86_64/AMD64/aarch64/arm64);
an install without it is supported and draws with OpenGL or the CPU. The
Render backend list offers Auto, Vulkan, Metal, Direct3D 12, OpenGL and CPU,
only what the computer has; Auto prefers wgpu, then OpenGL, then the CPU.

The wgpu renderer (spacr.layers.WGPUVolumeRenderer) draws from the same rays
as the CPU and OpenGL renderers. Its pixel tests run only where a wgpu
adapter exists and the GPU may be used; they skip at a computer that hides
its GPU (CUDA_VISIBLE_DEVICES empty or SPACR_NO_GPU_RENDER set). Everything
else runs everywhere with mocked adapters and fake GPU contexts.
"""
from __future__ import annotations

import ast
import logging
import subprocess
import sys
import types
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from packaging.requirements import Requirement

from spacr import doctor, layers
from spacr.layers import (VOLUME_FINISHES, VOLUME_RENDERINGS, ImageLayer, LabelsLayer,
                          VolumeLens, VolumeRays, render_volume)
from spacr.qt.screens import make_masks as mm

from .test_673_napari_true3d import _spheres, _wait_render, screen, view
from .test_673_true3d_gpu import _gl_context

__all__ = ["screen", "view"]

ROOT = Path(__file__).resolve().parents[2]


def _core_dependencies():
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "dependencies" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError("setup.py has no dependencies list")


def _environment(python, machine, platform="linux"):
    return {"python_version": python, "python_full_version": f"{python}.0",
            "platform_machine": machine, "sys_platform": platform,
            "platform_system": {"linux": "Linux", "darwin": "Darwin", "win32": "Windows"}[platform],
            "os_name": "nt" if platform == "win32" else "posix",
            "implementation_name": "cpython", "platform_python_implementation": "CPython",
            "platform_release": "", "platform_version": "", "extra": ""}


def _wgpu_for(python, machine, platform="linux"):
    wanted = [Requirement(line) for line in _core_dependencies()
              if Requirement(line).name == "wgpu"]
    chosen = [req for req in wanted
              if req.marker is None or req.marker.evaluate(_environment(python, machine, platform))]
    assert len(chosen) <= 1, chosen
    return chosen[0] if chosen else None


class _FakeAdapter:
    def __init__(self, backend_type, adapter_type="DiscreteGPU", device="Fake GPU"):
        self.info = {"backend_type": backend_type, "adapter_type": adapter_type,
                     "device": device, "description": f"{device} driver"}
        self.limits = {"max-texture-dimension-3d": 2048}


class _FakeGPU:
    """A GPU context that draws on the CPU, so the chain runs without a GPU."""

    made = []

    def __init__(self, backend, fail=None):
        if fail:
            raise RuntimeError(fail)
        self.backend = backend
        self.limits = {"renderer": f"Fake {backend}", "max_3d": 2048}
        self.failing = None
        _FakeGPU.made.append(backend)

    def name(self):
        return self.limits["renderer"]

    def render(self, request):
        if self.failing:
            raise RuntimeError(self.failing)
        return mm._render_volume_view(request)

    def close(self):
        pass


def _chain(monkeypatch, wgpu_fail=None, gl_fail=None, offered=("auto", "vulkan", "opengl", "cpu")):
    _FakeGPU.made = []
    monkeypatch.setattr(mm, "volume_gpu_refusal", lambda: "")
    monkeypatch.setattr(mm, "volume_render_backends", lambda: tuple(offered))
    monkeypatch.setattr(mm, "_WGPUVolumeContext",
                        lambda backend, allow_software=True: _FakeGPU(backend, wgpu_fail))
    monkeypatch.setattr(mm, "_GLVolumeContext",
                        lambda allow_software=True: _FakeGPU("opengl", gl_fail))
    monkeypatch.setattr(sys, "platform", "linux")


def _restart(stage, qtbot):
    if stage._gl is not None:
        stage._gl.close()
    stage._gl = None
    for backend in ("cpu", "auto"):
        stage._backend_box.setCurrentIndex(stage._backend_box.findData(backend))
        _wait_render(stage, qtbot)


def test_setup_py_installs_wgpu_by_default_with_markers_per_python_and_cpu():
    assert _wgpu_for("3.9", "x86_64") is None
    for machine, platform in (("x86_64", "linux"), ("aarch64", "linux"), ("AMD64", "win32"),
                              ("ARM64", "win32"), ("arm64", "darwin"), ("x86_64", "darwin")):
        ten = _wgpu_for("3.10", machine, platform)
        assert ten is not None and ten.specifier.contains("0.31.0")
        assert not ten.specifier.contains("0.31.1") and not ten.specifier.contains("0.24.0")
        for python in ("3.11", "3.12", "3.13", "3.14"):
            later = _wgpu_for(python, machine, platform)
            assert later is not None and later.specifier.contains("0.32.0")
            assert not later.specifier.contains("0.33.0") and later.specifier.contains("0.25.0")
    for machine in ("ppc64le", "s390x", "riscv64", "i686"):
        assert _wgpu_for("3.12", machine) is None
    names = {Requirement(line).name for line in _core_dependencies()}
    assert "pygfx" not in names and "rendercanvas" not in names


def test_the_conda_recipe_bounds_wgpu_py_without_requiring_it():
    recipe = (ROOT / "conda-forge" / "recipe" / "recipe.yaml").read_text(encoding="utf-8")
    run, constraints = recipe.split("  run_constraints:", 1)
    assert "wgpu" not in run.split("  run:", 1)[1]
    assert "- wgpu-py >=0.25,<0.33" in constraints.split("\ntests:", 1)[0]


def test_wgpu_is_imported_only_when_the_view_asks():
    code = ("import sys, spacr.layers, spacr.doctor; spacr.layers.render_scale(50, 1000); "
            "print('wgpu' in sys.modules)")
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True, cwd=ROOT)
    assert found.stdout.strip().splitlines()[-1] == "False"


def test_adapters_are_listed_by_backend_and_ranked(monkeypatch):
    adapters = [_FakeAdapter("Vulkan", "CPU", "llvmpipe"),
                _FakeAdapter("Vulkan", "IntegratedGPU", "Iris"),
                _FakeAdapter("Vulkan", "DiscreteGPU", "RTX"),
                _FakeAdapter("OpenGL", "DiscreteGPU", "RTX GL"),
                _FakeAdapter("D3D12", "DiscreteGPU", "RTX D3D")]
    monkeypatch.setattr(layers, "_wgpu_enumerate", lambda: list(adapters))
    found = layers.wgpu_adapters(refresh=True)
    assert [info["backend"] for info in found] == ["vulkan", "vulkan", "vulkan", "d3d12"]
    assert layers.wgpu_adapter("vulkan") is adapters[2]
    assert layers.wgpu_adapter("d3d12") is adapters[4]
    monkeypatch.setattr(layers, "_wgpu_enumerate", lambda: [adapters[0]])
    assert layers.wgpu_adapter("vulkan", allow_software=True) is adapters[0]
    with pytest.raises(RuntimeError, match="hardware vulkan"):
        layers.wgpu_adapter("vulkan", allow_software=False)
    with pytest.raises(RuntimeError, match="metal"):
        layers.wgpu_adapter("metal")
    monkeypatch.setattr(layers, "_wgpu_enumerate", lambda: [])
    assert layers.wgpu_adapters(refresh=True) == ()
    layers.wgpu_adapters(refresh=True)


def test_the_backend_list_offers_only_what_the_computer_has(monkeypatch):
    monkeypatch.setattr(mm, "volume_gpu_refusal", lambda: "")
    monkeypatch.setattr(layers, "_wgpu_enumerate",
                        lambda: [_FakeAdapter("Vulkan"), _FakeAdapter("OpenGL")])
    layers.wgpu_adapters(refresh=True)
    expected = ("auto", "vulkan", "opengl", "cpu") if find_spec("vispy") else ("auto", "vulkan", "cpu")
    assert mm.volume_render_backends() == expected
    monkeypatch.setattr(layers, "_wgpu_enumerate",
                        lambda: [_FakeAdapter("Metal"), _FakeAdapter("D3D12")])
    layers.wgpu_adapters(refresh=True)
    assert mm.volume_render_backends()[:3] == ("auto", "metal", "d3d12")
    monkeypatch.setattr(layers, "_wgpu_enumerate", lambda: [])
    layers.wgpu_adapters(refresh=True)
    assert not set(mm.VOLUME_WGPU_BACKENDS) & set(mm.volume_render_backends())
    assert set(mm._VolumeStage._backend_names()) == set(mm.VOLUME_BACKENDS)
    layers.wgpu_adapters(refresh=True)


@pytest.mark.parametrize("variable, value, reason", [
    ("SPACR_NO_GPU_RENDER", "1", "SPACR_NO_GPU_RENDER is set"),
    ("CUDA_VISIBLE_DEVICES", "", "CUDA_VISIBLE_DEVICES is empty")])
def test_a_computer_that_hides_its_gpu_offers_auto_and_cpu_only(monkeypatch, variable, value,
                                                                reason):
    monkeypatch.delenv("SPACR_NO_GPU_RENDER", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert layers.gpu_render_refusal() == ""
    monkeypatch.setenv("SPACR_NO_GPU_RENDER", "0")
    assert layers.gpu_render_refusal() == ""
    monkeypatch.setenv(variable, value)
    assert layers.gpu_render_refusal() == reason
    assert mm.volume_render_backends() == ("auto", "cpu")
    assert mm.volume_backend_chain("auto") == ()


def test_auto_prefers_wgpu_by_platform_then_opengl():
    every = mm.VOLUME_BACKENDS
    assert mm.volume_backend_chain("auto", every, "linux") == ("vulkan", "opengl")
    assert mm.volume_backend_chain("auto", every, "win32") == ("vulkan", "d3d12", "opengl")
    assert mm.volume_backend_chain("auto", every, "darwin") == ("metal", "vulkan", "opengl")
    no_vulkan = ("auto", "d3d12", "opengl", "cpu")
    assert mm.volume_backend_chain("auto", no_vulkan, "win32") == ("d3d12", "opengl")
    assert mm.volume_backend_chain("auto", ("auto", "opengl", "cpu"), "linux") == ("opengl",)
    assert mm.volume_backend_chain("auto", ("auto", "cpu"), "darwin") == ()
    assert mm.volume_backend_chain("cpu", every, "linux") == ()
    assert mm.volume_backend_chain("metal", every, "linux") == ("metal",)
    assert mm.volume_backend_chain("metal", ("auto", "cpu"), "darwin") == ()


def test_auto_draws_with_wgpu_when_it_works(view, qtbot, monkeypatch):
    stage = view
    _chain(monkeypatch)
    _restart(stage, qtbot)
    assert stage._active_backend() == "vulkan" and _FakeGPU.made == ["vulkan"]
    assert stage._render_state.backend == "vulkan" and stage._render_state.rgb.any()
    assert "the GPU (Vulkan, Fake vulkan)" in stage._detail.text()
    assert stage._backend_note == ""


def test_auto_falls_back_from_wgpu_to_opengl_with_one_note(view, qtbot, monkeypatch, caplog):
    stage = view
    _chain(monkeypatch, wgpu_fail="no Vulkan driver")
    with caplog.at_level(logging.WARNING, logger="spacr.qt.make_masks"):
        _restart(stage, qtbot)
    assert stage._active_backend() == "opengl"
    notes = [record.getMessage() for record in caplog.records if "True 3D" in record.getMessage()]
    assert len(notes) == 1 and "\n" not in notes[0]
    assert "Vulkan is unavailable (no Vulkan driver)" in notes[0] and "OpenGL" in notes[0]
    assert "the GPU (OpenGL" in stage._detail.text() and "no Vulkan driver" in stage._detail.text()


def test_auto_falls_back_to_the_cpu_when_no_gpu_backend_works(view, qtbot, monkeypatch, caplog):
    stage = view
    _chain(monkeypatch, wgpu_fail="no Vulkan driver", gl_fail="no OpenGL context")
    with caplog.at_level(logging.WARNING, logger="spacr.qt.make_masks"):
        _restart(stage, qtbot)
    assert stage._active_backend() == "cpu" and stage._gl is None
    notes = [record.getMessage() for record in caplog.records if "True 3D" in record.getMessage()]
    assert len(notes) == 1 and "CPU" in notes[0] and "no Vulkan driver" in notes[0]
    assert stage._render_state.backend == "cpu" and stage._render_state.rgb.any()


def test_a_wgpu_frame_that_fails_hands_auto_to_opengl_then_the_cpu(view, qtbot, monkeypatch):
    stage = view
    _chain(monkeypatch)
    _restart(stage, qtbot)
    assert stage._gl.backend == "vulkan"
    stage._gl.failing = "device lost"
    assert stage._submit_render("preview")
    _wait_render(stage, qtbot)
    assert stage._gl.backend == "opengl" and stage._render_state.backend == "opengl"
    assert "Vulkan is unavailable (device lost)" in stage._backend_note
    stage._gl.failing = "GL_OUT_OF_MEMORY"
    assert stage._submit_render("preview")
    _wait_render(stage, qtbot)
    assert stage._gl is None and stage._render_state.backend == "cpu"
    assert "GL_OUT_OF_MEMORY" in stage._backend_note


def test_an_explicit_wgpu_choice_that_fails_draws_on_the_cpu(view, qtbot, monkeypatch):
    stage = view
    _chain(monkeypatch, wgpu_fail="no Vulkan driver")
    stage._backend_choice = "vulkan"
    stage._gl, stage._gl_tried, stage._gpu_refused = None, False, set()
    assert stage._active_backend() == "cpu"
    assert _FakeGPU.made == [] and "no Vulkan driver" in stage._backend_note


def test_the_gpu_turned_off_draws_on_the_cpu_with_a_note(view, qtbot, monkeypatch):
    stage = view
    _chain(monkeypatch)
    monkeypatch.setattr(mm, "volume_gpu_refusal", lambda: "SPACR_NO_GPU_RENDER is set")
    _restart(stage, qtbot)
    assert stage._active_backend() == "cpu" and _FakeGPU.made == []
    assert "turned off (SPACR_NO_GPU_RENDER is set)" in stage._detail.text()


def test_doctor_treats_a_missing_wgpu_as_the_supported_fallback(monkeypatch):
    real = doctor._distribution_version
    monkeypatch.setattr(doctor, "_distribution_version",
                        lambda name: None if name == "wgpu" else real(name))
    result = doctor.check_3d_backends(doctor.Context(probe_gpu=True))
    assert result.status == doctor.PASS and result.check == "3D backends"
    assert "wgpu is not installed" in result.message and "supported" in result.message
    assert any(line.startswith("CPU") for line in result.details)
    assert doctor.check_3d_backends in doctor.CHECKS


def test_doctor_lists_wgpu_adapters_and_warns_on_a_broken_wgpu(monkeypatch):
    monkeypatch.setattr(doctor, "_distribution_version",
                        lambda name: "0.31.0" if name == "wgpu" else None)
    monkeypatch.setitem(sys.modules, "wgpu", types.ModuleType("wgpu"))
    monkeypatch.setattr(layers, "gpu_render_refusal", lambda: "")
    monkeypatch.setattr(layers, "wgpu_adapters", lambda refresh=False: (
        {"backend": "vulkan", "name": "RTX", "adapter_type": "DiscreteGPU", "description": ""},))
    listed = doctor.check_3d_backends(doctor.Context(probe_gpu=True))
    assert listed.status == doctor.PASS and "(vulkan)" in listed.message
    assert "wgpu vulkan: RTX (DiscreteGPU)" in listed.details
    unprobed = doctor.check_3d_backends(doctor.Context(probe_gpu=False))
    assert unprobed.status == doctor.PASS and "adapters not listed" in unprobed.details[0]
    monkeypatch.setattr(layers, "wgpu_adapters", lambda refresh=False: ())
    none = doctor.check_3d_backends(doctor.Context(probe_gpu=True))
    assert none.status == doctor.PASS and "no adapter" in none.message
    monkeypatch.setitem(sys.modules, "wgpu", None)
    broken = doctor.check_3d_backends(doctor.Context(probe_gpu=True))
    assert broken.status == doctor.WARN and "pip uninstall -y wgpu rendercanvas" in broken.fix


def test_doctor_knows_who_a_default_install_gives_wgpu(monkeypatch):
    import platform

    monkeypatch.setattr(sys, "version_info", (3, 9, 18, "final", 0))
    assert doctor._wgpu_expected() == (False, "wgpu needs Python 3.10 or newer")
    monkeypatch.setattr(sys, "version_info", (3, 12, 4, "final", 0))
    monkeypatch.setattr(platform, "machine", lambda: "ppc64le")
    assert not doctor._wgpu_expected()[0]
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "libc_ver", lambda: ("glibc", "2.17"))
    assert "glibc 2.28" in doctor._wgpu_expected()[1]
    monkeypatch.setattr(platform, "libc_ver", lambda: ("glibc", "2.39"))
    assert doctor._wgpu_expected() == (True, "")


def test_doctor_3d_status_and_platform_reason_use_runtime_translation(monkeypatch):
    from spacr.qt import i18n

    monkeypatch.setattr(i18n, "tr", lambda text, **values: "translated: " + text.format(**values))
    monkeypatch.setattr(sys, "version_info", (3, 9, 18, "final", 0))
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: None)

    expected, reason = doctor._wgpu_expected()
    assert not expected and reason == "translated: wgpu needs Python 3.10 or newer"
    row = doctor.check_3d_backends(doctor.Context(probe_gpu=False))
    assert row.check == "translated: 3D backends"
    assert row.message.startswith("translated: wgpu is not installed")
    assert all(detail.startswith("translated: ") for detail in row.details)
    assert reason in row.details[0]


def _wgpu_context():
    """A real wgpu context, or skip: never where the GPU is hidden."""
    refusal = layers.gpu_render_refusal()
    if refusal:
        pytest.skip(f"GPU rendering is turned off here ({refusal})")
    if find_spec("wgpu") is None:
        pytest.skip("wgpu is not installed")
    found = layers.wgpu_adapters(refresh=True)
    if not found:
        pytest.skip("wgpu finds no Vulkan, Metal or Direct3D 12 adapter")
    return mm._WGPUVolumeContext(found[0]["backend"])


def _rays(shape=(30, 36, 30)):
    return VolumeRays(origin=(-4.031, -2.073, -2.117), across=(0.0, 0.1, 0.9),
                      down=(0.2, 0.9, 0.0), depth=(0.8, 0.1, 0.3), shape=shape)


def _layers(rendering="mip", labels_rendering="surface", finish="matte"):
    labels = _spheres()
    image = labels.astype(np.float32) * 100 + np.indices(labels.shape)[2]
    channel = ImageLayer(image.astype(np.float32), name="c", contrast_limits=[(0, 120)],
                         colormaps=["green"], metadata={"rendering": rendering})
    masks = LabelsLayer(labels, name="m", opacity=0.6,
                        metadata={"rendering": labels_rendering, "finish": finish})
    return channel, masks


@pytest.mark.parametrize("size", [None, (4, 7), (1, 2)])
def test_wgpu_reads_colour_and_picking_from_one_completed_frame(size):
    rays = VolumeRays(origin=(0, 0, 0), across=(0, 0, 1), down=(0, 1, 0),
                      depth=(1, 0, 0), shape=(2, 3, 2))
    height, width = size or (2, 3)
    colour = np.arange(height * width * 4, dtype=np.uint8).reshape(height, width, 4)
    codes = np.array([0, 3, 9, 65536, 2**31, 2**32-1], dtype=np.uint32).reshape(2, 3)
    packed = np.zeros((2, 3, 2), dtype=np.uint32)
    packed[..., 0] = codes
    buffers, copies, reads, events = {}, [], [], []

    def buffer(name, count, readable):
        buffers[name] = types.SimpleNamespace(name=name, data=bytearray(count + (64 if name == "readback" else 0)))
        return buffers[name]

    def dispatch(compute, name, params, resources, groups):
        if name == "clear":
            resources[1].data[:] = packed.tobytes()
        if name == "pack":
            resources[1].data[:] = colour.tobytes()

    class Encoder:
        def begin_compute_pass(self):
            return types.SimpleNamespace(end=lambda: events.append("end"))

        def copy_buffer_to_buffer(self, source, source_offset, target, target_offset, count):
            assert events == ["end"]
            copies.append((source.name, target.name, target_offset, count))
            target.data[target_offset:target_offset+count] = source.data[source_offset:source_offset+count]

        def finish(self):
            return self

    def read_buffer(resource, *, size=None):
        assert events[-1] == "submit"
        reads.append((resource.name, size))
        assert size == colour.nbytes + packed.nbytes
        return memoryview(resource.data)[:size]

    renderer = object.__new__(layers.WGPUVolumeRenderer)
    renderer._device = types.SimpleNamespace(
        create_command_encoder=Encoder,
        queue=types.SimpleNamespace(submit=lambda commands: events.append("submit"),
                                    read_buffer=read_buffer))
    renderer._buffer = buffer
    renderer._dispatch = dispatch
    renderer._forget_unused = lambda: None
    rgb, picked = renderer.render([], rays, size=size)
    assert reads == [("readback", colour.nbytes + packed.nbytes)]
    assert copies == [("picture", "readback", 0, colour.nbytes),
                      ("pick", "readback", colour.nbytes, packed.nbytes)]
    assert np.array_equal(rgb, colour[..., :3]) and rgb.dtype == np.uint8
    assert np.array_equal(picked, codes) and picked.dtype == np.int64


@pytest.mark.parametrize("readable", [False, True])
def test_wgpu_readable_buffers_can_receive_the_combined_gpu_copy(monkeypatch, readable):
    flags = types.SimpleNamespace(STORAGE=1, COPY_SRC=2, COPY_DST=4)
    monkeypatch.setitem(sys.modules, "wgpu", types.SimpleNamespace(BufferUsage=flags))
    allocated, destroyed = [], []

    def create_buffer(**kwargs):
        allocated.append(kwargs)
        return types.SimpleNamespace(destroy=lambda: destroyed.append(kwargs["size"]))

    renderer = object.__new__(layers.WGPUVolumeRenderer)
    renderer._buffers = {}
    renderer._textures, renderer._luts, renderer._pipelines = {}, {}, {}
    renderer._device = types.SimpleNamespace(create_buffer=create_buffer)
    name = "readback" if readable else "frame"
    held = renderer._buffer(name, 80, readable)
    assert renderer._buffer(name, 80, readable) is held
    assert renderer._buffer(name, 48, readable) is held
    grown = renderer._buffer(name, 160, readable)
    assert grown is not held and destroyed == [80]
    assert renderer._buffer(name, 100, readable) is grown
    usage = flags.STORAGE | (flags.COPY_SRC | flags.COPY_DST if readable else 0)
    assert allocated == [{"size": 80, "usage": usage}, {"size": 160, "usage": usage}]
    renderer.close()
    assert destroyed == [80, 160]
    assert renderer._buffers == {}
    renderer.close()
    assert destroyed == [80, 160]


def test_wgpu_frames_match_the_cpu_for_every_rendering_finish_and_lens():
    gpu = _wgpu_context()
    rays = _rays()
    lens = VolumeLens((6.0, 12.0, 14.0), (4.0, 4.0, 4.0), zoom=2.0)
    for rendering in VOLUME_RENDERINGS:
        channel, _masks = _layers(rendering=rendering)
        cpu, _ = render_volume([channel], rays)
        drawn, _ = gpu.renderer.render([channel], rays)
        assert np.abs(cpu * 255 - drawn).max() <= 1, rendering
    for labels_rendering in ("surface", "translucent"):
        for finish in VOLUME_FINISHES:
            for magnifier in (None, lens):
                channel, masks = _layers(labels_rendering=labels_rendering, finish=finish)
                cpu, cpu_picked = render_volume([channel, masks], rays, highlight=2,
                                                voxel_size=(2.0, 1.0, 1.0), lens=magnifier)
                drawn, picked = gpu.renderer.render(
                    [channel, masks], rays, highlight=2, voxel_size=(2.0, 1.0, 1.0),
                    lens=magnifier)
                assert np.abs(cpu * 255 - drawn).max() <= 1, (labels_rendering, finish)
                assert np.array_equal(cpu_picked, picked)
    channel, masks = _layers()
    big, _ = gpu.renderer.render([channel, masks], rays, size=(60, 72))
    assert big.shape == (60, 72, 3) and big.dtype == np.uint8
    for shape in ((15, 18, 20), (40, 44, 30), (8, 9, 12)):
        current = _rays(shape)
        cpu, cpu_picked = render_volume([channel, masks], current)
        drawn, picked = gpu.renderer.render([channel, masks], current)
        assert drawn.shape == (*shape[:2], 3)
        assert np.abs(cpu * 255 - drawn).max() <= 1
        assert np.array_equal(cpu_picked, picked)
    gpu.close()


def test_wgpu_and_opengl_frames_agree():
    gpu = _wgpu_context()
    try:
        gl = _gl_context()
        try:
            channel, masks = _layers(labels_rendering="surface", finish="glossy")
            with gl:
                opengl, gl_picked = gl.renderer.render([channel, masks], _rays(), highlight=1)
            drawn, picked = gpu.renderer.render([channel, masks], _rays(), highlight=1)
            assert np.abs(opengl.astype(int) - drawn.astype(int)).max() <= 3
            assert np.array_equal(gl_picked, picked)
        finally:
            gl.close()
    finally:
        gpu.close()
