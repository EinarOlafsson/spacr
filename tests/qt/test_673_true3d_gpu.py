"""N673 speed follow-up (2026-10-16): the True 3D view on the GPU.

The maintainer's review of the napari-like True 3D: "much better, now
speed". Turning, zooming and panning must feel instant. The view now draws
on the GPU (VisPy gloo ray-marching, OpenGL) with the CPU renderer as the
fallback; a Render backend list (Auto, OpenGL, CPU) and a Render scale
slider (1-100 %) replace "Render at full resolution"; the mask surfaces
have a Finish (Matte, Satin, Glossy, Metallic); and the magnifier works in
3D as a cube that magnifies what is inside it and dims the rest.

GPU tests run only where an OpenGL context can be made offscreen and are
skipped elsewhere. The GPU renderer draws from the same rays as the CPU
one, so its frames are compared with the CPU's pixel for pixel.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from spacr import layers
from spacr.layers import (CUBE_EDGES, GPU_MAX_VOXELS, RENDER_DRAG_VOXELS,
                          RENDER_SCALE_MAX_PIXEL, SURFACE_FINISHES, VOLUME_FINISHES,
                          VOLUME_RENDERINGS, ImageLayer, LabelsLayer, LayerError,
                          SurfaceLight, VolumeLens, VolumeRays, render_scale,
                          render_volume, shade_surface, surface_finish)
from spacr.qt import preferences
from spacr.qt.screens import make_masks as mm

from .test_673_napari_true3d import _settle, _spheres, _wait_render, screen, view

__all__ = ["screen", "view"]


def _gl_context():
    """An offscreen OpenGL context for the GPU renderer, or skip the test.

    2026-10-16 (wgpu follow-up): never where the GPU is hidden
    (CUDA_VISIBLE_DEVICES empty or SPACR_NO_GPU_RENDER set), as at a
    computer whose GPU another job owns.
    """
    refusal = layers.gpu_render_refusal()
    if refusal:
        pytest.skip(f"GPU rendering is turned off here ({refusal})")
    try:
        return mm._GLVolumeContext()
    except RuntimeError as exc:
        if str(exc) not in {
            "VisPy is not installed",
            "no OpenGL context could be created",
            "only OpenGL ES is available, and the shaders need desktop OpenGL",
            "no offscreen OpenGL surface could be created",
            "the OpenGL context could not be made current",
        }:
            raise
        pytest.skip(f"no OpenGL context offscreen: {exc}")


def _rays(shape=(30, 36, 30)):
    """Oblique rays that never land on a voxel boundary exactly."""
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


def _use(stage, backend, qtbot):
    stage._backend_box.setCurrentIndex(stage._backend_box.findData(backend))
    _wait_render(stage, qtbot)


def test_the_render_scale_slider_maps_to_voxels_pixels_and_samples():
    voxels = 64 * 256 * 256
    full = render_scale(100, voxels)
    assert full.max_voxels is None and full.pixel == 1.0 and full.sample == 1.0
    half = render_scale(50, voxels)
    assert half.max_voxels == voxels // 8 and half.pixel == 2.0
    quarter = render_scale(25, voxels)
    assert quarter.max_voxels == voxels // 64 and quarter.pixel == 4.0
    least = render_scale(1, voxels)
    assert least.max_voxels == layers.RENDER_SCALE_MIN_VOXELS
    assert least.pixel == RENDER_SCALE_MAX_PIXEL
    assert render_scale(0, voxels).percent == 1 and render_scale(150, voxels).percent == 100
    dragged = render_scale(100, voxels, dragging=True)
    assert dragged.max_voxels == RENDER_DRAG_VOXELS and dragged.pixel >= 4 and dragged.sample > 1
    gpu = render_scale(100, voxels, gpu=True)
    gpu_dragged = render_scale(100, voxels, dragging=True, gpu=True)
    assert gpu.max_voxels is None and gpu_dragged.max_voxels is None
    assert gpu_dragged.pixel == 2.0 and gpu_dragged.sample == 1.0
    huge = render_scale(100, 4 * GPU_MAX_VOXELS, gpu=True)
    assert huge.max_voxels == GPU_MAX_VOXELS


def test_the_slider_replaces_full_resolution_and_is_remembered(view, qtbot):
    stage = view
    assert stage.findChild(object, "VolumeRenderFull") is None
    slider = stage._scale_slider
    assert (slider.minimum(), slider.maximum()) == (1, 100)
    _use(stage, "cpu", qtbot)
    slider.setValue(25)
    _wait_render(stage, qtbot)
    request = stage._render_request("preview")
    assert request.scale.percent == 25 and request.scale.pixel == 4.0
    assert np.prod([-(-n // s) for n, s in zip(stage.session.image.shape, request.steps)]) \
        <= max(layers.RENDER_SCALE_MIN_VOXELS, stage.session.image.size // 64)
    assert stage._scale_value.text() == "25%"
    assert stage._render_state.scale.percent == 25 and "25% scale" in stage._detail.text()
    slider.setValue(100)
    _wait_render(stage, qtbot)
    assert stage._render_request("preview").steps == (1, 1, 1)
    assert preferences.get_volume_view()["scale"] == 100


def test_backends_offered_and_the_choice_is_remembered(view, qtbot):
    stage = view
    offered = [stage._backend_box.itemData(i) for i in range(stage._backend_box.count())]
    assert offered == list(mm.volume_render_backends())
    assert offered[0] == "auto" and offered[-1] == "cpu"
    # 2026-10-16: wgpu ships by default, so Vulkan, Metal and Direct3D 12 are
    # offered where wgpu finds an adapter (tests/qt/test_673_true3d_wgpu.py).
    assert set(offered) <= set(mm.VOLUME_BACKENDS)
    _use(stage, "cpu", qtbot)
    assert preferences.get_volume_view()["backend"] == "cpu"
    assert stage._render_state.backend == "cpu" and "the CPU" in stage._detail.text()
    assert mm._volume_view_settings()[0] == "cpu"


def test_auto_falls_back_to_the_cpu_with_a_one_line_note(view, qtbot, monkeypatch, caplog):
    stage = view

    def broken(*_args, **_kwargs):
        raise RuntimeError("no OpenGL context could be created")

    monkeypatch.setattr(mm, "_GLVolumeContext", broken)
    # 2026-10-16: the chain is OpenGL alone here, with the GPU allowed; the
    # wgpu steps of Auto are covered in test_673_true3d_wgpu.py.
    monkeypatch.setattr(mm, "volume_gpu_refusal", lambda: "")
    monkeypatch.setattr(mm, "volume_render_backends", lambda: ("auto", "opengl", "cpu"))
    if stage._gl is not None:
        stage._gl.close()
    stage._gl = None
    with caplog.at_level(logging.WARNING, logger="spacr.qt.make_masks"):
        _use(stage, "cpu", qtbot)
        _use(stage, "auto", qtbot)
    assert stage._active_backend() == "cpu" and stage._gl is None
    notes = [record.getMessage() for record in caplog.records if "True 3D" in record.getMessage()]
    assert len(notes) == 1 and "\n" not in notes[0] and "CPU" in notes[0]
    assert "no OpenGL context" in stage._detail.text()
    assert stage._render_state.backend == "cpu" and stage._render_state.rgb.any()


def test_a_gpu_that_fails_mid_frame_hands_the_frame_to_the_cpu(view, qtbot, monkeypatch):
    stage = view
    stage._gl = None
    stage._gl_tried = True
    monkeypatch.setattr(mm, "volume_render_backends", lambda: ("auto", "opengl", "cpu"))
    _gl = _gl_context()
    stage._gl = _gl

    def failing(_request):
        raise RuntimeError("GL_OUT_OF_MEMORY")

    monkeypatch.setattr(_gl, "render", failing)
    assert stage._submit_render("preview")
    _wait_render(stage, qtbot)
    assert stage._gl is None and stage._render_state.backend == "cpu"
    assert "GL_OUT_OF_MEMORY" in stage._backend_note


def test_auto_does_not_choose_a_software_rasteriser(monkeypatch):
    _gl_context()
    real = layers.GLVolumeRenderer.limits

    def software(self):
        found = real(self)
        found["renderer"] = "llvmpipe (LLVM 15.0.7, 256 bits)"
        return found

    monkeypatch.setattr(layers.GLVolumeRenderer, "limits", software)
    with pytest.raises(RuntimeError, match="software"):
        mm._GLVolumeContext(allow_software=False)
    assert mm._GLVolumeContext(allow_software=True).name().startswith("llvmpipe")


def test_finishes_are_ordered_from_dull_to_shiny_and_checked():
    assert VOLUME_FINISHES == ("matte", "satin", "glossy", "metallic")
    speculars = [SURFACE_FINISHES[name].specular for name in VOLUME_FINISHES]
    assert speculars == sorted(speculars) and speculars[0] == 0.0
    assert SURFACE_FINISHES["glossy"].shininess > SURFACE_FINISHES["satin"].shininess
    assert SURFACE_FINISHES["metallic"].tint > 0.5
    assert all(SURFACE_FINISHES[name].tint == 0.0 for name in ("matte", "satin", "glossy"))
    assert surface_finish(None).name == "matte"
    with pytest.raises(LayerError):
        surface_finish("chrome")
    with pytest.raises(LayerError):
        render_volume([_layers(finish="chrome")[1]], _rays())


def test_finish_lighting_matte_has_no_highlight_and_metal_is_tinted():
    rays = _rays()
    light = SurfaceLight.for_rays(rays)
    red = np.array([[0.9, 0.1, 0.1]])
    mirror = np.asarray([light.half])
    away = np.asarray([light.view]) * 0 + np.asarray([[0.0, 0.0, 0.0]])
    matte = shade_surface(red, mirror, light, surface_finish("matte"))
    glossy = shade_surface(red, mirror, light, surface_finish("glossy"))
    metal = shade_surface(red, mirror, light, surface_finish("metallic"))
    flat = SURFACE_FINISHES["matte"]
    expected = red * (flat.ambient + flat.diffuse * float(np.dot(light.half, light.light)))
    assert np.allclose(matte, expected)
    assert glossy[0, 1] - red[0, 1] * 0.6 > 0.5
    assert metal[0, 0] / metal[0, 1] > glossy[0, 0] / glossy[0, 1]
    assert np.allclose(shade_surface(red, away, light, surface_finish("glossy")),
                       red * SURFACE_FINISHES["glossy"].ambient)
    pictures = {}
    for name in VOLUME_FINISHES:
        masks = _layers(finish=name)[1]
        masks.opacity = 1.0
        pictures[name] = render_volume([masks], rays)[0]
    assert all(not np.allclose(pictures["matte"], pictures[name]) for name in VOLUME_FINISHES[1:])
    assert pictures["glossy"].max() > pictures["matte"].max()


def test_the_finish_list_sets_the_masks_layer_and_is_remembered(view, qtbot):
    stage = view
    row = stage._layer_rows["Masks"]
    assert row.finish is not None and stage._layer_rows["Channel 1"].finish is None
    assert [row.finish.itemData(i) for i in range(row.finish.count())] == list(VOLUME_FINISHES)
    row.finish.setCurrentIndex(row.finish.findData("metallic"))
    _wait_render(stage, qtbot)
    assert stage._layers["Masks"].metadata["finish"] == "metallic"
    finishes = {entry[0]: entry[-1] for entry in stage._render_request("preview").layers}
    assert finishes["Masks"] == "metallic"
    assert preferences.get_volume_view()["finish"] == "metallic"


def test_the_lens_cube_geometry_warp_and_dimming_mask():
    lens = VolumeLens((6.0, 12.0, 14.0), (3.0, 4.0, 5.0), zoom=2.0, dim=0.25)
    corners = lens.corners()
    assert corners.shape == (8, 3)
    assert np.allclose(corners.mean(axis=0), lens.centre)
    assert np.allclose(corners.max(axis=0) - corners.min(axis=0), (6.0, 8.0, 10.0))
    assert len(CUBE_EDGES) == 12 and len({tuple(sorted(e)) for e in CUBE_EDGES}) == 12
    for a, b in CUBE_EDGES:
        assert np.count_nonzero(corners[a] != corners[b]) == 1
    points = np.array([[6.0, 8.0, 0.0], [12.0, 14.0, 0.0], [14.0, 18.0, 0.0]], np.float32)
    where, weight = lens.warp(points)
    assert np.allclose(where[:, 0], (6.0, 12.0, 14.0)) and weight[0] == 1.0
    assert np.allclose(where[:, 1], (7.0, 13.0, 16.0)) and weight[1] == 1.0
    assert np.allclose(where[:, 2], points[:, 2]) and weight[2] == pytest.approx(0.25)
    with pytest.raises(LayerError):
        VolumeLens((0.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    with pytest.raises(LayerError):
        VolumeLens((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), zoom=0.5)
    rays = _rays()
    covered = lens.covers(rays)
    assert covered.any() and not covered.all()
    channel = ImageLayer(np.random.default_rng(1).random(_spheres().shape).astype(np.float32),
                         name="g", contrast_limits=[(0.0, 1.0)], colormaps=["gray"])
    plain, _ = render_volume([channel], rays)
    magnified, _ = render_volume([channel], rays, lens=lens)
    assert np.allclose(magnified[~covered], plain[~covered] * 0.25, atol=1e-5)
    assert not np.allclose(magnified[covered], plain[covered] * 0.25)


def test_the_m_key_turns_the_3d_magnifier_cube_on_and_shift_ctrl_resize_and_zoom(view, screen, qtbot):
    stage = view
    _use(stage, "cpu", qtbot)
    assert not stage._cube.on and stage.cube_corners() is None
    assert screen._toggle_magnifier_key()
    assert stage._cube.on and stage._cube_button.isChecked()
    assert not screen._canvas.magnifier.enabled
    corners = stage.cube_corners()
    side = stage._cube.side
    assert np.allclose(corners.max(axis=0) - corners.min(axis=0), side)
    assert np.allclose(corners.mean(axis=0), [sum(pair) / 2 for pair in stage.session.limits])
    ax = stage._view_axes()
    cube = next(artist for artist in ax.collections if artist.get_gid() == "VolumeMagnifierCube")
    assert cube.get_visible() and len(cube._segments3d) == 12
    _wait_render(stage, qtbot)
    lens = stage._render_request("preview").lens
    assert lens is not None and lens.zoom == stage._cube.zoom and lens.dim == mm.VOLUME_CUBE_DIM
    widget = stage._panel.canvas._canvas
    speed = stage._wheel_speed()

    def wheel(modifiers, up=True):
        centre = QPointF(widget.width() / 2, widget.height() / 2)
        event = QWheelEvent(centre, widget.mapToGlobal(centre), QPoint(0, 0),
                            QPoint(0, 120 if up else -120), Qt.NoButton, modifiers,
                            Qt.NoScrollPhase, False)
        QApplication.sendEvent(widget, event)

    zoom = stage._cube.zoom
    view_zoom = stage._panel.canvas._volume_zoom
    wheel(Qt.ShiftModifier)
    assert stage._cube.side == pytest.approx(side * speed)
    wheel(Qt.ShiftModifier, up=False)
    wheel(Qt.ShiftModifier, up=False)
    assert stage._cube.side == pytest.approx(side / speed)
    wheel(Qt.ControlModifier)
    assert stage._cube.zoom == pytest.approx(min(mm._MAGNIFIER_ZOOM_RANGE[1], zoom * speed))
    assert stage._panel.canvas._volume_zoom == view_zoom
    for _ in range(80):
        stage.step_cube_size(True)
        stage.step_cube_zoom(True)
    assert stage._cube.side == pytest.approx(stage._cube_side_range()[1])
    assert stage._cube.zoom == mm._MAGNIFIER_ZOOM_RANGE[1]
    for _ in range(200):
        stage.step_cube_size(False)
    assert stage._cube.side == pytest.approx(stage._cube_side_range()[0])
    assert screen._toggle_magnifier_key()
    assert not stage._cube.on and not cube.get_visible()
    assert stage._render_request("preview").lens is None


def test_the_cube_follows_the_pointer_on_the_slice_plane_and_dims_outside(view, qtbot):
    stage = view
    _use(stage, "cpu", qtbot)
    canvas = stage._panel.canvas
    canvas._view_angles = (89.0, -90.0, 0.0)
    canvas.render_now()
    canvas._canvas.draw()
    stage.set_slice(6)
    stage.set_magnifier(True)
    sx, sy, sz = stage.session.xyz_spacing
    target = np.array([8.5 * sx, 8.5 * sy, 6.5 * sz])
    origin, matrix, screen0 = stage._projection(stage._view_axes())
    point = screen0 + matrix @ (target - origin)
    assert stage._move_cube(point[0], point[1])
    assert np.allclose(stage._cube.centre, target, atol=1e-6)
    stage._layer_rows["Masks"].eye.setChecked(False)
    _wait_render(stage, qtbot)
    dimmed = stage._render_state.rgb.astype(float)
    stage.set_magnifier(False)
    _wait_render(stage, qtbot)
    plain = stage._render_state.rgb.astype(float)
    assert dimmed.sum() < plain.sum()
    x0, x1, y0, y1 = stage._render_state.rect
    column, row = int(round(point[0] - x0)), int(round(y1 - point[1]))
    assert dimmed[row, column].max() == pytest.approx(plain[row, column].max(), abs=3)


def test_cpu_fallback_meets_its_frame_budget_at_low_scale(view, qtbot):
    stage = view
    _use(stage, "cpu", qtbot)
    stage._scale_slider.setValue(10)
    _wait_render(stage, qtbot)
    times = []
    for _ in range(3):
        request = stage._render_request("interactive")
        started = time.perf_counter()
        assert mm._render_volume_view(request) is not None
        times.append(time.perf_counter() - started)
    assert request.scale.pixel == RENDER_SCALE_MAX_PIXEL
    assert sorted(times)[1] < 0.1


def test_gpu_frames_match_the_cpu_for_every_rendering_finish_and_lens():
    gl = _gl_context()
    rays = _rays()
    lens = VolumeLens((6.0, 12.0, 14.0), (4.0, 4.0, 4.0), zoom=2.0)
    with gl:
        for rendering in VOLUME_RENDERINGS:
            channel, _masks = _layers(rendering=rendering)
            cpu, _ = render_volume([channel], rays)
            gpu, _ = gl.renderer.render([channel], rays)
            assert np.abs(cpu * 255 - gpu).max() <= 3, rendering
        for labels_rendering in ("surface", "translucent"):
            for finish in VOLUME_FINISHES:
                for magnifier in (None, lens):
                    channel, masks = _layers(labels_rendering=labels_rendering, finish=finish)
                    cpu, cpu_picked = render_volume([channel, masks], rays, highlight=2,
                                                    voxel_size=(2.0, 1.0, 1.0), lens=magnifier)
                    gpu, gpu_picked = gl.renderer.render(
                        [channel, masks], rays, highlight=2, voxel_size=(2.0, 1.0, 1.0),
                        lens=magnifier)
                    assert np.abs(cpu * 255 - gpu).max() <= 3, (labels_rendering, finish)
                    assert np.array_equal(cpu_picked, gpu_picked)
        channel, masks = _layers()
        big, _ = gl.renderer.render([channel, masks], rays, size=(60, 72))
        assert big.shape == (60, 72, 3) and big.dtype == np.uint8
    gl.close()


def test_the_gpu_view_uses_the_cpu_camera_and_picks_the_same_objects(view, qtbot, monkeypatch):
    stage = view
    monkeypatch.setattr(mm, "volume_render_backends", lambda: ("auto", "opengl", "cpu"))
    _use(stage, "auto", qtbot)
    if stage._active_backend() != "opengl":
        pytest.skip("no OpenGL context offscreen")
    stage._scale_slider.setValue(99)
    stage._scale_slider.setValue(100)
    _wait_render(stage, qtbot)
    gpu_request = stage._render_request("preview")
    gpu_state = stage._render_state
    assert gpu_request.backend == "opengl" and gpu_state.backend == "opengl"
    assert "the GPU (OpenGL" in stage._detail.text()
    projection = stage._projection(stage._view_axes())
    _use(stage, "cpu", qtbot)
    cpu_request = stage._render_request("preview")
    cpu_state = stage._render_state
    for gpu_part, cpu_part in zip(projection, stage._projection(stage._view_axes())):
        assert np.array_equal(gpu_part, cpu_part)
    assert gpu_request.rays == cpu_request.rays and gpu_request.rect == cpu_request.rect
    assert gpu_state.rgb.shape == cpu_state.rgb.shape
    assert np.array_equal(gpu_state.picked, cpu_state.picked)
    assert np.abs(gpu_state.rgb.astype(int) - cpu_state.rgb.astype(int)).mean() < 2.0
    x0, x1, y0, y1 = gpu_state.rect
    rows, columns = gpu_state.picked.shape
    row, column = np.argwhere(gpu_state.picked == 2)[0]
    x = x0 + (column + .5) * (x1 - x0) / columns
    y = y1 - (row + .5) * (y1 - y0) / rows
    _use(stage, "opengl", qtbot)
    assert stage._render_state.backend == "opengl"
    assert stage.pick_at(x, y) == 2


def test_gpu_drag_frames_stay_inside_an_interactive_budget(view, qtbot, monkeypatch):
    stage = view
    monkeypatch.setattr(mm, "volume_render_backends", lambda: ("auto", "opengl", "cpu"))
    _use(stage, "auto", qtbot)
    if stage._active_backend() != "opengl":
        pytest.skip("no OpenGL context offscreen")
    stage._scale_slider.setValue(100)
    _wait_render(stage, qtbot)
    times = []
    for _ in range(5):
        started = time.perf_counter()
        assert stage._submit_render("interactive")
        times.append(time.perf_counter() - started)
    assert stage._render_state.quality == "interactive"
    assert sorted(times)[2] < 0.1
    _settle()
