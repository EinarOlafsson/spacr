"""N673 follow-up (2026-10-15): a napari-like True 3D view in Make Masks.

The maintainer's review of True 3D: it looked like a 3-D UMAP (a point
cloud), the image itself was not visible, the axes and grid stayed the same
size while the objects zoomed, and there was no opacity per channel or for
the masks. The view now draws the volume as layers (one per channel, then
the masks) with an eye, colour, opacity, contrast and rendering each; the
axes and grid are off until asked for and zoom with the data; Paint, Erase
and Pick work in the 3-D view on the slice plane it outlines; the volume is
downsampled for turning and drawn off the GUI thread.

No GPU and no network: small synthetic stacks.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtWidgets import QApplication, QMessageBox

from spacr.layers import (ImageLayer, LabelsLayer, LayerError, VolumeRays,
                          downsample_volume, render_volume, volume_steps)
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import MakeMasksScreen

WAIT = 60000


def _settle(rounds=4):
    for _ in range(rounds):
        QApplication.processEvents()


def _spheres(shape=(12, 24, 28)):
    z, y, x = np.indices(shape)
    labels = np.zeros(shape, np.uint16)
    labels[(z - 6) ** 2 + (y - 8) ** 2 + (x - 8) ** 2 <= 16] = 1
    labels[(z - 6) ** 2 + (y - 16) ** 2 + (x - 20) ** 2 <= 9] = 2
    return labels


def _two_channel(folder, axes="CZYX"):
    labels = _spheres()
    first = (labels == 1).astype(np.uint16) * 900 + 50
    second = (labels == 2).astype(np.uint16) * 400 + 20
    data = np.stack([first, second]) if axes == "CZYX" else np.stack([first, second], axis=1)
    tifffile.imwrite(Path(folder) / "cells.tif", data, metadata={"axes": axes})
    (Path(folder) / "masks").mkdir(exist_ok=True)
    tifffile.imwrite(Path(folder) / "masks" / "cells.tif", labels, metadata={"axes": "ZYX"})
    return Path(folder) / "cells.tif", labels, first, second


@pytest.fixture
def screen(qtbot, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.resize(1500, 950)
    widget.show()
    qtbot.waitExposed(widget)
    _settle()
    yield widget
    stage = widget._mask_stage
    qtbot.waitUntil(lambda: stage._worker.idle() and stage._mesher.idle()
                    and stage._renderer.idle(), timeout=WAIT)
    stage.dirty = False


@pytest.fixture
def view(screen, tmp_path, qtbot):
    source, labels, first, second = _two_channel(tmp_path)
    stage = screen.open_volume(str(source))
    assert stage is screen._mask_stage
    assert stage.set_view("true3d")
    _settle()
    qtbot.waitUntil(stage.render_idle, timeout=WAIT)
    _settle()
    stage.expected = (labels, first, second)
    return stage


@pytest.fixture
def cpu_view(request):
    from spacr.qt import preferences

    settings = preferences._settings()
    key = preferences._KEY_VOLUME_VIEW
    present = settings.contains(key)
    previous = settings.value(key) if present else None
    try:
        preferences.set_volume_view(backend="cpu")
        yield request.getfixturevalue("view")
    finally:
        if present:
            settings.setValue(key, previous)
        else:
            settings.remove(key)


def _wait_render(stage, qtbot):
    qtbot.waitUntil(stage.render_idle, timeout=WAIT)
    _settle()


def _screen_of(stage, point):
    origin, matrix, screen0 = stage._projection(stage._view_axes())
    return screen0 + matrix @ (np.asarray(point, float) - origin)


def test_a_multichannel_tiff_opens_with_every_channel(tmp_path):
    for axes in ("CZYX", "ZCYX"):
        folder = tmp_path / axes
        folder.mkdir()
        source, labels, first, second = _two_channel(folder, axes)
        assert engine._is_volume_file(source)
        session = engine.load_volume_image_and_mask(str(folder), "cells.tif")
        assert session.axes == "ZYX" and session.image.shape == labels.shape
        assert len(session.channels) == 2
        assert np.array_equal(session.channels[0], first) and np.array_equal(session.channels[1], second)
        assert np.array_equal(session.image, first) and np.array_equal(session.mask, labels)
    one = engine.VolumeMaskSession(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), np.uint16))
    assert len(one.channels) == 1 and one.channels[0] is one.image
    with pytest.raises(ValueError):
        engine.VolumeMaskSession(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), np.uint16),
                                 channels=[np.zeros((2, 3, 5))])


def test_one_layer_per_channel_and_one_for_the_masks(view):
    stage = view
    names = [layer.name for layer in stage._layers]
    assert names == ["Channel 1", "Channel 2", "Masks"]
    first, second, masks = list(stage._layers)
    assert isinstance(first, ImageLayer) and isinstance(masks, LabelsLayer)
    assert first.colormap.name == "green" and second.colormap.name == "magenta"
    assert masks.data is stage.session.mask
    assert set(stage._layer_rows) == set(names)
    assert stage._layer_rows["Masks"].colour is None and stage._layer_rows["Masks"].low is None


def test_each_layer_has_its_own_visibility_opacity_colour_contrast_and_rendering(view, qtbot):
    stage = view
    rows = stage._layer_rows
    second, masks = stage._layers["Channel 2"], stage._layers["Masks"]
    rows["Channel 2"].eye.setChecked(False)
    assert not second.visible and stage._layers["Channel 1"].visible
    rows["Channel 2"].opacity.setValue(35)
    assert second.opacity == pytest.approx(0.35)
    rows["Channel 2"].colour.setCurrentIndex(rows["Channel 2"].colour.findData("cyan"))
    assert second.colormap.name == "cyan" and stage._layers["Channel 1"].colormap.name == "green"
    rows["Channel 2"].low.setValue(30)
    rows["Channel 2"].high.setValue(300)
    assert second.contrast_limits(0) == (30, 300)
    rows["Channel 2"].rendering.setCurrentIndex(rows["Channel 2"].rendering.findData("attenuated_mip"))
    assert second.metadata["rendering"] == "attenuated_mip"
    assert stage._layers["Channel 1"].metadata["rendering"] == "mip"
    rows["Masks"].opacity.setValue(20)
    rows["Masks"].eye.setChecked(False)
    rows["Masks"].rendering.setCurrentIndex(rows["Masks"].rendering.findData("translucent"))
    assert masks.opacity == pytest.approx(0.2) and not masks.visible
    assert masks.metadata["rendering"] == "translucent"
    _wait_render(stage, qtbot)
    request = stage._render_request("preview")
    shown = {name: (visible, opacity) for name, _image, _data, visible, opacity, *_rest in request.layers}
    assert shown == {"Channel 1": (True, 1.0), "Channel 2": (False, 0.35), "Masks": (False, 0.2)}
    assert not stage._render_state.picked.any()


def test_the_rendered_view_shows_the_image_and_the_objects(view, qtbot):
    stage = view
    state = stage._render_state
    assert state is not None and state.rgb.dtype == np.uint8
    assert set(np.unique(state.picked)) == {0, 1, 2}
    x0, x1, y0, y1 = state.rect
    assert state.rgb.shape[:2] == (round(y1 - y0), round(x1 - x0))
    artist = stage._render_artist
    assert artist in stage._panel.canvas._figure.images and artist.get_zorder() < 0
    stage._layer_rows["Masks"].eye.setChecked(False)
    _wait_render(stage, qtbot)
    green = stage._render_state.rgb[..., 1].astype(int)
    assert green.max() > 200
    stage._layer_rows["Channel 1"].eye.setChecked(False)
    stage._layer_rows["Channel 2"].eye.setChecked(False)
    _wait_render(stage, qtbot)
    assert not stage._render_state.rgb.any()


def test_axes_and_grid_are_off_by_default_and_toggle(view, qtbot):
    stage = view
    assert not stage._grid_toggle.isChecked()
    assert not stage._view_axes()._axis3don
    stage._grid_toggle.setChecked(True)
    _settle()
    assert stage._view_axes()._axis3don
    stage._grid_toggle.setChecked(False)
    _settle()
    assert not stage._view_axes()._axis3don


def test_the_grid_zooms_with_the_data(view, qtbot):
    stage = view
    stage._grid_toggle.setChecked(True)
    canvas = stage._panel.canvas
    canvas._canvas.draw()
    ax = stage._view_axes()
    limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
    (x0, x1), (y0, y1), (z0, z1) = stage.session.limits
    before = np.linalg.norm(_screen_of(stage, (x1, y1, z1))[:2] - _screen_of(stage, (x0, y0, z0))[:2])
    ticks = ax.get_xticks().copy()
    canvas._volume_zoom = 2.0
    canvas.render_now()
    canvas._canvas.draw()
    ax = stage._view_axes()
    assert (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()) == limits
    assert np.array_equal(ax.get_xticks(), ticks)
    after = np.linalg.norm(_screen_of(stage, (x1, y1, z1))[:2] - _screen_of(stage, (x0, y0, z0))[:2])
    assert after == pytest.approx(2.0 * before, rel=0.02)
    assert ax._axis3don


def test_shift_drag_pans_the_box_and_reset_view_centres_it(view, qtbot):
    stage = view
    ax = stage._view_axes()
    start = ax.get_position(original=True).x0
    stage._pan_by(100, 0)
    assert stage._view_axes().get_position(original=True).x0 > start
    stage._panel.reset_view()
    _settle()
    assert stage._pan == (0.0, 0.0)
    assert stage._view_axes().get_position(original=True).x0 == pytest.approx(start)


def test_display_translation_invalidates_the_volume_once_without_changing_the_camera():
    from types import SimpleNamespace

    matrix = np.eye(3)
    affine = SimpleNamespace(get_matrix=lambda: matrix)
    ax = SimpleNamespace(M=np.eye(4), transData=SimpleNamespace(get_affine=lambda: affine))
    figure = SimpleNamespace(bbox=SimpleNamespace(bounds=(0, 0, 600, 400)))
    calls = []
    stage = SimpleNamespace(
        _view_axes=lambda: ax, view='true3d', _view_signature=None,
        _panel=SimpleNamespace(canvas=SimpleNamespace(_figure=figure)),
        _view_changed=lambda: calls.append(True))
    mm._VolumeStage._view_drawn(stage, None)
    original = stage._view_signature
    mm._VolumeStage._view_drawn(stage, None)
    assert len(calls) == 1
    matrix[0, 2] = 31.0
    matrix[1, 2] = -17.0
    mm._VolumeStage._view_drawn(stage, None)
    assert len(calls) == 2
    assert stage._view_signature != original
    mm._VolumeStage._view_drawn(stage, None)
    matrix[0, 2] += 1e-12
    mm._VolumeStage._view_drawn(stage, None)
    assert len(calls) == 2
    assert np.array_equal(ax.M, np.eye(4))


def test_a_panned_frame_matches_independent_current_view_rays_and_pixels(view, qtbot, monkeypatch):
    stage = view
    canvas = stage._panel.canvas._canvas
    canvas.draw()
    _wait_render(stage, qtbot)
    before = stage._render_request('preview')
    matrix = stage._view_axes().M.copy()
    figure_bounds = tuple(stage._panel.canvas._figure.bbox.bounds)
    labels = stage.session.mask.copy()
    requests = []
    original_request = stage._render_request

    def request(quality):
        value = original_request(quality)
        if value is not None:
            requests.append(value)
        return value

    monkeypatch.setattr(stage, '_render_request', request)
    stage._pan_by(23, -13)
    canvas.draw()
    _wait_render(stage, qtbot)
    shown = stage._render_state
    fresh = original_request(shown.quality)
    assert np.array_equal(stage._view_axes().M, matrix)
    assert tuple(stage._panel.canvas._figure.bbox.bounds) == figure_bounds
    assert fresh.rect != before.rect
    assert shown.rect == fresh.rect
    assert requests
    delivered = next(value for value in reversed(requests) if value.serial == stage._shown_serial)
    assert delivered.rays == fresh.rays
    assert delivered.steps == fresh.steps and delivered.lens == fresh.lens
    rgb, picked = mm._render_volume_view(fresh)
    assert np.array_equal(shown.rgb, rgb)
    assert np.array_equal(shown.picked, picked)
    assert np.array_equal(stage.session.mask, labels)


def test_the_slice_plane_is_outlined_and_follows_the_slice(view, monkeypatch):
    stage = view
    ax = stage._view_axes()
    plane = next(artist for artist in ax.collections if artist.get_gid() == "VolumeSlicePlane")
    updates = []
    actual_set_verts = plane.set_verts

    def record_verts(vertices, *args, **kwargs):
        updates.append(np.asarray(vertices).copy())
        return actual_set_verts(vertices, *args, **kwargs)

    monkeypatch.setattr(plane, "set_verts", record_verts)
    stage.set_slice(3)
    assert stage._slice_artist is plane
    assert updates and updates[-1].shape == (1, 4, 3)
    z = updates[-1][..., 2]
    assert np.allclose(z, 3.5 * stage.session.xyz_spacing[2])


def _top_view(stage):
    canvas = stage._panel.canvas
    canvas._view_angles = (89.0, -90.0, 0.0)
    canvas.render_now()
    canvas._canvas.draw()


def test_paint_and_erase_in_3d_change_only_the_slice_and_undo_exactly(view, screen, qtbot):
    stage = view
    _top_view(stage)
    stage.set_slice(2)
    original = stage.session.mask.copy()
    stage._tool_buttons["paint"].click()
    assert stage._tool3d == "paint"
    stage.canvas.brush_radius = 1
    sx, sy, sz = stage.session.xyz_spacing
    for column in (3, 6):
        point = _screen_of(stage, ((column + .5) * sx, (4 + .5) * sy, 2.5 * sz))
        stage._stroke_to(point[0], point[1])
    changed = stage._finish_stroke()
    assert changed > 0
    mask = stage.session.mask
    label = stage._selected
    assert label == 3
    differs = np.argwhere(mask != original)
    assert set(differs[:, 0]) == {2}
    assert mask[2, 4, 3] == label and mask[2, 4, 6] == label and mask[2, 4, 4] == label
    after_paint = mask.copy()
    stage._tool_buttons["erase"].click()
    point = _screen_of(stage, ((8 + .5) * sx, (8 + .5) * sy, 2.5 * sz))
    stage.set_slice(6)
    point = _screen_of(stage, ((8 + .5) * sx, (8 + .5) * sy, 6.5 * sz))
    stage._stroke_to(point[0], point[1])
    assert stage._finish_stroke() > 0
    assert stage.session.mask[6, 8, 8] == 0 and after_paint[6, 8, 8] == 1
    assert set(np.argwhere(stage.session.mask != after_paint)[:, 0]) == {6}
    assert stage.step_history(False) is not None
    assert np.array_equal(stage.session.mask, after_paint)
    assert stage.step_history(False) is not None
    assert np.array_equal(stage.session.mask, original)
    stage.step_history(True)
    assert np.array_equal(stage.session.mask, after_paint)
    stage._tool_buttons["erase"].click()
    assert stage._tool3d is None


def test_paint_refuses_a_slice_plane_seen_edge_on(view, screen):
    stage = view
    canvas = stage._panel.canvas
    canvas._view_angles = (0.0, -90.0, 0.0)
    canvas.render_now()
    canvas._canvas.draw()
    centre = _screen_of(stage, [sum(pair) / 2 for pair in stage.session.limits])
    assert stage._plane_point(centre[0], centre[1]) is None
    assert "edge" in screen._status_label.text() or "faces you" in screen._status_label.text()


def test_clicking_an_object_with_pick_chooses_it(view, qtbot, screen):
    stage = view
    state = stage._render_state
    x0, x1, y0, y1 = state.rect
    rows, columns = state.picked.shape
    row, column = np.argwhere(state.picked == 2)[0]
    x = x0 + (column + .5) * (x1 - x0) / columns
    y = y1 - (row + .5) * (y1 - y0) / rows
    assert stage.pick_at(x, y) == 2
    assert stage._selected == 2 and stage._labels.currentData() == 2
    row, column = np.argwhere(state.picked == 0)[0]
    assert stage.pick_at(x0 + (column + .5) * (x1 - x0) / columns,
                         y1 - (row + .5) * (y1 - y0) / rows) == 0
    assert "No object" in screen._status_label.text()


def test_volume_steps_and_downsampling_keep_exact_voxels():
    assert volume_steps((40, 1000, 1000), None) == (1, 1, 1)
    steps = volume_steps((40, 1000, 1000), 2_000_000)
    assert steps[0] == 1 and np.prod([-(-n // s) for n, s in zip((40, 1000, 1000), steps)]) <= 2_000_000
    labels = _spheres((12, 40, 50))
    small, steps = downsample_volume(labels, 3000)
    assert small.size <= 3000 and max(steps) > 1
    assert np.array_equal(small, labels[::steps[0], ::steps[1], ::steps[2]])
    same, ones = downsample_volume(labels, None)
    assert same is labels and ones == (1, 1, 1)


def test_interactive_renders_are_downsampled_and_full_scale_renders_are_not(view, qtbot):
    """2026-10-16: "Render at full resolution" became the Render scale slider
    (item 673 speed follow-up); 100 % draws every voxel."""
    stage = view
    stage._backend_box.setCurrentIndex(stage._backend_box.findData("cpu"))
    stage._scale_slider.setValue(100)
    _wait_render(stage, qtbot)
    quick = stage._render_request("interactive")
    full = stage._render_request("preview")
    assert full.steps == (1, 1, 1) and quick.scale.pixel > full.scale.pixel
    assert quick.rays.shape[0] < full.rays.shape[0]
    assert stage._render_state.quality == "preview" and stage._render_state.steps == (1, 1, 1)
    assert "every voxel" in stage._detail.text()


def test_the_render_runs_off_the_gui_thread_and_stale_results_are_dropped(cpu_view, monkeypatch, qtbot):
    """2026-10-16: on the CPU backend; the GPU draws on the GUI thread, which
    owns its GL context (tests/qt/test_673_true3d_gpu.py)."""
    stage = cpu_view
    stage._backend_box.setCurrentIndex(stage._backend_box.findData("cpu"))
    _wait_render(stage, qtbot)
    threads = []
    import spacr.layers as layers

    real = layers.render_volume

    def recording(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(layers, "render_volume", recording)
    assert stage._submit_render("preview")
    _wait_render(stage, qtbot)
    assert threads and all(thread is not threading.main_thread() for thread in threads)
    shown = stage._render_state
    old = stage._render_request("preview")
    stage._render_request("preview")
    stage._take_render((old, (np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.int64)), None))
    assert stage._render_state is shown
    assert old.cancelled()


def test_a_cancelled_cpu_render_keeps_the_frame_until_the_new_render_arrives(
        cpu_view, qtbot, monkeypatch):
    import spacr.layers as layers

    stage = cpu_view
    assert stage._backend_choice == "cpu"
    assert stage._render_state.backend == "cpu"
    previous = stage._render_state
    artist = stage._render_artist
    status = stage._screen._status_label.text()
    detail = stage._detail.text()
    first_entered, release_first = threading.Event(), threading.Event()
    newer_entered, release_newer = threading.Event(), threading.Event()
    delivered = []
    real = layers.render_volume

    def gated_cpu_render(*args, **kwargs):
        cancelled = kwargs["cancelled"]
        entered, release = ((first_entered, release_first)
                            if not first_entered.is_set()
                            else (newer_entered, release_newer))

        def gate_at_chunk():
            entered.set()
            assert release.wait(WAIT / 1000)
            return cancelled()

        return real(*args, **dict(kwargs, cancelled=gate_at_chunk))

    monkeypatch.setattr(layers, "render_volume", gated_cpu_render)
    receive = delivered.append
    stage._rendered.connect(receive)
    try:
        assert stage._submit_render("preview")
        qtbot.waitUntil(first_entered.is_set, timeout=WAIT)
        stage._layer_rows["Masks"].eye.setChecked(False)
        assert stage._submit_render("preview")
        release_first.set()
        qtbot.waitUntil(newer_entered.is_set, timeout=WAIT)
        qtbot.waitUntil(lambda: any(result is None for _, result, _ in delivered),
                        timeout=WAIT)
        cancelled_request, result, error = next(
            payload for payload in delivered if payload[1] is None)
        assert cancelled_request.session is stage.session
        assert cancelled_request.backend == "cpu"
        assert cancelled_request.cancelled()
        assert result is None and error is None
        assert stage._render_state is previous
        assert stage._render_artist is artist
        np.testing.assert_array_equal(np.asarray(artist.get_array()), previous.rgb)
        assert stage._screen._status_label.text() == status
        assert stage._detail.text() == detail

        release_newer.set()
        _wait_render(stage, qtbot)
        assert stage._render_state is not previous
        assert stage._render_state.backend == "cpu"
        assert not stage._render_state.picked.any()
        assert stage._render_state.rgb.any()
        assert "Drawn on the CPU" in stage._detail.text()
        assert all(error is None for _, _, error in delivered)
        assert any(request.session is stage.session and result is not None
                   for request, result, _ in delivered)
    finally:
        release_first.set()
        release_newer.set()
        _wait_render(stage, qtbot)
        stage._rendered.disconnect(receive)


def test_a_cpu_drag_shows_frames_while_the_view_still_turns(view, qtbot):
    """2026-10-19 macOS acceptance: every mouse move cancelled the CPU frame
    being drawn, so a drag showed no new picture until the mouse stopped."""
    stage = view
    stage._backend_box.setCurrentIndex(stage._backend_box.findData("cpu"))
    _wait_render(stage, qtbot)
    frame = (np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.int64))
    drawing = stage._render_request("interactive")
    newer = stage._render_request("interactive")
    assert not drawing.cancelled()
    stage._take_render((drawing, frame, None))
    assert stage._render_state.rgb is frame[0]

    stage._take_render((newer, (frame[0].copy(), frame[1]), None))
    shown = stage._render_state
    stage._take_render((drawing, frame, None))
    assert stage._render_state is shown, "an older frame never replaces a newer one"

    turning = stage._render_request("interactive")
    stage._version += 1
    stage._render_request("interactive")
    assert turning.cancelled(), "changed content cancels the frame being drawn"
    turning = stage._render_request("interactive")
    settled = stage._render_request("preview")
    assert turning.cancelled() and not settled.cancelled()
    stage._take_render((turning, frame, None))
    assert stage._render_state is shown


def test_renderer_modes_hidden_layers_and_label_opacity():
    labels = _spheres()
    image = labels.astype(np.float32) * 100
    layer = ImageLayer(image, name="c", contrast_limits=[(0, 100)], colormaps=["green"])
    masks = LabelsLayer(labels, name="m")
    rays = VolumeRays(origin=(0., 0., 0.), across=(0., 0., 1.), down=(0., 1., 0.),
                      depth=(1., 0., 0.), shape=(24, 28, 12))
    values = {}
    for mode in ("mip", "attenuated_mip", "translucent", "average", "minip"):
        layer.metadata["rendering"] = mode
        rgb, picked = render_volume([layer], rays)
        values[mode] = float(rgb[8, 8, 1])
        assert rgb[0, 0].max() == 0 and not picked.any()
    assert values["mip"] == pytest.approx(1.0, abs=0.02)
    assert values["minip"] == 0 and 0 < values["average"] < values["mip"]
    assert values["attenuated_mip"] <= values["mip"]
    rgb, picked = render_volume([layer, masks], rays)
    assert picked[8, 8] == 1 and picked[16, 20] == 2 and picked[0, 0] == 0
    layer.metadata["rendering"] = "mip"
    masks.opacity = 0.0
    hidden, _ = render_volume([layer, masks], rays)
    plain, _ = render_volume([layer], rays)
    assert plain.max() > 0
    assert np.allclose(hidden, plain)
    masks.metadata["rendering"] = "plane"
    with pytest.raises(LayerError):
        render_volume([masks], rays)
    assert render_volume([layer], rays, cancelled=lambda: True) is None
