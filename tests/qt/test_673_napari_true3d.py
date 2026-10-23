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


@pytest.mark.parametrize('shape,offset', [((7, 11), (3, 5)), ((8, 12), (0, 0)), ((1, 1), (10, 10))])
def test_volume_image_fastpath_matches_image_and_complete_figure(shape, offset):
    from functools import partial
    from types import SimpleNamespace
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(3, 2), dpi=100)
    canvas = FigureCanvasAgg(figure)
    rgb = np.random.default_rng(673).integers(0, 256, shape + (3,), dtype=np.uint8)
    rgb[0, 0], rgb[-1, -1] = 0, 255
    stage = SimpleNamespace(_render_state=SimpleNamespace(rgb=rgb, rect=(offset[0], 0, offset[1], 0)),
                            _panel=SimpleNamespace(canvas=SimpleNamespace(_figure=figure)),
                            _render_artist=None,
                            _make_volume_image=partial(mm._VolumeStage._make_volume_image, None))
    mm._VolumeStage._show_render(stage, draw=False)
    artist = stage._render_artist
    original = artist.make_image.args[-1]
    renderer = canvas.get_renderer()
    expected = original(renderer)
    def forbidden(*_args):
        pytest.fail('supported volume image fell back')
    actual = mm._VolumeStage._make_volume_image(None, artist, forbidden, renderer)
    assert np.array_equal(expected[0], actual[0])
    assert expected[1:3] == actual[1:3]
    assert np.array_equal(expected[3].get_matrix(), actual[3].get_matrix())
    hook = artist.make_image
    artist.make_image = original
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba()).copy()
    artist.make_image = hook
    canvas.draw()
    assert np.array_equal(pixels, np.asarray(canvas.buffer_rgba()))


@pytest.mark.parametrize('kind', ['none', 'nonarray', 'ndim', 'masked', 'rgba', 'float',
                                'nonaffine', 'transform', 'magnification', 'unsampled',
                                'renderer', 'dpi', 'clip', 'path', 'alpha', 'origin',
                                'fractional', 'outside', 'filter', 'effects'])
def test_volume_image_unsupported_inputs_preserve_original(kind):
    from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    from matplotlib.transforms import Affine2D, Bbox, Transform

    figure = Figure(figsize=(3, 2), dpi=100)
    canvas = FigureCanvasAgg(figure)
    artist = figure.figimage(np.ones((7, 11, 3), np.uint8), xo=3, yo=5, origin='upper')
    renderer, magnification, unsampled = canvas.get_renderer(), 1., False
    if kind in ('none', 'nonarray', 'ndim', 'masked', 'rgba', 'float'):
        artist._A = {'none': None, 'nonarray': [], 'ndim': np.zeros((7, 11), np.uint8),
                     'masked': np.ma.array(np.ones((7, 11, 3), np.uint8), mask=True),
                     'rgba': np.ones((7, 11, 4), np.uint8),
                     'float': np.ones((7, 11, 3), float)}[kind]
    elif kind in ('nonaffine', 'transform'):
        artist.set_transform(Transform() if kind == 'nonaffine' else Affine2D().translate(1, 0))
    elif kind == 'magnification':
        magnification = 2.
    elif kind == 'unsampled':
        unsampled = True
    elif kind == 'renderer':
        renderer = object()
    elif kind == 'dpi':
        renderer = RendererAgg(300, 200, 200)
    elif kind == 'clip':
        artist.set_clip_box(Bbox.from_bounds(3, 5, 1, 1))
    elif kind == 'path':
        artist.set_clip_path(Circle((0, 0), 1))
    elif kind == 'alpha':
        artist.set_alpha(.5)
    elif kind == 'origin':
        artist.origin = 'lower'
    elif kind in ('fractional', 'outside'):
        artist.ox = 1.5 if kind == 'fractional' else -1
    elif kind == 'filter':
        artist.set_agg_filter(lambda *args: args)
    else:
        artist.set_path_effects([object()])
    calls = []
    sentinel = object()
    def original(*args):
        calls.append(args)
        return sentinel
    assert mm._VolumeStage._make_volume_image(None, artist, original, renderer,
                                              magnification, unsampled) is sentinel
    assert calls == [(renderer, magnification, unsampled)]


def test_a_camera_draw_consumes_volume_pixels_for_the_axes_pose_it_composes(view, monkeypatch):
    from types import SimpleNamespace

    stage = view
    canvas = stage._panel.canvas._canvas
    ax = stage._view_axes()
    poses = {}
    def render(request):
        marker = len(poses) + 1
        poses[marker] = np.asarray(ax.M).copy()
        rows, columns = request.rays.shape[:2]
        rgb = np.zeros((rows, columns, 3), np.uint8)
        rgb[..., 0] = marker
        return rgb, np.full((rows, columns), marker, np.int64)
    monkeypatch.setattr(stage, '_active_backend', lambda: 'metal')
    monkeypatch.setattr(stage, '_backend_choice', 'metal')
    monkeypatch.setattr(stage, '_gl', SimpleNamespace(backend='metal', render=render, limits={}, name=lambda: 'test-only synchronous renderer', close=lambda: None))
    assert stage._submit_render('preview')
    canvas.draw()
    stage._settle_timer.stop()
    initial_pose = np.asarray(ax.M).copy()
    consumed, composed = [], []
    artist = stage._render_artist
    make_image, draw_axes = artist.make_image, ax.draw
    def consume_pixels(*args, **kwargs):
        marker = int(np.asarray(artist.get_array())[0, 0, 0])
        consumed.append(poses[marker].copy())
        return make_image(*args, **kwargs)
    def compose_axes(*args, **kwargs):
        result = draw_axes(*args, **kwargs)
        composed.append(np.asarray(ax.M).copy())
        return result
    monkeypatch.setattr(artist, 'make_image', consume_pixels)
    monkeypatch.setattr(ax, 'draw', compose_axes)
    ax.view_init(elev=ax.elev + 17, azim=ax.azim + 23, roll=ax.roll)
    canvas.draw()
    stage._settle_timer.stop()
    assert consumed and composed
    assert not np.array_equal(initial_pose, composed[-1])
    assert all(np.array_equal(pose, composed[-1]) for pose in consumed), 'FigureImage consumed pixels from a different camera pose than the axes composed in that draw'


@pytest.fixture
def prepaint():
    from types import SimpleNamespace as NS

    calls = []
    affine = np.eye(3)
    ax = NS(_unstale_viewLim=lambda: None, get_axes_locator=lambda: None,
            apply_aspect=lambda _position: None, get_proj=lambda: np.eye(4),
            transData=NS(get_affine=lambda: NS(get_matrix=lambda: affine)))
    figure = NS(get_layout_engine=lambda: None, bbox=NS(bounds=(0, 0, 40, 30)))
    widget = NS(_spacr_draw_timer=NS(stop=lambda: calls.append('stop')),
                update=lambda: calls.append('update'), draw_idle=lambda: calls.append('idle'))
    stage = NS(_panel=NS(canvas=NS(_figure=figure, _canvas=widget)), view='true3d',
               session=object(), _gl=NS(backend='metal'), _backend_choice='metal',
               _view_axes=lambda: ax, _view_signature=None,
               _view_changed=lambda: calls.append('render'))
    stage._prepaint_axes = lambda f, w: mm._VolumeStage._prepaint_axes(stage, f, w)
    return stage, figure, widget, ax, calls


@pytest.mark.parametrize('kind', ['cpu', 'context_cpu', 'active', 'custom', 'subclass', 'legacy', 'import', 'old_figure', 'old_widget'])
def test_prepaint_guard_preserves_legacy_draw_and_queue(prepaint, monkeypatch, kind):
    import builtins
    from types import SimpleNamespace
    from matplotlib.layout_engine import PlaceHolderLayoutEngine, TightLayoutEngine

    stage, figure, widget, _ax, calls = prepaint
    if kind == 'cpu':
        stage._backend_choice = 'cpu'
    elif kind == 'context_cpu':
        stage._gl.backend = 'cpu'
    elif kind in ('active', 'custom', 'subclass'):
        class CustomPlaceholder(PlaceHolderLayoutEngine):
            pass
        engine = {'active': TightLayoutEngine(), 'custom': object(),
                  'subclass': CustomPlaceholder(False, False)}[kind]
        figure.get_layout_engine = lambda: engine
    elif kind == 'legacy':
        del figure.get_layout_engine
    elif kind == 'import':
        original = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == 'matplotlib.layout_engine':
                raise ImportError('legacy matplotlib')
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', missing)
    elif kind == 'old_figure':
        figure = SimpleNamespace()
    else:
        widget = SimpleNamespace()
    mm._VolumeStage._queue_volume_frame(stage, figure, widget, lambda: calls.append('queue'))
    mm._VolumeStage._draw_volume_frame(stage, figure, widget, lambda renderer: calls.append(renderer), 'draw')
    assert calls == ['queue', 'draw']


@pytest.mark.parametrize('placeholder', [False, True])
def test_prepaint_accepts_only_verified_layout_and_resets_suppression(prepaint, placeholder):
    from matplotlib.layout_engine import PlaceHolderLayoutEngine

    stage, figure, widget, _ax, calls = prepaint
    if placeholder:
        figure.get_layout_engine = lambda: PlaceHolderLayoutEngine(False, False)
    mm._VolumeStage._queue_volume_frame(stage, figure, widget, lambda: calls.append('legacy'))
    assert widget._draw_pending and calls == ['stop', 'update']
    mm._VolumeStage._draw_volume_frame(stage, figure, widget, lambda renderer: calls.append(renderer), 'draw')
    assert calls == ['stop', 'update', 'render', 'draw']
    assert stage._volume_prepainting is False


@pytest.mark.parametrize('error', ['prepare', 'draw', 'gpu'])
def test_prepaint_error_fallback_and_original_draw_are_not_repeated(prepaint, error):
    from types import SimpleNamespace

    stage, figure, widget, _ax, calls = prepaint
    def fail(*_args):
        raise RuntimeError(error)
    def draw(renderer):
        calls.append(renderer)
        if error == 'draw':
            fail()
    if error == 'prepare':
        stage._view_changed = fail
    elif error == 'gpu':
        stage._gl.render = fail
        stage._render_request = lambda _quality: SimpleNamespace(backend='metal' if stage._gl else 'cpu')
        stage._gpu_failed = lambda _exc: setattr(stage, '_gl', None)
        stage._renderer = SimpleNamespace(submit=lambda request: calls.append(request.backend))
        stage._submit_render = lambda quality: mm._VolumeStage._submit_render(stage, quality)
        stage._view_changed = lambda: stage._submit_render('interactive')
    if error == 'draw':
        with pytest.raises(RuntimeError, match='draw'):
            mm._VolumeStage._draw_volume_frame(stage, figure, widget, draw, 'draw')
    else:
        mm._VolumeStage._draw_volume_frame(stage, figure, widget, draw, 'draw')
    assert calls.count('draw') == 1 and not stage._volume_prepainting
    if error == 'prepare':
        assert figure._spacr_volume_prepaint_disabled and stage._view_signature is None
    if error == 'gpu':
        assert stage._gl is None and calls == ['cpu', 'draw']


@pytest.mark.parametrize('changed', ['projection', 'affine'])
def test_prepaint_composition_mismatch_disables_fastpath(prepaint, changed):
    stage, figure, widget, ax, calls = prepaint
    def draw(_renderer):
        matrix = ax.M if changed == 'projection' else ax.transData.get_affine().get_matrix()
        matrix[0, 0] += 1
    mm._VolumeStage._draw_volume_frame(stage, figure, widget, draw, None)
    assert figure._spacr_volume_prepaint_disabled and stage._view_signature is None
    assert calls == ['render', 'idle']


@pytest.mark.parametrize('preparing', [False, True])
def test_prepaint_suppresses_only_image_requeue_not_image_delivery(prepaint, preparing):
    from types import SimpleNamespace

    stage, figure, widget, _ax, calls = prepaint
    artist = SimpleNamespace(set_data=lambda _rgb: calls.append('pixels'))
    figure.images = [artist]
    stage._render_artist = artist
    stage._render_state = SimpleNamespace(rgb=np.zeros((2, 2, 3), np.uint8), rect=(0, 2, 0, 2))
    stage._volume_prepainting = preparing
    mm._VolumeStage._show_render(stage)
    assert calls == (['pixels'] if preparing else ['pixels', 'idle'])


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


def test_one_spin_gesture_changes_the_authoritative_camera_after_axes_rebuild(cpu_view, qtbot, monkeypatch):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    stage = cpu_view
    assert stage.drag_key('spin')
    canvas = stage._panel.canvas
    widget = canvas._canvas
    before = np.asarray(stage._view_axes().get_proj()).copy()
    saved = canvas._view_angles
    requests = []
    original_request = stage._render_request

    def observed_request(quality):
        request = original_request(quality)
        requests.append(request)
        return request

    monkeypatch.setattr(stage, '_render_request', observed_request)
    centre = (widget.width() * .5, widget.height() * .5)
    moved = (centre[0] + 22, centre[1] - 14)
    for kind, point, button, buttons in (
            (QEvent.MouseButtonPress, centre, Qt.LeftButton, Qt.LeftButton),
            (QEvent.MouseMove, moved, Qt.NoButton, Qt.LeftButton),
            (QEvent.MouseButtonRelease, moved, Qt.LeftButton, Qt.NoButton)):
        local = QPointF(*point)
        event = QMouseEvent(kind, local, QPointF(widget.mapToGlobal(local.toPoint())),
                            button, buttons, Qt.NoModifier)
        QApplication.sendEvent(widget, event)
    assert canvas._view_angles != saved
    _wait_render(stage, qtbot)
    assert any(request is not None and request.backend == 'cpu' for request in requests)
    canvas.render_now()
    _wait_render(stage, qtbot)
    current = stage._view_axes()
    assert (current.elev, current.azim, current.roll) == canvas._view_angles
    assert not np.array_equal(current.get_proj(), before)


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
