"""Make Masks shows a field as napari-style layers and opens multichannel files.

Every raw channel of a multichannel file is an image layer and the edited
mask is a labels layer drawn in its role colour. The mask channel chosen for
a multichannel file feeds the tools and detectors the same pixels as that
channel's own file, so the masks are identical. Restyling a layer repaints
from the same stack without rebuilding the screen, and a large view is
composited on a worker.
"""
from __future__ import annotations

import threading

import imageio.v2 as imageio
import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt

from spacr.layers import ImageLayer, LabelsLayer

PLANE = {'channel': 0}


@pytest.fixture(autouse=True)
def _reviewed_plane(monkeypatch):
    """Answer the plane review a multichannel TIFF asks for with PLANE."""
    from PySide6.QtWidgets import QInputDialog

    PLANE['channel'] = 0
    monkeypatch.setattr(QInputDialog, 'getInt', staticmethod(
        lambda *args, **kwargs: (PLANE['channel'], True)))


def _planes(n, dtype=np.uint16, shape=(96, 128)):
    rng = np.random.default_rng(7)
    top = 255 if dtype == np.uint8 else 4000
    planes = []
    for index in range(n):
        plane = rng.integers(0, top // 10, shape).astype(dtype)
        plane[10 + 8 * index:40 + 8 * index, 20 + 6 * index:60 + 6 * index] = top
        plane[60:80, 90 - 10 * index:110 - 10 * index] = top - 1
        planes.append(plane)
    return planes


def _open(qtbot, folder):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.resize(1200, 800)
    screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: not screen._loading and screen._load_worker is None
                    and screen._reader_worker is None
                    and screen._canvas.mask is not None, timeout=20000)
    return screen


def _choose_channel(qtbot, screen, channel):
    combo = screen._mask_channel_combo
    combo.setCurrentIndex(combo.findData(channel))
    qtbot.waitUntil(lambda: screen._field_channels.channel == channel)


def _imagej(folder, planes):
    tifffile.imwrite(folder / 'field.tif', np.stack(planes), imagej=True,
                     metadata={'axes': 'CYX'})


def _channel_last(folder, planes):
    tifffile.imwrite(folder / 'field.tif', np.stack(planes, axis=-1),
                     metadata={'axes': 'YXC'})


def _ome(folder, planes):
    tifffile.imwrite(folder / 'field.ome.tif', np.stack(planes), ome=True,
                     metadata={'axes': 'CYX'})


@pytest.mark.parametrize('writer', [_imagej, _channel_last, _ome],
                         ids=['imagej', 'channel-last', 'ome-tiff'])
def test_a_multichannel_file_and_its_channel_file_give_identical_masks(
        qtbot, tmp_path, writer):
    planes = _planes(4)
    multi, single = tmp_path / 'multi', tmp_path / 'single'
    multi.mkdir()
    single.mkdir()
    writer(multi, planes)
    tifffile.imwrite(single / 'field.tif', planes[2])
    PLANE['channel'] = 2
    a = _open(qtbot, multi)
    b = _open(qtbot, single)
    assert a._field_channels is not None and len(a._field_channels.planes) == 4
    assert a._field_channels.channel == 2 and a._field_channels.view is not None
    for plane, raw in zip(a._field_channels.planes, planes):
        assert plane.dtype == raw.dtype and plane.tobytes() == raw.tobytes()
    assert a._canvas.image.dtype == b._canvas.image.dtype
    assert a._canvas.image.tobytes() == b._canvas.image.tobytes()
    a._on_detect_otsu()
    b._on_detect_otsu()
    assert int(a._canvas.mask.max()) >= 2
    assert a._canvas.mask.dtype == b._canvas.mask.dtype
    assert a._canvas.mask.tobytes() == b._canvas.mask.tobytes()


def test_another_channel_of_a_reviewed_plane_opens_with_its_own_mask(qtbot, tmp_path):
    planes = _planes(3)
    _imagej(tmp_path, planes)
    screen = _open(qtbot, tmp_path)
    first = screen._masks_dir
    screen._canvas.mask[5:15, 5:15] = 4
    screen._on_save()
    _choose_channel(qtbot, screen, 1)
    qtbot.waitUntil(lambda: not screen._loading and screen._load_worker is None)
    assert screen._canvas.image.tobytes() == planes[1].tobytes()
    assert screen._masks_dir != first and not screen._canvas.mask.any()
    assert [layer.name for layer in screen._field_stack][:3] == [
        'Channel 1', 'Channel 2', 'Channel 3']
    _choose_channel(qtbot, screen, 0)
    qtbot.waitUntil(lambda: not screen._loading and screen._load_worker is None)
    assert screen._masks_dir == first and int(screen._canvas.mask[10, 10]) != 0


def test_an_rgb_picture_edits_on_grey_or_on_one_colour(qtbot, tmp_path):
    rgb = np.stack(_planes(3, np.uint8), axis=-1)
    imageio.imwrite(tmp_path / 'field.png', rgb)
    screen = _open(qtbot, tmp_path)
    from spacr.qt import mask_engine as engine

    grey, _mask = engine.load_image_and_mask(str(tmp_path), 'field.png')
    assert screen._field_channels.colour and screen._field_channels.channel is None
    assert screen._canvas.image.tobytes() == grey.tobytes()
    assert [l.name for l in screen._field_stack][:3] == ['Red', 'Green', 'Blue']
    _choose_channel(qtbot, screen, 1)
    assert screen._canvas.image.tobytes() == engine._as_field_image(
        rgb[..., 1], 'x').tobytes()


def test_every_channel_and_the_mask_are_layers(qtbot, tmp_path):
    _imagej(tmp_path, _planes(4))
    screen = _open(qtbot, tmp_path)
    stack = screen._field_stack
    assert stack.names == ('Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Mask')
    assert all(isinstance(stack[i], ImageLayer) for i in range(4))
    mask = stack['Mask']
    assert isinstance(mask, LabelsLayer) and mask is screen._canvas.mask_layer
    assert mask.data is screen._canvas.mask
    assert screen._canvas.layered and screen._canvas.draws_layers()
    assert stack['Channel 1'].colormap.name != stack['Channel 2'].colormap.name
    from spacr.layers import to_rgba
    from spacr.qt.screens.make_masks import MASK_ROLE_COLOURS

    assert mask.color == to_rgba(MASK_ROLE_COLOURS['object'])
    role = screen._mask_role_combo
    role.setCurrentIndex(role.findData('nucleus'))
    assert mask.color == to_rgba(MASK_ROLE_COLOURS['nucleus'])
    assert 'Nucleus mask' in stack.names


def test_the_role_colours_are_the_outline_palette():
    from spacr.plot import OUTLINE_PALETTES
    from spacr.qt.screens.make_masks import MASK_ROLE_COLOURS

    for role in ('cell', 'nucleus', 'pathogen'):
        assert MASK_ROLE_COLOURS[role] == OUTLINE_PALETTES['default'][role]
    assert MASK_ROLE_COLOURS['object'] == OUTLINE_PALETTES['default']['organelle']


def _picture(screen):
    return screen._canvas._composed[2].copy()


def test_layer_controls_repaint_without_rebuilding(qtbot, tmp_path):
    _imagej(tmp_path, _planes(3))
    screen = _open(qtbot, tmp_path)
    stack, panel = screen._field_stack, screen._layer_panel
    layers = list(stack)
    widgets = (panel, screen._canvas, screen._mask_channel_combo)
    before = _picture(screen)

    def changed():
        qtbot.waitUntil(lambda: screen._canvas._composed is not None
                        and not np.array_equal(_picture(screen), before))
        return _picture(screen)

    stack.select(stack['Channel 2'])
    item = panel.layer_list.item(stack.names[::-1].index('Channel 2'))
    item.setCheckState(Qt.Unchecked)
    assert not stack['Channel 2'].visible
    before = changed()
    panel.opacity_slider.setValue(40)
    assert stack['Channel 2'].opacity == pytest.approx(0.4)
    stack['Channel 2'].visible = True
    before = changed()
    panel.colormap_combo.setCurrentText('red')
    assert stack['Channel 2'].colormap.name == 'red'
    before = changed()
    panel.high_spin.setValue(1000.0)
    panel.low_spin.setValue(10.0)
    assert stack['Channel 2'].contrast_limits(0) == (10.0, 1000.0)
    before = changed()
    panel.blending_combo.setCurrentIndex(panel.blending_combo.findData('opaque'))
    assert stack['Channel 2'].blending == 'opaque'
    before = changed()
    panel.raise_button.click()
    assert stack.names.index('Channel 2') == 2
    stack.select(screen._canvas.mask_layer)
    screen._canvas.mask[5:30, 5:30] = 1
    screen._canvas.refresh()
    before = _picture(screen)
    panel.labels_combo.setCurrentIndex(1)
    assert screen._canvas.mask_layer.contour == 1
    changed()
    assert list(stack) == [layers[0], layers[2], layers[1], layers[3]]
    assert (panel, screen._canvas, screen._mask_channel_combo) == widgets


def test_a_single_channel_field_keeps_its_picture_until_restyled(qtbot, tmp_path):
    tifffile.imwrite(tmp_path / 'field.tif', _planes(1)[0])
    screen = _open(qtbot, tmp_path)
    canvas = screen._canvas
    assert screen._field_channels is None
    assert screen._field_stack.names == ('Image', 'Mask')
    assert not canvas.layered and not canvas.draws_layers()
    screen._field_stack['Mask'].opacity = 0.8
    qtbot.waitUntil(lambda: canvas.draws_layers() and canvas._composed is not None)
    screen._reset_layers_button.click()
    assert not canvas.layered and screen._field_stack['Mask'].opacity == 0.5


def test_a_stroke_redraws_only_its_box_and_matches_a_full_composite(qtbot, tmp_path):
    _imagej(tmp_path, _planes(3))
    screen = _open(qtbot, tmp_path)
    canvas = screen._canvas
    canvas.mask_layer.contour = 1
    qtbot.waitUntil(lambda: canvas._composed is not None
                    and canvas._composed[0][3][-1] == canvas.layers_version)
    stack = screen._field_stack
    calls = []
    gathered = stack._render_uint8_gathered

    def spy(view, rows=slice(None), cols=slice(None)):
        calls.append((rows, cols))
        return gathered(view, rows, cols)

    stack._render_uint8_gathered = spy
    canvas.mask[40:52, 50:61] = 9
    canvas.refresh()
    assert len(calls) == 1 and calls[0] != (slice(None), slice(None))
    partial = _picture(screen)
    canvas.layers_version += 1
    canvas.refresh()
    np.testing.assert_array_equal(partial, _picture(screen))


def test_a_large_view_is_composited_off_the_gui_thread(qtbot, tmp_path, monkeypatch):
    from spacr.qt.screens import make_masks

    _imagej(tmp_path, _planes(2, shape=(160, 200)))
    monkeypatch.setattr(make_masks, 'LAYERS_SYNC_PIXELS', 100)
    threads = []
    original = make_masks._layers_picture_for

    def record(request):
        threads.append(threading.current_thread())
        return original(request)

    monkeypatch.setattr(make_masks, '_layers_picture_for', record)
    screen = _open(qtbot, tmp_path)
    canvas = screen._canvas
    qtbot.waitUntil(lambda: canvas._composed is not None and bool(threads))
    assert all(thread is not threading.main_thread() for thread in threads)
    screen._field_stack['Channel 1'].visible = False
    qtbot.waitUntil(lambda: canvas._composed is not None
                    and canvas._composed[0][3][-1] == canvas.layers_version)
    assert len(threads) >= 2
    assert canvas.close_layers()


def test_channel_planes_are_read_off_the_gui_thread(qtbot, tmp_path, monkeypatch):
    from spacr.qt.screens import make_masks as engine
    from spacr.qt.screens.make_masks import MakeMasksScreen

    _imagej(tmp_path, _planes(3))
    monkeypatch.setattr(MakeMasksScreen, '_should_background_load',
                        staticmethod(lambda path: True))
    threads = []
    original = engine._load_field_pair

    def record(*args, **kwargs):
        threads.append(threading.current_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, '_load_field_pair', record)
    screen = _open(qtbot, tmp_path)
    assert threads and all(t is not threading.main_thread() for t in threads)
    assert screen._field_channels is not None
    assert all(limit[1] > limit[0] for limit in screen._field_channels.limits)


def test_a_short_window_folds_the_panels_before_the_image(qtbot, tmp_path):
    tifffile.imwrite(tmp_path / 'field.tif', _planes(1)[0])
    screen = _open(qtbot, tmp_path)
    screen.show()
    image = screen._image_container
    font = screen.font()
    font.setPixelSize(30)
    screen.setFont(font)
    assert screen._canvas_room()[1] >= 300
    view = screen._view_tabs.currentWidget()
    screen.resize(1100, 600)
    qtbot.waitUntil(lambda: image.is_collapsed('Bottom bar')
                    or view.height() < screen._canvas_room()[1], timeout=15000)
    screen._keep_the_canvas_usable()
    assert image.is_collapsed('Bottom bar')
    folded = [name for _splitter, name, _axis in screen._folded_for_room]
    assert 'Bottom bar' in folded
    # 2026-11-01 maintainer decision: the title folds to one line (its
    # instruction) before the bottom bar folds.
    assert 'Title line' in folded
    assert folded.index('Title line') < folded.index('Bottom bar')
    assert screen._header.instruction_label.isHidden()
    screen.resize(1800, 1300)
    qtbot.waitUntil(lambda: not image.is_collapsed('Bottom bar')
                    or view.height() > 2 * screen._canvas_room()[1], timeout=15000)
    screen._keep_the_canvas_usable()
    assert not image.is_collapsed('Bottom bar')
    assert not screen._header.instruction_label.isHidden()


def test_reset_buttons_are_neutral_and_label_steps_are_named(qtbot, tmp_path):
    tifffile.imwrite(tmp_path / 'field.tif', _planes(1)[0])
    screen = _open(qtbot, tmp_path)
    from spacr.qt.button_roles import action_role

    for button in (screen._btn_reset_zoom, screen._reset_layers_button):
        assert action_role(button.text()) is None
        assert button.property('buttonActionRole') is None
    from PySide6.QtWidgets import QToolButton

    names = {button.objectName(): button.accessibleName()
             for button in screen.findChildren(QToolButton)
             if button.objectName() in ('MakeMasksLabelDecrease', 'MakeMasksLabelIncrease')}
    assert names == {'MakeMasksLabelDecrease': 'Previous label',
                     'MakeMasksLabelIncrease': 'Next label'}
