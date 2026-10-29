"""Numeric NPY selection reaches native editable scenes with reviewed axes."""
import os
from pathlib import Path

import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MakeMasksScreen, QInputDialog


@pytest.fixture
def screen(qtbot, monkeypatch, trace_mask_reader_native_jobs):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.warnings = []
    monkeypatch.setattr(widget, '_warn', lambda title, text: widget.warnings.append((title, text)))
    return widget


def _settle(qtbot, screen):
    qtbot.waitUntil(lambda: not screen._loading and screen._load_worker is None, timeout=20000)


def test_numeric_yx_opens_the_selected_npy_with_original_pixels(screen, qtbot, tmp_path, monkeypatch):
    pixels = np.arange(40 * 48, dtype=np.uint16).reshape(40, 48)
    path = tmp_path / 'selected.npy'
    np.save(path, pixels)
    original = path.read_bytes()
    np.save(tmp_path / 'other.npy', pixels + 5)
    monkeypatch.setattr(screen, '_should_background_load', lambda path: True)
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert screen._image_files == ['selected.npy']
    np.testing.assert_array_equal(screen._canvas.image, pixels)
    assert not screen._canvas.mask.any()
    assert not screen.warnings
    assert path.read_bytes() == original


def test_canonical_merged_npy_reviews_channel_and_keeps_mask_destinations_separate(
        screen, qtbot, tmp_path, monkeypatch):
    merged = tmp_path / 'plate1' / 'merged'
    merged.mkdir(parents=True)
    path = merged / 'AssayPlate%5FPerkinElmer%5FCellCarrier-384_A03_8_1.npy'
    pixels = np.stack([np.arange(40 * 48, dtype=np.uint16).reshape(40, 48) + i * 2000
                       for i in range(4)], axis=-1)
    np.save(path, pixels)
    original = path.read_bytes()
    seen = []
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: seen.append(args) or ('YXC', True))
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (1, True))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert seen and '(40, 48, 4)' in seen[0][2]
    np.testing.assert_array_equal(screen._canvas.image, pixels[..., 1])
    assert not screen._canvas.mask.any()
    first_target = screen._masks_dir
    labels = np.zeros((40, 48), np.uint16)
    labels[3:8, 7:13] = 37
    engine.save_mask(str(merged), path.name, labels, masks_dir=first_target, preserve_ids=True)
    first_saved = Path(engine.mask_save_path(str(merged), path.name, first_target))
    saved_bytes = first_saved.read_bytes()
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (2, True))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert screen._masks_dir != first_target
    np.testing.assert_array_equal(screen._canvas.image, pixels[..., 2])
    assert not screen._canvas.mask.any()
    assert first_saved.read_bytes() == saved_bytes and path.read_bytes() == original
    restored, mask = engine.load_image_and_mask(str(merged), path.name, first_target)
    np.testing.assert_array_equal(restored, pixels[..., 1])
    np.testing.assert_array_equal(mask, labels)


@pytest.mark.parametrize('axes', ['CYX', 'ZYX', 'ZYXC', 'CZYX'])
def test_reviewed_array_axes_open_real_native_image_or_volume(screen, qtbot, tmp_path, monkeypatch, axes):
    spatial = np.arange(3 * 24 * 32, dtype=np.uint16).reshape(3, 24, 32)
    arrays = {'CYX': np.stack((spatial[0], spatial[0] + 2000)),
              'ZYX': spatial, 'ZYXC': np.stack((spatial, spatial + 2000), axis=-1),
              'CZYX': np.stack((spatial, spatial + 2000))}
    path = tmp_path / 'reviewed.npy'
    np.save(path, arrays[axes])
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: (axes, True))
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (1, True))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert not screen.warnings
    if axes == 'CYX':
        np.testing.assert_array_equal(screen._canvas.image, spatial[0] + 2000)
    else:
        session = screen._mask_stage.session
        assert session is not None and session.axes == 'ZYX'
        np.testing.assert_array_equal(session.image, spatial if axes == 'ZYX' else spatial + 2000)
        assert not session.mask.any()


def test_npy_review_cancel_or_invalid_object_leaves_existing_scene_unchanged(screen, qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'first.npy'
    np.save(path, np.arange(40 * 48, dtype=np.uint16).reshape(40, 48))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    original_image, original_target = screen._canvas.image.copy(), screen._masks_dir
    choices = tmp_path / 'channels.npy'
    np.save(choices, np.zeros((40, 48, 4), np.uint16))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('YXC', False))
    assert not screen.open_paths([choices])
    unsafe = tmp_path / 'unsafe.npy'
    np.save(unsafe, np.array([{'secret': 'not unpickled'}], dtype=object))
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('invalid data reached the array loader'))
    assert not screen.open_paths([unsafe])
    assert screen.warnings
    np.testing.assert_array_equal(screen._canvas.image, original_image)
    assert screen._masks_dir == original_target and screen._image_files == [path.name]


@pytest.mark.parametrize('pixels', [np.zeros(3), np.zeros((1, 2, 3, 4, 5)),
                                   np.zeros((3, 4), complex), np.zeros((3, 4), dtype='U2'),
                                   np.zeros((3, 4), dtype=[('x', 'u2')])])
def test_unsupported_npy_headers_fail_before_pixels_are_loaded(tmp_path, monkeypatch, pixels):
    path = tmp_path / 'unsupported.npy'
    np.save(path, pixels)
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('header validation decoded pixels'))
    with pytest.raises(ValueError, match='NPY images require'):
        engine._npy_image_header(path)


@pytest.mark.parametrize('prefix', [b'notNPY\x01\x00\x00\x00',
                                   b'\x93NUMPY\x04\x00',
                                   b'\x93NUMPY\x01\x00\x00\x00'],
                         ids=['wrong-magic', 'unsupported-version', 'zero-header-length'])
def test_malformed_npy_prefixes_refuse_public_loading_without_decoding(tmp_path, monkeypatch, prefix):
    source = tmp_path / 'malformed.npy'
    source.write_bytes(prefix)
    reviewed = dict(engine._NPY_IMAGE_VIEWS)

    def refuse_array_load(*args, **kwargs):
        pytest.fail('Malformed NPY metadata must not reach array loading or pickle')

    monkeypatch.setattr(np, 'load', refuse_array_load)
    with pytest.raises(ValueError, match='NPY images require complete real numeric'):
        engine.load_image_and_mask(str(tmp_path), source.name)
    assert source.read_bytes() == prefix
    assert list(tmp_path.iterdir()) == [source]
    assert engine._NPY_IMAGE_VIEWS == reviewed


def test_npy_truncation_and_reviewed_source_changes_fail_closed(tmp_path):
    path = tmp_path / 'image.npy'
    np.save(path, np.ones((3, 4, 2), np.uint16))
    engine._set_npy_image_view(path, 'YXC', 0)
    before = path.stat()
    np.save(path, np.ones((3, 4, 2), np.uint16) * 2)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
    with pytest.raises(ValueError, match='source changed'):
        engine._read_npy_image_view(path)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match='NPY images require'):
        engine._npy_image_header(path)


def test_npy_only_folder_asks_for_the_exact_file_and_keeps_bundle_route_separate(
        screen, qtbot, tmp_path, monkeypatch):
    for name, offset in [('first.npy', 0), ('second.npy', 2000)]:
        np.save(tmp_path / name, np.arange(40 * 48, dtype=np.uint16).reshape(40, 48) + offset)
    selected = tmp_path / 'second.npy'
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: (str(selected), True))
    assert screen.open_paths([tmp_path])
    _settle(qtbot, screen)
    assert screen._image_files == [selected.name]
    np.testing.assert_array_equal(screen._canvas.image, np.arange(40 * 48, dtype=np.uint16).reshape(40, 48) + 2000)
    bundle = tmp_path / 'cell_seg.npy'
    np.save(bundle, np.zeros((40, 48), np.uint16))
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('numeric route opened a Cellpose bundle'))
    with pytest.raises(ValueError, match='separate routes'):
        engine._npy_image_header(bundle)


def test_invalid_npy_folder_and_disappearing_folder_refuse_honestly(screen, tmp_path, monkeypatch):
    invalid = tmp_path / 'invalid'
    invalid.mkdir()
    np.save(invalid / 'only.npy', np.zeros((4, 5), dtype=complex))
    assert not screen.open_paths([invalid])
    assert screen.warnings and 'real numeric' in screen.warnings[-1][1]
    assert screen._canvas.image is None
    monkeypatch.setattr(os, 'scandir', lambda path: (_ for _ in ()).throw(FileNotFoundError('source disappeared')))
    assert not screen.open_paths([invalid])
    assert 'source disappeared' in screen.warnings[-1][1]
    assert screen._canvas.image is None


@pytest.mark.parametrize('entry', ['picker', 'first-in-queue', 'later-in-queue'])
def test_cancelled_numeric_channel_keeps_the_displayed_field_and_its_masks(
        screen, qtbot, tmp_path, monkeypatch, entry):
    from PIL import Image

    raster = tmp_path / 'initial.png'
    Image.fromarray(np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)).save(raster)
    selected = tmp_path / 'channels.npy'
    np.save(selected, np.arange(24 * 32 * 2, dtype=np.uint16).reshape(24, 32, 2))
    originals = {path: path.read_bytes() for path in (raster, selected)}
    assert screen.open_paths([raster])
    _settle(qtbot, screen)
    displayed = screen._canvas.image.copy()
    target = screen._masks_dir
    previous_views = dict(engine._NPY_IMAGE_VIEWS)
    prompts = []
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('YXC', True))

    def cancel_channel(parent, title, prompt, current, minimum, maximum):
        assert parent is screen and title == 'NPY image layout'
        assert 'Label-mask planes are not imported automatically' in prompt
        assert (current, minimum, maximum) == (0, 0, 1)
        prompts.append('channel')
        return 0, False

    monkeypatch.setattr(QInputDialog, 'getInt', cancel_channel)
    if entry == 'picker':
        assert not screen._on_pick_file(str(selected))
    elif entry == 'first-in-queue':
        assert not screen._open_queue([str(selected), str(raster)])
    else:
        assert screen.open_paths([raster, selected])
        _settle(qtbot, screen)
        screen._go_to_index(1)
        _settle(qtbot, screen)
        assert screen._image_files == [raster.name, selected.name]
    assert prompts == ['channel']
    assert screen._current_index == 0 and screen._folder == str(tmp_path)
    assert screen._masks_dir == target and not screen.warnings
    np.testing.assert_array_equal(screen._canvas.image, displayed)
    assert not screen._canvas.mask.any()
    assert engine._NPY_IMAGE_VIEWS == previous_views
    assert str(selected) not in screen._npy_queue_targets
    assert all(path.read_bytes() == content for path, content in originals.items())
