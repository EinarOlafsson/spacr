"""Shared-reader entry keeps the exact source and explicitly reviewed plane."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image

from spacr import convert
from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MakeMasksScreen, QInputDialog


@pytest.fixture
def screen(qtbot, monkeypatch, trace_mask_reader_native_jobs):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.warnings = []
    monkeypatch.setattr(widget, '_warn', lambda title, message: widget.warnings.append((title, message)))
    return widget


def _settle(qtbot, screen):
    qtbot.waitUntil(lambda: not screen._loading and screen._load_worker is None
                   and screen._reader_worker is None, timeout=20000)


def test_exact_picker_does_not_substitute_a_supported_sibling(screen, qtbot, tmp_path):
    selected = tmp_path / 'selected.png'
    other = tmp_path / 'a-sibling.tif'
    Image.fromarray(np.full((24, 32), 193, np.uint8)).save(selected)
    tifffile.imwrite(other, np.zeros((24, 32), np.uint16))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert screen._on_pick_file(str(selected))
    _settle(qtbot, screen)
    assert screen._image_files == [selected.name]
    np.testing.assert_array_equal(engine.read_image(selected), np.full((24, 32), 193, np.uint8))
    np.testing.assert_array_equal(screen._canvas.image, np.full((24, 32), 65535, np.uint16))
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    assert not screen.warnings


def test_real_tiff_series_time_z_and_channel_keep_separate_masks(screen, qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'acquired.ome.tif'
    pixels = np.arange(2 * 2 * 3 * 24 * 32, dtype=np.uint16).reshape(2, 2, 3, 24, 32)
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((24, 32), np.uint16), metadata={'axes': 'YX'},
                     resolution=(1e7 / 3.5, 1e7 / 3.5), resolutionunit='CENTIMETER')
        writer.write(pixels, metadata={'axes': 'TCZYX'},
                     resolution=(1e7 / 17.25, 1e7 / 17.25), resolutionunit='CENTIMETER')
    original = path.read_bytes()
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: (args[3][1], True))
    choices = iter((1, 2, 1))
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (next(choices), True))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    np.testing.assert_array_equal(screen._canvas.image, pixels[1, 1, 2])
    first = screen._masks_dir
    labels = np.zeros((24, 32), np.uint16)
    labels[3:8, 4:9] = 73
    saved = engine.save_mask(str(tmp_path), path.name, labels, masks_dir=first, preserve_ids=True)
    from spacr.io import tiff_pixel_size_nm
    assert engine.pixel_size_nm(path) == pytest.approx(17.25, rel=1e-8)
    assert tiff_pixel_size_nm(saved) == pytest.approx(17.25, rel=1e-8)
    saved_view = engine.load_mask_metadata(str(tmp_path), path.name, first)['image_view']
    assert (saved_view['series'], saved_view['time'], saved_view['z'], saved_view['channel']) == (1, 1, 2, 1)
    assert saved_view['pixel_size_nm'] == pytest.approx(17.25, rel=1e-8)
    header = engine._container_image_header(path)
    second = engine._set_container_image_view(header, series=1, time=0, z=0, channel=0)
    assert second['masks_dir'] != first
    old_image, old_labels = engine.load_image_and_mask(str(tmp_path), path.name, first)
    np.testing.assert_array_equal(old_image, pixels[1, 1, 2])
    np.testing.assert_array_equal(old_labels, labels)
    new_image, new_labels = engine.load_image_and_mask(str(tmp_path), path.name, second['masks_dir'])
    np.testing.assert_array_equal(new_image, pixels[0, 0, 0])
    assert not new_labels.any() and path.read_bytes() == original
    changed = tmp_path / 'changed-calibration.tif'
    changed.write_bytes(original)
    engine._set_container_image_view(engine._container_image_header(changed), series=1)
    assert engine.pixel_size_nm(changed) == pytest.approx(17.25, rel=1e-8)
    tifffile.imwrite(changed, np.ones((24, 32), np.uint16),
                     resolution=(1e7 / 3.5, 1e7 / 3.5), resolutionunit='CENTIMETER')
    assert engine.pixel_size_nm(changed) == pytest.approx(3.5, rel=1e-8)
    assert path.read_bytes() == original


@pytest.mark.parametrize('suffix', ['.gif', '.webp'])
def test_installed_raster_reader_uses_real_bytes_and_explicit_frame_channel(screen, qtbot, tmp_path, monkeypatch, suffix):
    path = tmp_path / ('color' + suffix)
    first = Image.fromarray(np.full((24, 32, 3), (21, 61, 121), np.uint8))
    if suffix == '.gif':
        first.save(path, save_all=True, append_images=[Image.fromarray(np.full((24, 32, 3), (51, 91, 151), np.uint8))])
        choices = iter((1, 1))
        expected = 91
    else:
        first.save(path, lossless=True)
        choices = iter((1,))
        expected = 61
    original = path.read_bytes()
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (next(choices), True))
    assert path.name in engine.list_images(str(tmp_path))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert screen._image_files == [path.name]
    np.testing.assert_array_equal(engine.read_image(path), np.full((24, 32), expected, np.uint8))
    np.testing.assert_array_equal(screen._canvas.image, np.full((24, 32), 65535, np.uint16))
    assert not screen.warnings and path.read_bytes() == original


def test_series_review_cancel_preserves_existing_editor_and_queue(screen, qtbot, tmp_path, monkeypatch):
    initial = tmp_path / 'initial.png'
    Image.fromarray(np.full((24, 32), 71, np.uint8)).save(initial)
    assert screen.open_paths([initial])
    _settle(qtbot, screen)
    before = screen._canvas.image.copy(), list(screen._image_files), screen._masks_dir
    path = tmp_path / 'series.tif'
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((24, 32), np.uint16))
        writer.write(np.ones((25, 33), np.uint16))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('', False))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    np.testing.assert_array_equal(screen._canvas.image, before[0])
    assert (screen._image_files, screen._masks_dir) == before[1:]
    assert not screen.warnings
    assert screen.open_paths([initial, path])
    _settle(qtbot, screen)
    assert screen._image_files == [initial.name, path.name]
    screen._go_to_index(1)
    _settle(qtbot, screen)
    assert screen._current_index == 0 and screen._folder == str(tmp_path)
    assert screen._masks_dir == before[2] and screen._image_files == [initial.name, path.name]
    np.testing.assert_array_equal(screen._canvas.image, before[0])
    assert not screen.warnings


def test_missing_vendor_reader_refuses_exact_file_without_decoding_or_sibling(screen, qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'not-a-decoded-fixture.nd2'
    path.write_bytes(b'only the missing-reader admission boundary is under test')
    tifffile.imwrite(tmp_path / 'sibling.tif', np.zeros((24, 32), np.uint16))
    monkeypatch.setattr(convert, 'reader_available', lambda ext: False if ext == '.nd2' else True)
    monkeypatch.setattr(convert, '_describe', lambda *args: pytest.fail('missing reader reached metadata'))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert screen.warnings and 'pip install nd2reader' in screen.warnings[-1][1]
    assert not screen._image_files


def test_excessive_real_tiff_series_preserve_the_open_field_without_decoding(screen, qtbot, tmp_path, monkeypatch):
    initial = tmp_path / 'initial.png'
    pixels = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
    Image.fromarray(pixels).save(initial)
    assert screen.open_paths([initial])
    _settle(qtbot, screen)
    displayed = screen._canvas.image.copy()
    state = (list(screen._image_files), screen._folder,
             screen._current_index, screen._masks_dir)
    assert not screen.warnings

    excessive = tmp_path / 'too-many-series.tif'
    with tifffile.TiffWriter(excessive) as writer:
        for index in range(513):
            writer.write(np.full((2, 2), index % 256, np.uint8),
                         photometric='minisblack', metadata={'axes': 'YX'})
    with tifffile.TiffFile(excessive) as handle:
        assert len(handle.series) == 513
        assert all(series.shape == (2, 2) for series in handle.series)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()
              if path.is_file()}
    identity = engine._source_identity(excessive)

    def refuse_decode(*args, **kwargs):
        pytest.fail('Excessive series must be refused before decoding pixels')

    monkeypatch.setattr(tifffile.TiffPageSeries, 'asarray', refuse_decode)
    assert screen.open_paths([excessive])
    _settle(qtbot, screen)
    assert len(screen.warnings) == 1
    assert 'too many series to review safely' in screen.warnings[0][1]
    np.testing.assert_array_equal(screen._canvas.image, displayed)
    np.testing.assert_array_equal(engine.read_image(initial), pixels)
    assert (screen._image_files, screen._folder,
            screen._current_index, screen._masks_dir) == state
    assert engine._source_identity(excessive) == identity
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()
                      if path.is_file()}


def test_a_reviewed_file_replaced_by_a_directory_preserves_the_open_field(screen, qtbot, tmp_path, monkeypatch):
    initial = tmp_path / 'initial.png'
    pixels = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
    Image.fromarray(pixels).save(initial)
    assert screen.open_paths([initial])
    _settle(qtbot, screen)
    displayed = screen._canvas.image.copy()
    state = (list(screen._image_files), screen._folder,
             screen._current_index, screen._masks_dir)
    initial_bytes = initial.read_bytes()
    assert not screen.warnings

    selected = tmp_path / 'reviewed.tif'
    backup = tmp_path / 'reviewed.backup'
    with tifffile.TiffWriter(selected) as writer:
        writer.write(np.zeros((24, 32), np.uint16))
        writer.write(np.ones((25, 33), np.uint16))
    source_bytes = selected.read_bytes()
    with tifffile.TiffFile(selected) as handle:
        assert [series.shape for series in handle.series] == [(24, 32), (25, 33)]
    decisions = []

    def replace_at_series_choice(parent, title, prompt, labels, current, editable):
        assert parent is screen
        assert title == 'Image series'
        assert prompt == 'Choose the stored image series.'
        assert labels == ['Series 0: (24, 32)', 'Series 1: (25, 33)']
        assert current == 0 and editable is False
        selected.rename(backup)
        selected.mkdir()
        decisions.append(labels[0])
        return labels[0], True

    def refuse_decode(*args, **kwargs):
        pytest.fail('A directory replacing the reviewed file must not decode pixels')

    monkeypatch.setattr(QInputDialog, 'getItem', replace_at_series_choice)
    monkeypatch.setattr(tifffile.TiffPageSeries, 'asarray', refuse_decode)
    assert screen.open_paths([selected])
    _settle(qtbot, screen)
    assert decisions == ['Series 0: (24, 32)']
    assert len(screen.warnings) == 1
    assert 'not a regular file' in screen.warnings[0][1]
    assert selected.is_dir() and list(selected.iterdir()) == []
    assert backup.read_bytes() == source_bytes
    assert initial.read_bytes() == initial_bytes
    np.testing.assert_array_equal(screen._canvas.image, displayed)
    np.testing.assert_array_equal(engine.read_image(initial), pixels)
    assert (screen._image_files, screen._folder,
            screen._current_index, screen._masks_dir) == state
    assert engine._current_container_image_view(selected) is None
    assert engine._current_native_image_header(selected) is None


@pytest.mark.parametrize('suffix', ['.nd2', '.lif'])
def test_vendor_plane_contract_never_requests_an_unselected_plane(tmp_path, monkeypatch, suffix):
    path = tmp_path / ('reader-contract' + suffix)
    path.write_bytes(b'contract double; not a real vendor-format fixture')
    calls = []
    expected = np.full((24, 32), 2381, np.uint16)
    monkeypatch.setattr(convert, 'reader_available', lambda ext: True)
    metadata = dict(shape=(24, 32), axes='ND2' if suffix == '.nd2' else 'LIF', dtype='',
                    n_t=3, n_z=4, n_c=2, n_series=2,
                    reader='nd2reader' if suffix == '.nd2' else 'readlif')
    if suffix == '.lif':
        metadata['per_series'] = [dict(shape=(24, 32), n_t=3, n_z=4, n_c=2,
                                       lif_image=0, lif_tile=index) for index in range(2)]
    monkeypatch.setattr(convert, '_describe', lambda *args: metadata)
    class Reader:
        metadata = {'pixel_microns': .625}
        scale = (4., 4., 1., 1.)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            calls.append('closed')
        def get_frame_2D(self, **kwargs):
            calls.append(kwargs)
            return expected
        def get_frame(self, **kwargs):
            calls.append(kwargs)
            return expected
    monkeypatch.setattr(convert, '_import_reader', lambda ext: SimpleNamespace(ND2Reader=lambda path: Reader()))
    monkeypatch.setattr(convert, '_lif_images', lambda path: [Reader()])
    header = engine._container_image_header(path)
    assert calls == (['closed'] if suffix == '.nd2' else [])
    view = engine._set_container_image_view(header, series=1, time=2, z=3, channel=1)
    np.testing.assert_array_equal(engine._read_container_image_view(path), expected)
    frame_calls = [call for call in calls if isinstance(call, dict)]
    assert frame_calls == [dict(t=2, z=3, c=1, v=1) if suffix == '.nd2' else dict(z=3, t=2, c=1, m=1)]
    assert calls.count('closed') == (2 if suffix == '.nd2' else 0)
    expected_nm = 625. if suffix == '.nd2' else 250.
    assert engine.pixel_size_nm(path) == expected_nm
    labels = np.zeros((24, 32), np.uint16)
    labels[2:6, 3:8] = 71
    saved = engine.save_mask(str(tmp_path), path.name, labels, masks_dir=view['masks_dir'], preserve_ids=True)
    from spacr.io import tiff_pixel_size_nm
    assert tiff_pixel_size_nm(saved) == expected_nm
    saved_view = engine.load_mask_metadata(str(tmp_path), path.name, view['masks_dir'])['image_view']
    assert (saved_view['series'], saved_view['time'], saved_view['z'], saved_view['channel']) == (1, 2, 3, 1)
    assert saved_view['source_identity'] == list(header['identity']) and saved_view['pixel_size_nm'] == expected_nm
    assert '_S1_T2_Z3_C1_' in view['masks_dir']
    assert path.read_bytes() == b'contract double; not a real vendor-format fixture'


def test_changed_reviewed_source_and_oversized_series_refuse_before_decode(tmp_path, monkeypatch):
    path = tmp_path / 'original.tif'
    tifffile.imwrite(path, np.zeros((24, 32), np.uint16))
    header = engine._container_image_header(path)
    engine._set_container_image_view(header)
    path.write_bytes(path.read_bytes() + b'changed')
    monkeypatch.setattr(tifffile.TiffPageSeries, 'asarray', lambda *args, **kwargs: pytest.fail('changed source decoded'))
    with pytest.raises(ValueError, match='source changed'):
        engine._read_container_image_view(path)
    header = engine._container_image_header(path)
    header['series'][0]['shape'] = (20000, 20000)
    with pytest.raises(ValueError, match='256 MiB'):
        engine._set_container_image_view(header)


def test_reopening_changed_source_reviews_again_and_never_reuses_old_masks(screen, qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'changed.gif'
    first_pixels = np.full((24, 32, 3), 31, np.uint8)
    first_pixels[3:9, 4:11] = 7
    Image.fromarray(first_pixels).save(path)
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (0, True))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    old_target = screen._masks_dir
    labels = np.zeros((24, 32), np.uint16)
    labels[2:5, 3:7] = 19
    engine.save_mask(str(tmp_path), path.name, labels, masks_dir=old_target, preserve_ids=True)
    saved = Path(engine.mask_save_path(str(tmp_path), path.name, old_target)).read_bytes()
    second_pixels = np.full((24, 32, 3), 87, np.uint8)
    second_pixels[3:9, 4:11] = 12
    Image.fromarray(second_pixels).save(path)
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert screen._masks_dir != old_target and not screen._canvas.mask.any()
    np.testing.assert_array_equal(engine.read_image(path), second_pixels[..., 0])
    np.testing.assert_array_equal(screen._canvas.image,
                                  (second_pixels[..., 0] / 87 * 65535).astype(np.uint16))
    assert Path(engine.mask_save_path(str(tmp_path), path.name, old_target)).read_bytes() == saved
    with pytest.raises(ValueError, match='source changed'):
        engine.load_image_and_mask(str(tmp_path), path.name, old_target)


@pytest.mark.parametrize('suffix', ['.dm3', '.dm4', '.map'])
def test_real_tem_files_use_the_same_exact_queue_and_keep_calibration(screen, qtbot, tmp_path, suffix):
    pixels = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    path = tmp_path / ('electron' + suffix)
    if suffix == '.map':
        import mrcfile
        with mrcfile.new(path) as handle:
            handle.set_data(pixels.astype(np.int16))
            handle.voxel_size = 3.7
    else:
        from tests.tem_files import write_dm
        write_dm(path, pixels, version=int(suffix[-1]), scale=.37, unit='nm')
    before = path.read_bytes()
    assert path.name in engine.list_images(str(tmp_path))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert screen._image_files == [path.name] and not screen.warnings
    np.testing.assert_array_equal(engine.read_image(path), pixels)
    expected_display = pixels if suffix != '.map' else (pixels.astype(np.float64) / 767 * 65535).astype(np.uint16)
    np.testing.assert_array_equal(screen._canvas.image, expected_display)
    assert engine.pixel_size_nm(path) == pytest.approx(.37, rel=1e-6)
    assert path.read_bytes() == before


def test_mixed_numeric_and_raster_queue_keeps_explicit_channel_binding(screen, qtbot, tmp_path, monkeypatch):
    raster = tmp_path / 'a.png'
    Image.fromarray(np.full((24, 32), 9, np.uint8)).save(raster)
    array = tmp_path / 'b.npy'
    pixels = np.stack((np.full((24, 32), 11, np.uint16), np.full((24, 32), 211, np.uint16)), axis=-1)
    np.save(array, pixels)
    original = array.read_bytes()
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('YXC', True))
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (1, True))
    assert screen.open_paths([raster, array])
    _settle(qtbot, screen)
    assert screen._image_files == [raster.name, array.name]
    screen._go_to_index(1)
    _settle(qtbot, screen)
    np.testing.assert_array_equal(screen._canvas.image, pixels[..., 1])
    assert '_YXC_C1' in screen._masks_dir and not screen._canvas.mask.any()
    assert array.read_bytes() == original and not screen.warnings


def test_reviewed_later_tiff_plane_cannot_be_replaced_by_the_first_volume(screen, qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'first-volume.tif'
    selected = np.full((25, 33), 1561, np.uint16)
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((3, 24, 32), np.uint16), metadata={'axes': 'ZYX'})
        writer.write(selected, metadata={'axes': 'YX'})
    before = path.read_bytes()
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: (args[3][1], True))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert screen._mask_stage.session is None
    np.testing.assert_array_equal(screen._canvas.image, selected)
    assert path.read_bytes() == before and not screen.warnings


def test_evicted_exact_plane_binding_cannot_borrow_the_latest_plane_masks(tmp_path):
    path = tmp_path / 'planes.tif'
    tifffile.imwrite(path, np.zeros((2, 24, 32), np.uint16), metadata={'axes': 'CYX'})
    header = engine._container_image_header(path)
    first = engine._set_container_image_view(header, channel=0)
    second = engine._set_container_image_view(header, channel=1)
    labels = np.zeros((24, 32), np.uint16)
    labels[4:8, 7:10] = 33
    engine.save_mask(str(tmp_path), path.name, labels, masks_dir=first['masks_dir'], preserve_ids=True)
    for index in range(129):
        other = tmp_path / f'bounded-{index}.tif'
        other.write_bytes(path.read_bytes())
        engine._set_container_image_view(engine._container_image_header(other), channel=0)
    assert str(path) not in engine._CONTAINER_CURRENT_VIEWS
    assert not any(key[0] == str(path) for key in engine._CONTAINER_IMAGE_VIEWS)
    with pytest.raises(ValueError, match='Review this source'):
        engine.load_image_and_mask(str(tmp_path), path.name, first['masks_dir'])
    engine._set_container_image_view(engine._container_image_header(path), channel=1)
    _image, current_labels = engine.load_image_and_mask(str(tmp_path), path.name, second['masks_dir'])
    assert not current_labels.any()


def test_real_czi_scene_time_z_channel_reaches_native_pixels(screen, qtbot, tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / 'data/watch_vendor_f548/zeiss_czi/two_scenes_tzc.czi'
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    original = path.read_bytes()
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: (args[3][1], True))
    choices = iter((1, 2, 1))
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (next(choices), True))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    expected = np.full((32, 32), 40, np.uint16)
    expected[5:17, 5:17] = 500 + 37 + 11 + 14 + 3
    expected[18:29, 18:29] = 800 + 37 + 11 + 14 + 3
    np.testing.assert_array_equal(screen._canvas.image, expected)
    assert not screen.warnings and path.read_bytes() == original
    assert '_S1_T1_Z2_C1_' in screen._masks_dir
    assert engine.pixel_size_nm(path) is None
    saved = engine.save_mask(str(tmp_path), path.name, np.zeros((32, 32), np.uint16), masks_dir=screen._masks_dir)
    from spacr.io import tiff_pixel_size_nm
    assert tiff_pixel_size_nm(saved) is None
    saved_view = engine.load_mask_metadata(str(tmp_path), path.name, screen._masks_dir)['image_view']
    assert (saved_view['series'], saved_view['time'], saved_view['z'], saved_view['channel']) == (1, 1, 2, 1)
    assert saved_view['pixel_size_nm'] is None


def test_real_lif_tile_and_plane_preserve_the_acquired_pixel_values(screen, qtbot, tmp_path, monkeypatch):
    from readlif.reader import LifFile
    source = Path(__file__).parents[1] / 'data/watch_vendor_f548/leica_lif/PR2729_frameOrderCombinedScanTypes.lif'
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    original = path.read_bytes()
    image = next(iter(LifFile(str(path)).get_iter_image()))
    expected = np.asarray(image.get_frame(t=1, z=2, c=1, m=2))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: (args[3][2], True))
    choices = iter((1, 2, 1))
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (next(choices), True))
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    np.testing.assert_array_equal(engine.read_image(path), expected)
    np.testing.assert_array_equal(screen._canvas.image, (expected / float(expected.max()) * 65535).astype(np.uint16))
    assert not screen.warnings and path.read_bytes() == original
    assert '_S2_T1_Z2_C1_' in screen._masks_dir
    assert image.scale[0] > 0 and image.scale[0] == pytest.approx(image.scale[1])
    expected_nm = 1000. / image.scale[0]
    assert engine.pixel_size_nm(path) == pytest.approx(expected_nm)
    labels = np.zeros(expected.shape, np.uint16)
    labels[2:5, 3:8] = 83
    saved = engine.save_mask(str(tmp_path), path.name, labels, masks_dir=screen._masks_dir, preserve_ids=True)
    from spacr.io import tiff_pixel_size_nm
    assert tiff_pixel_size_nm(saved) == pytest.approx(expected_nm)
    np.testing.assert_array_equal(tifffile.imread(saved), labels)
    saved_view = engine.load_mask_metadata(str(tmp_path), path.name, screen._masks_dir)['image_view']
    assert (saved_view['series'], saved_view['time'], saved_view['z'], saved_view['channel']) == (2, 1, 2, 1)
    assert saved_view['pixel_size_nm'] == pytest.approx(expected_nm)


def test_folder_with_new_raster_and_numeric_image_preserves_both_queue_entries(screen, qtbot, tmp_path, monkeypatch):
    raster = tmp_path / 'first.gif'
    Image.fromarray(np.full((24, 32, 3), 37, np.uint8)).save(raster)
    array = tmp_path / 'second.npy'
    np.save(array, np.full((24, 32), 2021, np.uint16))
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (0, True))
    assert screen.open_paths([tmp_path])
    _settle(qtbot, screen)
    assert screen._image_files == [raster.name, array.name]
    screen._go_to_index(1)
    _settle(qtbot, screen)
    np.testing.assert_array_equal(screen._canvas.image, np.full((24, 32), 2021))
    assert original == {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    assert not screen.warnings


def test_tiff_series_and_native_volume_metadata_are_read_off_gui_thread(screen, qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication
    path = tmp_path / 'ordinary.tif'
    tifffile.imwrite(path, np.full((24, 32), 147, np.uint16))
    read_header = engine._container_image_header
    observed = []
    def inspect(path):
        observed.append(QThread.currentThread() is QApplication.instance().thread())
        return read_header(path)
    monkeypatch.setattr(engine, '_container_image_header', inspect)
    assert screen.open_paths([path])
    _settle(qtbot, screen)
    assert observed == [False] and not screen.warnings
    np.testing.assert_array_equal(screen._canvas.image, np.full((24, 32), 147))
    assert screen._masks_dir is None


def test_superseded_metadata_cannot_replace_the_new_exact_image(screen, qtbot, tmp_path, monkeypatch):
    from threading import Event
    blocked, release = Event(), Event()
    old = tmp_path / 'old.tif'
    tifffile.imwrite(old, np.zeros((24, 32), np.uint16))
    selected = tmp_path / 'new.png'
    Image.fromarray(np.full((24, 32), 179, np.uint8)).save(selected)
    inspect = engine._container_image_header
    def delayed(path):
        blocked.set()
        assert release.wait(3)
        return inspect(path)
    monkeypatch.setattr(engine, '_container_image_header', delayed)
    assert screen.open_paths([old])
    qtbot.waitUntil(blocked.is_set, timeout=2000)
    assert screen.open_paths([selected])
    release.set()
    _settle(qtbot, screen)
    qtbot.waitUntil(lambda: not screen._reader_workers, timeout=20000)
    assert screen._image_files == [selected.name] and not screen.warnings
    np.testing.assert_array_equal(engine.read_image(selected), np.full((24, 32), 179, np.uint8))
    np.testing.assert_array_equal(screen._canvas.image, np.full((24, 32), 65535, np.uint16))


@pytest.mark.parametrize('suffix', ['.tif', '.gif'])
def test_queues_larger_than_metadata_cache_open_and_review_only_the_current_field(screen, qtbot, tmp_path, monkeypatch, suffix):
    for index in range(129):
        path = tmp_path / f'field{index:03}{suffix}'
        pixels = np.full((24, 32), index, np.uint8)
        if suffix == '.tif':
            tifffile.imwrite(path, pixels)
        else:
            Image.fromarray(pixels).save(path)
    inspected = []
    inspect = engine._container_image_header
    def count(path):
        inspected.append(Path(path).name)
        return inspect(path)
    monkeypatch.setattr(engine, '_container_image_header', count)
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (0, True))
    assert screen.open_paths([tmp_path])
    _settle(qtbot, screen)
    assert len(screen._image_files) == 129 and inspected == [f'field000{suffix}']
    screen._go_to_index(128)
    _settle(qtbot, screen)
    assert inspected == [f'field000{suffix}', f'field128{suffix}']
    np.testing.assert_array_equal(engine.read_image(tmp_path / f'field128{suffix}'), np.full((24, 32), 128, np.uint8))
    np.testing.assert_array_equal(screen._canvas.image, np.full((24, 32), 65535, np.uint16))
    assert screen._current_index == 128 and not screen.warnings


def test_uncalibrated_tiff_header_does_not_import_pipeline_libraries(tmp_path, monkeypatch):
    import builtins

    source = tmp_path / 'ordinary.tif'
    tifffile.imwrite(source, np.arange(48, dtype=np.uint16).reshape(6, 8),
                     metadata={'axes': 'YX'})
    before = source.read_bytes()
    original_import = builtins.__import__

    def metadata_import(name, globals=None, locals=None, fromlist=(), level=0):
        pipeline = name in ('spacr.core', 'spacr.io') or (
            level == 2 and name in ('core', 'io')
            and (globals or {}).get('__package__') == 'spacr.qt')
        if pipeline:
            raise AssertionError('Uncalibrated image metadata must not load pipeline libraries')
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as context:
        context.setattr(builtins, '__import__', metadata_import)
        header = engine._container_image_header(str(source))
    assert len(header['series']) == 1
    assert header['series'][0]['axes'] == 'YX'
    assert header['series'][0]['shape'] == (6, 8)
    assert header['series'][0]['dtype'] == 'uint16'
    assert header['series'][0]['pixel_size_nm'] is None
    assert source.read_bytes() == before
