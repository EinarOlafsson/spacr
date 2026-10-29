"""Native image choices and pending scientific edits keep their field identities."""
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import tifffile
from PIL import Image
from PySide6.QtCore import QPoint, Qt

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def editor(qtbot, qt_theme_applied, monkeypatch, trace_mask_reader_native_jobs):
    widget = mm.MakeMasksScreen()
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, '_confirm', lambda *args: True)
    yield widget
    assert mm._wait_for_mask_writes(timeout=10)
    widget.close()


def settle(qtbot, editor):
    qtbot.waitUntil(lambda: not editor._loading and editor._reader_worker is None
                   and editor._load_worker is None, timeout=20000)


def test_first_numeric_channel_and_later_cancel_return_to_the_actual_displayed_field(
        editor, qtbot, tmp_path, monkeypatch):
    array, middle, last = (tmp_path / name for name in ('a.npy', 'b.png', 'c.gif'))
    pixels = np.arange(24 * 32 * 2, dtype=np.uint16).reshape(24, 32, 2)
    np.save(array, pixels)
    Image.fromarray(np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)).save(middle)
    Image.fromarray(np.full((24, 32, 3), (17, 71, 139), np.uint8)).save(last)
    original = {path: path.read_bytes() for path in (array, middle, last)}
    monkeypatch.setattr(mm.QInputDialog, 'getItem', lambda *args: ('YXC', True))
    monkeypatch.setattr(mm.QInputDialog, 'getInt', lambda *args: (1, True))
    assert editor._open_queue([str(array), str(middle), str(last)])
    settle(qtbot, editor)
    np.testing.assert_array_equal(editor._canvas.image, pixels[..., 1])
    assert str(array) in editor._npy_queue_targets
    editor._btn_save_on_navigation.setChecked(False)
    assert editor._go_to_index(1)
    settle(qtbot, editor)
    displayed = editor._canvas.image.copy()
    monkeypatch.setattr(mm.QInputDialog, 'getInt', lambda *args: (0, False))
    assert editor._go_to_index(2)
    settle(qtbot, editor)
    assert editor._current_index == 1
    assert editor._image_files == [array.name, middle.name, last.name]
    np.testing.assert_array_equal(editor._canvas.image, displayed)
    assert engine._current_container_image_view(last) is None
    editor._set_mode(mm.MODE_PICK)
    editor._canvas.setFixedSize(400, 400)
    editor._canvas.refresh()
    assert editor._canvas._canvas_to_image(0, 0) is None
    mask = editor._canvas.mask.copy()
    qtbot.mouseClick(editor._canvas, Qt.LeftButton, pos=QPoint(0, 0))
    np.testing.assert_array_equal(editor._canvas.mask, mask)
    assert all(path.read_bytes() == content for path, content in original.items())


def test_time_cannot_start_during_an_owned_field_read(editor, qtbot, tmp_path, monkeypatch):
    source = tmp_path / 'pending.png'
    Image.fromarray(np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)).save(source)
    original = source.read_bytes()
    entered, release = Event(), Event()
    read = engine.load_image_and_mask

    def held_read(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return read(*args, **kwargs)

    monkeypatch.setattr(engine, 'load_image_and_mask', held_read)
    monkeypatch.setattr(editor, '_should_background_load', lambda path: True)
    try:
        assert editor.open_paths([source])
        qtbot.waitUntil(entered.is_set, timeout=3000)
        editor._toggle_label_time(True)
        assert not editor._time_label_enabled and not editor._btn_label_time.isChecked()
        assert editor._lineage is None
    finally:
        release.set()
    settle(qtbot, editor)
    assert not editor._time_label_enabled
    assert editor._image_files == [source.name] and source.read_bytes() == original


def test_folder_masking_rechecks_the_source_format_before_starting_a_job(
        editor, qtbot, tmp_path, monkeypatch):
    source = tmp_path / 'field.png'
    Image.fromarray(np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)).save(source)
    original = source.read_bytes()
    assert editor.open_paths([source])
    settle(qtbot, editor)
    displayed = editor._canvas.image.copy()
    monkeypatch.setattr(editor, '_start_folded_run', lambda screen: pytest.fail('incompatible folder started a run'))
    assert editor.mask_whole_folder() is False
    assert 'requires lowercase .tif' in editor._status_label.text()
    np.testing.assert_array_equal(editor._canvas.image, displayed)
    assert source.read_bytes() == original and not list(tmp_path.glob('masks*'))


def test_public_multiseries_tiff_read_and_explicit_plane_keep_distinct_pixels(tmp_path):
    source = tmp_path / 'two-series.tif'
    first = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    second = np.arange(25 * 33, dtype=np.uint16).reshape(25, 33) + 1000
    with tifffile.TiffWriter(source) as writer:
        writer.write(first, metadata={'axes': 'YX'})
        writer.write(second, metadata={'axes': 'YX'})
    original = source.read_bytes()
    np.testing.assert_array_equal(engine.read_image(source), first)
    header = engine._container_image_header(source)
    view = engine._set_container_image_view(header, series=1)
    np.testing.assert_array_equal(engine.read_image(source), second)
    assert view['series'] == 1 and view['shape'] == second.shape
    assert source.read_bytes() == original


def test_slow_lineage_rewrite_keeps_all_saved_labels_and_source_pixels(
        editor, qtbot, tmp_path, monkeypatch):
    pixels = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    labels = np.zeros_like(pixels, dtype=np.uint32)
    labels[3:9, 5:11] = 70001
    for index in range(2):
        source = tmp_path / f'field{index}.tif'
        tifffile.imwrite(source, pixels + index)
        engine.save_mask(str(tmp_path), source.name, labels, preserve_ids=True)
    original = {path: path.read_bytes() for path in tmp_path.glob('*.tif')}
    assert editor.open_paths([tmp_path])
    settle(qtbot, editor)
    editor._toggle_label_time(True)
    assert editor._canvas.assign_label_at((6, 4), 70001)
    graph = editor._lineage
    identity = graph['frames'][str(tmp_path / 'field0.tif')]['70001']
    assert editor._save_mask()
    assert mm._wait_for_mask_writes(timeout=10)
    entered, release = Event(), Event()
    save = engine.save_mask

    def held_save(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return save(*args, **kwargs)

    monkeypatch.setattr(engine, 'save_mask', held_save)
    monkeypatch.setattr(engine, '_MASK_SAVE_WAIT_SECONDS', 0.01)
    try:
        editor._rewrite_sequence_lineage()
        qtbot.waitUntil(entered.is_set, timeout=3000)
        assert editor._mask_write_requests
        np.testing.assert_array_equal(editor._canvas.mask, labels)
        editor._label_name.setText('still responsive')
        assert not editor._mask_write_requests[0]['done'].is_set()
    finally:
        release.set()
    assert mm._wait_for_mask_writes(timeout=10)
    qtbot.waitUntil(lambda: not editor._mask_write_requests, timeout=10000)
    for index in range(2):
        saved = Path(engine.mask_save_path(str(tmp_path), f'field{index}.tif'))
        np.testing.assert_array_equal(tifffile.imread(saved), labels)
        assert tifffile.imread(saved).dtype == np.uint32
        metadata = engine.load_mask_metadata(str(tmp_path), f'field{index}.tif')
        assert metadata['lineage']['cells'][identity]['label'] == 70001
    assert all(path.read_bytes() == data for path, data in original.items())


def test_cli_queue_cancel_before_the_first_series_never_invents_a_displayed_field(
        editor, qtbot, tmp_path, monkeypatch):
    from spacr.curation_queue import build_queue

    source = tmp_path / 'first.tif'
    with tifffile.TiffWriter(source) as writer:
        writer.write(np.zeros((24, 32), np.uint16), metadata={'axes': 'YX'})
        writer.write(np.ones((25, 33), np.uint16), metadata={'axes': 'YX'})
    engine.save_mask(str(tmp_path), source.name, np.zeros((24, 32), np.uint16))
    original = {path: path.read_bytes() for path in tmp_path.rglob('*.tif')}
    queue = build_queue(tmp_path, include_reviewed=True, announce=lambda text: None)
    decisions = []

    def cancel_series(parent, title, prompt, choices, current, editable):
        assert title == 'Image series' and len(choices) == 2
        decisions.append(title)
        return '', False

    monkeypatch.setattr(mm.QInputDialog, 'getItem', cancel_series)
    monkeypatch.setattr(tifffile.TiffPageSeries, 'asarray',
                        lambda *args, **kwargs: pytest.fail('cancelled first review decoded pixels'))
    assert editor.open_queue(queue)
    settle(qtbot, editor)
    assert decisions == ['Image series']
    assert editor._canvas.image is None and editor._canvas.mask is None
    assert editor._reader_displayed_field is None
    assert engine._current_container_image_view(source) is None
    assert all(path.read_bytes() == data for path, data in original.items())


def test_opening_another_image_during_keep_handoff_drops_the_old_verdict_message(
        editor, qtbot, tmp_path, monkeypatch):
    paths = [tmp_path / f'{name}.png' for name in ('a', 'b', 'c')]
    pixels = [np.arange(24 * 32, dtype=np.uint16).reshape(24, 32) + offset
              for offset in (0, 1000, 2000)]
    for path, image in zip(paths, pixels):
        Image.fromarray(image).save(path)
    original = {path: path.read_bytes() for path in paths}
    assert editor._open_queue([str(path) for path in paths[:2]])
    settle(qtbot, editor)
    entered, release = Event(), Event()
    read = engine.load_image_and_mask

    def held_next(folder, filename, **kwargs):
        if filename == paths[1].name:
            entered.set()
            assert release.wait(10)
        return read(folder, filename, **kwargs)

    monkeypatch.setattr(engine, 'load_image_and_mask', held_next)
    monkeypatch.setattr(editor, '_should_background_load', lambda path: True)
    editor._btn_hide_judged.setChecked(False)
    editor._btn_save_on_navigation.setChecked(False)
    try:
        editor._btn_keep.click()
        qtbot.waitUntil(entered.is_set, timeout=3000)
        assert editor._pending_curation_status[0] == str(paths[1])
        assert editor.open_paths([paths[2]])
        assert editor._pending_curation_status is None
    finally:
        release.set()
    settle(qtbot, editor)
    assert editor._image_files == [paths[2].name]
    np.testing.assert_array_equal(editor._canvas.image, pixels[2])
    assert 'kept' not in editor._status_label.text().lower()
    verdict = engine.read_curation(str(tmp_path))[str(paths[0])]
    assert verdict['keep'] == 'true'
    assert all(path.read_bytes() == data for path, data in original.items())


def test_repeating_the_same_real_otsu_fuse_preserves_pixels_and_edit_history(
        editor, qtbot, tmp_path):
    image = np.zeros((24, 32), np.uint16)
    image[4:12, 7:15] = 50000
    labels = (image > 0).astype(np.uint16)
    source = tmp_path / 'object.tif'
    tifffile.imwrite(source, image)
    saved = Path(engine.save_mask(str(tmp_path), source.name, np.zeros_like(labels), preserve_ids=True))
    original = source.read_bytes(), saved.read_bytes()
    assert editor.open_paths([source])
    settle(qtbot, editor)
    request = mm._MagnifierRequest(
        key=('same-object',), crop=image, box=(0, 0, 32, 24), shape=image.shape,
        mode='otsu', sensitivity=0, bright=True, min_area=0, model_name='',
        diameter=0, colour=(255, 0, 0))
    detected, mode, note = mm._segment_region(request)
    assert set(np.unique(detected)) == {0, 1} and detected[8, 11] == 1
    result = mm._MagnifierResult(request, detected, mode, note, None, 1)
    index = editor._mag_overlap.findData('merge')
    assert index >= 0
    editor._mag_overlap.setCurrentIndex(index)
    editor._min_area.setValue(0)
    assert editor._commit_magnifier_result(result) == [1]
    np.testing.assert_array_equal(editor._canvas.mask, detected)
    history = len(editor._history._undo), len(editor._history._redo), len(editor._log.edits)
    assert editor._commit_magnifier_result(result) == []
    assert (len(editor._history._undo), len(editor._history._redo), len(editor._log.edits)) == history
    np.testing.assert_array_equal(editor._canvas.mask, detected)
    assert (source.read_bytes(), saved.read_bytes()) == original


def test_reviewed_grayscale_png_reads_the_original_2d_plane_without_channel_inference(tmp_path):
    source = tmp_path / 'grayscale.png'
    pixels = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
    Image.fromarray(pixels).save(source)
    original = source.read_bytes()
    header = engine._container_image_header(source)
    assert header['series'][0]['reader'] == 'PIL' and header['series'][0]['n_c'] == 1
    view = engine._set_container_image_view(header)
    image, mask = engine.load_image_and_mask(str(tmp_path), source.name, view['masks_dir'])
    np.testing.assert_array_equal(engine.read_image(source), pixels)
    assert image.shape == pixels.shape and not mask.any()
    assert source.read_bytes() == original
