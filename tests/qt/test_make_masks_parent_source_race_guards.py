"""Native puncta rejects missing or changed source files without replacing masks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr.qt import cpu_modes
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path):
    image = np.full((32, 32), 40, dtype=np.float32)
    parents = np.full(image.shape, 7, dtype=np.uint16)
    folder = tmp_path / 'images'
    folder.mkdir()
    image_file = folder / 'a.tif'
    parent_file = tmp_path / 'parents.tif'
    tifffile.imwrite(image_file, image)
    tifffile.imwrite(parent_file, parents)
    screen = mm.MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._open_folder(str(folder))
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(cpu_modes.PUNCTA))
    yield screen, image, parents, image_file, parent_file
    screen._magnifier.close()
    screen.close_folded()


def _load_parent(opened, qtbot):
    screen, image, parents, image_file, parent_file = opened
    screen._primary_selector.path.setText(str(parent_file))
    screen._primary_selector._source_changed()
    qtbot.waitUntil(lambda: screen._primary_selector.snapshot is not None,
                    timeout=10000)
    screen._detector_image()
    return screen._primary_selector.snapshot


def test_puncta_requires_an_explicit_valid_parent(opened):
    screen, image, *_rest = opened
    prior = screen._canvas.mask.copy()
    with pytest.raises(ValueError, match='valid parent mask'):
        screen._cpu_detect(image, cpu_modes.PUNCTA, screen._otsu_settings())
    np.testing.assert_array_equal(screen._canvas.mask, prior)


@pytest.mark.parametrize('changed_source', ['image', 'parent'])
def test_native_sources_changed_during_detection_cannot_publish_a_result(
        opened, qtbot, monkeypatch, changed_source):
    screen, image, parents, image_file, parent_file = opened
    _load_parent(opened, qtbot)
    prior = screen._canvas.mask.copy()

    def detect_then_change(_image, _parents, _params, *, measurements):
        assert measurements
        if changed_source == 'image':
            tifffile.imwrite(image_file, image + 1)
        else:
            tifffile.imwrite(parent_file, parents * 0)
        return np.zeros(image.shape, dtype=np.int32), pd.DataFrame()

    monkeypatch.setattr(cpu_modes, 'puncta', detect_then_change)
    with pytest.raises(ValueError, match='changed'):
        screen._cpu_detect(image, cpu_modes.PUNCTA, screen._otsu_settings())
    np.testing.assert_array_equal(screen._canvas.mask, prior)
    assert getattr(screen, '_puncta_measurement_snapshot', None) is None
    assert not list(image_file.parent.glob('**/*.puncta.csv'))


def test_parent_association_and_save_guard_work_without_an_edit_log(
        opened, qtbot, monkeypatch):
    screen, *_rest = opened
    source = _load_parent(opened, qtbot)
    monkeypatch.setattr(screen, '_log', None)
    screen._retain_secondary_ids(source.provenance())
    assert screen._canvas.preserve_ids
    assert screen._paired_source == source.provenance()
    screen._validate_secondary_save()
    monkeypatch.setattr(mm.engine, 'mask_save_path', lambda *_args, **_kw: source.path)
    with pytest.raises(ValueError, match='different files'):
        screen._validate_secondary_save()


def test_parent_invalidation_during_magnifier_teardown_updates_the_report(
        opened, qtbot, monkeypatch):
    screen, *_rest = opened
    _load_parent(opened, qtbot)
    screen._primary_selector.snapshot = None
    with monkeypatch.context() as during_close:
        during_close.delattr(screen, '_magnifier')
        screen._on_primary_source_changed()
    assert 'Load a primary mask' in screen._secondary_relations.text()


def test_cancelled_puncta_region_discards_a_completed_detector_result(monkeypatch):
    image = np.zeros((32, 32), dtype=np.float32)
    parents = np.full(image.shape, 7, dtype=np.uint16)
    ticket = mm._RunTicket()
    request = mm._MagnifierRequest(
        key=(), crop=image, box=(0, 0, 32, 32), shape=image.shape,
        mode=cpu_modes.PUNCTA, sensitivity=1, bright=True, min_area=1,
        model_name='', diameter=0, colour=(1, 2, 3), scope='image',
        primary_labels=parents, ticket=ticket)

    def finish_after_cancel(_image, _parents, _params):
        ticket.cancel()
        return np.full(image.shape, 7, dtype=np.uint16)

    monkeypatch.setattr(cpu_modes, 'puncta', finish_after_cancel)
    with pytest.raises(mm._RunCancelled):
        mm._segment_region(request)


def test_puncta_provenance_records_parent_and_raw_detector_input(opened, qtbot):
    screen, image, *_rest = opened
    source = _load_parent(opened, qtbot)
    request = mm._MagnifierRequest(
        key=(), crop=image, box=(0, 0, 32, 32), shape=image.shape,
        mode=cpu_modes.PUNCTA, sensitivity=1, bright=True, min_area=1,
        model_name='', diameter=0, colour=(1, 2, 3), scope='image',
        primary_labels=source.labels, primary_provenance=source.provenance(),
        detection_percentiles=(1, 99), invert=True)
    detail = mm._magnifier_provenance(request, cpu_modes.PUNCTA)
    assert detail['primary_source'] == source.provenance()
    assert detail['detect_on_normalized'] is False
    assert detail['normalization_percentiles'] is None
    assert detail['invert'] is False


def test_native_read_rejects_a_file_replaced_during_loading(opened, monkeypatch):
    screen, image, _parents, image_file, _parent_file = opened
    before = screen._canvas.mask.copy()
    screen._puncta_native_cache = None
    original_read = mm.engine.imageio.imread

    def replace_during_read(path):
        read = original_read(path)
        tifffile.imwrite(image_file, image + 1)
        return read

    with monkeypatch.context() as during_read:
        during_read.setattr(mm.engine.imageio, 'imread', replace_during_read)
        with pytest.raises(ValueError, match='changed while being read'):
            screen._puncta_native_image()
    assert screen._puncta_native_cache is None
    np.testing.assert_array_equal(screen._canvas.mask, before)
