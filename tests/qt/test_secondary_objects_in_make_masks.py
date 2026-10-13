"""Secondary detections preserve parent identity through UI requests and saves."""
from __future__ import annotations

import hashlib
import json

import imageio.v3 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPointF

from spacr.qt import cpu_modes
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    images, primaries = tmp_path / 'images', tmp_path / 'primary'
    images.mkdir()
    primaries.mkdir()
    field = np.zeros((96, 96), np.uint16)
    field[16:80, 16:80] = 200
    primary = np.zeros_like(field)
    primary[42:54, 42:54] = 900
    field[primary > 0] = 0
    for name in ('a.tif', 'b.tif'):
        imageio.imwrite(images / name, field)
        imageio.imwrite(primaries / name, primary)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    made.warnings = []
    monkeypatch.setattr(made, '_warn', lambda *args: made.warnings.append(args))
    assert made._open_folder(str(images))
    made._min_area.setValue(1)
    made._detect_normalized.setChecked(False)
    made._primary_selector.path.setText(str(primaries))
    made._primary_selector._source_changed()
    qtbot.waitUntil(lambda: made._primary_selector.snapshot is not None)
    made._mag_mode.setCurrentIndex(made._mag_mode.findData(cpu_modes.SECONDARY))
    made._secondary_widgets['propagate_sigma'].setValue(0)
    made._secondary_widgets['propagate_stop'].setCurrentIndex(
        made._secondary_widgets['propagate_stop'].findData('absolute'))
    made._secondary_widgets['propagate_stop_value'].setValue(100)
    made._combine_mode.setCurrentIndex(made._combine_mode.findData('replace'))
    made._magnifier.size = 96
    made._magnifier._cursor = (48, 48)
    made._magnifier._anchor = QPointF(48, 48)
    made._magnifier.enabled = True
    made._magnifier.exclude_border = False
    made.requests = []
    monkeypatch.setattr(made._magnifier._worker, 'submit', lambda req, **kw: made.requests.append(req))
    monkeypatch.setattr(made._magnifier._image_worker, 'submit', made.requests.append)
    yield made
    made.close()


def test_whole_field_save_reload_queue_and_readout_keep_primary_identity(screen, qtbot):
    primary_path = screen._primary_selector.snapshot.path
    before = hashlib.sha256(open(primary_path, 'rb').read()).hexdigest()
    screen._on_detect_otsu()
    assert not screen.warnings
    assert set(np.unique(screen._canvas.mask)) == {0, 900}
    assert screen._canvas.mask[20, 20] == screen._canvas.mask[48, 48] == 900
    assert 'Matched: 1 (900)' in screen._secondary_relations.text()
    assert screen._canvas._object_lookup().labels[48, 48] == 900
    screen._on_save()
    saved = screen._curation_paths()[1]
    assert set(np.unique(imageio.imread(saved))) == {0, 900}
    detail = json.loads(open(saved + '.curation.json').read())['edits'][-1]['detail']
    assert detail['primary_source']['sha256'] == before
    assert detail['primary_secondary']['matched_ids'] == [900]
    screen._on_next()
    qtbot.waitUntil(lambda: screen._primary_selector.snapshot is not None)
    assert screen._primary_selector.snapshot.image_path.endswith('b.tif')
    assert not screen._canvas.preserve_ids
    screen._on_prev()
    qtbot.waitUntil(lambda: screen._primary_selector.snapshot is not None)
    assert screen._canvas.preserve_ids
    assert screen._canvas.mask[48, 48] == 900
    assert hashlib.sha256(open(primary_path, 'rb').read()).hexdigest() == before


@pytest.mark.parametrize('growth', ['intensity', 'distance'])
@pytest.mark.parametrize('scope', ['region', 'image'])
def test_both_magnifier_scopes_preserve_ids_and_accept_clicks(screen, scope, growth):
    box = screen._secondary_widgets['secondary_growth']
    box.setCurrentIndex(box.findData(growth))
    magnifier = screen._magnifier
    if scope == 'image':
        magnifier._start_image(magnifier._image_key_now())
        request = screen.requests[-1]
    else:
        request = magnifier.build_request()
    result = magnifier._run(request)
    assert result.mode == cpu_modes.SECONDARY
    assert request.cpu_params.secondary_growth == growth
    assert set(np.unique(result.labels)) == {0, 900}
    magnifier.scope = scope
    if scope == 'image':
        magnifier._image_result = result
    else:
        magnifier._shown = result
    assert magnifier.press()
    assert magnifier._stroke is None
    assert magnifier.release()
    assert screen._canvas.mask[48, 48] == 900
    assert screen._canvas.preserve_ids
    assert screen._log.edits[-1].detail['primary_source']['primary_class'] == 'nucleus'
    assert screen._log.edits[-1].detail['method_parameters']['secondary_growth'] == growth


def test_changed_primary_rejects_old_result_without_falling_back_to_otsu(screen, qtbot):
    request = screen._magnifier.build_request()
    result = screen._magnifier._run(request)
    screen._primary_selector.primary_class.setEditText('another primary')
    assert screen._magnifier.build_request() is None
    assert screen._commit_magnifier_result(result) == []
    qtbot.waitUntil(lambda: screen._primary_selector.snapshot is not None)
    assert screen._commit_magnifier_result(result) == []
    assert not screen._canvas.mask.any()
    with pytest.raises(ValueError, match='primary'):
        mm._segment_region(request._replace(primary_labels=None))


@pytest.mark.parametrize('rule', ['clip', 'skip', 'replace'])
def test_roi_acceptance_extends_same_id_and_preserves_disconnected_parts(screen, rule):
    request = screen._magnifier.build_request()
    result = screen._magnifier._run(request)
    assert screen._commit_magnifier_result(result._replace(labels=request.primary_labels)) == [900]
    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData(rule))
    assert screen._commit_magnifier_result(result) == [900]
    assert screen._canvas.mask[20, 20] == 900
    assert set(np.unique(screen._canvas.mask)) == {0, 900}


def test_missing_and_orphan_objects_are_reported_after_filter_and_undo(screen):
    screen._on_detect_otsu()
    screen._canvas.mask[0:3, 0:3] = 7
    screen._refresh_secondary_report()
    assert 'No primary: 1 (7)' in screen._secondary_relations.text()
    screen._filter_list.set_filter("area", 5000)
    screen.apply_object_filter()
    assert not screen._canvas.mask.any()
    assert 'Missing secondary: 1 (900)' in screen._secondary_relations.text()
    screen._on_undo()
    assert screen._canvas.mask[48, 48] == 900
    assert 'Matched: 1 (900)' in screen._secondary_relations.text()


def test_independent_growth_settings_and_applied_enhancement_never_rekey_labels(screen):
    screen._secondary_widgets['propagate_sigma'].setValue(1.5)
    screen._enh_morphology.setCurrentIndex(screen._enh_morphology.findData('close'))
    screen._enh_split.setChecked(True)
    screen._enh_gamma.setValue(.8)
    screen._btn_apply.setChecked(True)
    request = screen._magnifier.build_request()
    assert request.chain.gamma == .8
    assert request.chain.morphology == 'none' and not request.chain.split
    assert not screen._enh_split.isEnabled()
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(cpu_modes.PROPAGATE))
    assert screen._cpu_params().propagate_sigma == 2
    assert screen._cpu_params().propagate_stop == 'seed_fraction'
    assert screen._enh_split.isEnabled()
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(cpu_modes.SECONDARY))
    assert screen._cpu_params().propagate_sigma == 1.5


def test_all_filtered_secondary_objects_clear_replace_and_report_missing(screen):
    screen._on_detect_otsu()
    screen._min_area.setValue(10000)
    screen._on_detect_otsu()
    assert not screen._canvas.mask.any()
    assert 'Missing secondary: 1 (900)' in screen._secondary_relations.text()


def test_primary_alias_cannot_be_overwritten_after_output_path_changes(screen, monkeypatch):
    screen._on_detect_otsu()
    source = screen._primary_selector.snapshot
    before = open(source.path, 'rb').read()
    monkeypatch.setattr(engine, 'mask_save_path', lambda *a, **k: source.path)
    screen._on_save()
    assert screen.warnings[-1][0] == 'Save failed'
    assert open(source.path, 'rb').read() == before


def test_hole_fill_and_small_object_cleanup_keep_sparse_and_disconnected_ids(screen):
    screen._on_detect_otsu()
    screen._canvas.mask[45:50, 45:50] = 0
    screen._on_fill_holes()
    assert screen._canvas.mask[48, 48] == 900
    screen._canvas.mask[0:2, 0:2] = 7
    screen._min_area.setValue(10)
    screen._on_remove_small()
    assert set(np.unique(screen._canvas.mask)) == {0, 900}
    screen._on_relabel()
    assert set(np.unique(screen._canvas.mask)) == {0, 900}


def test_unrelated_existing_mask_cannot_silently_share_a_primary_id(screen):
    screen._canvas.mask = screen._canvas.mask.astype(np.uint16)
    screen._canvas.mask[5:10, 5:10] = 900
    before = screen._canvas.mask.copy()
    result = screen._magnifier._run(screen._magnifier.build_request())
    assert screen._commit_magnifier_result(result) == []
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert 'not paired' in screen._status_label.text()
    screen._combine_mode.setCurrentIndex(screen._combine_mode.findData('merge'))
    screen._on_detect_otsu()
    assert 'not paired' in screen.warnings[-1][1]
    np.testing.assert_array_equal(screen._canvas.mask, before)
    screen._combine_mode.setCurrentIndex(screen._combine_mode.findData('replace'))
    screen._on_detect_otsu()
    assert screen._canvas.mask[7, 7] == 0
    assert screen._canvas.mask[48, 48] == 900


def test_new_primary_snapshot_cannot_merge_into_an_old_pairing(screen, qtbot):
    screen._on_detect_otsu()
    before = screen._canvas.mask.copy()
    screen._primary_selector.primary_class.setEditText('different primary')
    qtbot.waitUntil(lambda: screen._primary_selector.snapshot is not None)
    result = screen._magnifier._run(screen._magnifier.build_request())
    assert screen._commit_magnifier_result(result) == []
    np.testing.assert_array_equal(screen._canvas.mask, before)
