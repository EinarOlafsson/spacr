"""Accepted masks retain the detector requests that produced their pixels."""
from __future__ import annotations

import json

import imageio.v3 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPointF

from spacr.qt import cpu_modes, detect_chain, mask_engine as engine
from spacr.qt._magnifier_drag import _DragStroke
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    field = np.random.default_rng(43).integers(100, 150, (96, 96), dtype=np.uint16)
    field[20:76, 20:76] += 200
    field[36:60, 36:60] += 200
    imageio.imwrite(tmp_path / 'field.tif', field)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(tmp_path))
    made._otsu_smoothing.setValue(0.5)
    made._otsu_fill_holes.setChecked(False)
    made._otsu_split.setChecked(False)
    made._min_area.setValue(8)
    made._otsu_window.setValue(15)
    made._otsu_local_k.setValue(0.35)
    made._detect_normalized.setChecked(True)
    made._norm_lo.setValue(2)
    made._norm_hi.setValue(98)
    made._enh_gamma.setValue(0.7)
    made._magnifier.size = 96
    made._magnifier._cursor = (48, 48)
    made._magnifier._anchor = QPointF(48, 48)
    made._magnifier.enabled = True
    made._magnifier.exclude_border = False
    made.requests = []
    monkeypatch.setattr(made._magnifier._worker, 'submit', lambda request, **kw: made.requests.append(request))
    monkeypatch.setattr(made._magnifier._image_worker, 'submit', made.requests.append)
    yield made
    made._magnifier.close()
    made.close_folded()


def choose(screen, mode):
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(mode))


def request_now(screen, scope):
    if scope == 'region':
        return screen._magnifier.build_request()
    screen._magnifier._start_image(screen._magnifier._image_key_now())
    return screen.requests[-1]


def replay(source, detail):
    """Reconstruct tested CPU requests using JSON metadata and the source field."""
    field = engine.invert_normalized(source) if detail['invert'] else source
    if detail['detect_on_normalized']:
        field = engine.normalize_for_detection(field, *detail['normalization_percentiles'])
    x0, y0, x1, y1 = detail['detection_box']
    enhancement = detail['enhancement']
    chain = detect_chain.Chain(gamma=enhancement.get('gamma', 1.0)) if isinstance(enhancement, dict) else detect_chain.NO_CHAIN
    request = mm._MagnifierRequest(
        key=('replay',), crop=field[y0:y1, x0:x1].copy(), box=(x0,y0,x1,y1), shape=source.shape,
        mode=detail['mode'], sensitivity=detail['sensitivity'], bright=detail['bright'],
        min_area=detail['min_area'], model_name=detail['model'], diameter=detail['diameter'], colour=(1,2,3),
        exclude_border=detail['exclude_border'], scope=detail['scope'], chain=chain,
        otsu_correction=detail['otsu_correction'], otsu_smoothing=detail.get('otsu_smoothing', 0),
        otsu_fill_holes=detail['otsu_fill_holes'], otsu_split=detail.get('otsu_split', False),
        otsu_classes=detail.get('otsu_classes', 2), otsu_foreground_class=detail.get('otsu_foreground_class'),
        otsu_window=detail.get('otsu_window', 51), cpu_params=cpu_modes.CpuParams(**detail['method_parameters']))
    labels, used, note = mm._segment_region(request)
    assert used == detail['mode'] and not note
    return labels


@pytest.mark.parametrize('mode', ['niblack', 'multiotsu', 'propagate'])
@pytest.mark.parametrize('scope', ['region', 'image'])
def test_click_history_uses_completed_request_and_replays_after_panel_changes(screen, mode, scope):
    choose(screen, mode)
    request = request_now(screen, scope)
    assert request.detection_percentiles == (2, 98)
    completed = screen._magnifier._run(request)
    assert completed.count > 0
    clicked = mm._single_object(completed, 1) if scope == 'image' else completed
    choose(screen, 'mean')
    screen._detect_normalized.setChecked(False)
    screen._otsu_local_k.setValue(-0.5)
    screen._otsu_window.setValue(33)
    screen._enh_gamma.setValue(1.5)
    screen._otsu_smoothing.setValue(2)
    screen._min_area.setValue(3)
    assert screen._commit_magnifier_result(clicked)
    detail = json.loads(json.dumps(screen._log.edits[-1].detail))
    assert detail['mode'] == mode
    assert detail['normalization_percentiles'] == [2, 98]
    assert detail['enhancement']['gamma'] == 0.7
    assert detail['min_area'] == 8 and detail['paste_min_area'] == 3
    assert detail['scope'] == scope
    if mode == 'niblack':
        assert detail['otsu_window'] == 15
        assert detail['method_parameters']['local_k'] == 0.35
    if scope == 'image':
        assert detail['detection_box'] == [0, 0, 96, 96]
        assert detail['box'] == list(clicked.request.box)
        assert detail['source_labels'] == [1]
    np.testing.assert_array_equal(replay(screen._canvas.image, detail), completed.labels)
    screen._on_save()
    saved = screen._curation_paths()[1]
    persisted = json.loads(open(saved + '.curation.json').read())
    assert persisted['edits'][-1]['detail'] == detail
    screen._on_undo()
    assert not screen._canvas.mask.any()


def test_drag_records_each_actual_frame_and_path_instead_of_the_current_mode(screen):
    choose(screen, 'niblack')
    magnifier = screen._magnifier
    assert magnifier.press()
    stroke = magnifier._stroke
    first = screen.requests[-1]
    magnifier._stroke_delivered((first, magnifier._run(first), None))
    stroke.extend((55, 48))
    choose(screen, 'multiotsu')
    screen._otsu_classes.setValue(4)
    screen._otsu_foreground.setValue(1)
    magnifier._stroke_frame((55, 48))
    second = screen.requests[-1]
    magnifier._stroke_delivered((second, magnifier._run(second), None))
    choose(screen, 'mean')
    magnifier.save_mode = 'touching'
    stroke.release()
    magnifier._stroke_show(final=True)
    detail = json.loads(json.dumps(screen._log.edits[-1].detail))
    assert detail['mode'] == 'mixed'
    assert detail['save'] == 'zoom'
    assert detail['path'] == [[48,48], [55,48]]
    assert [frame['mode'] for frame in detail['frame_requests']] == ['niblack', 'multiotsu']
    assert detail['frames'] == len(detail['frame_requests']) == 2
    rebuilt = _DragStroke(detail['shape'], detail['path'][0], step=detail['frame_step'],
                          keep_untouched=detail['keep_untouched'])
    for point in detail['path'][1:]:
        rebuilt.extend(point)
    for index, frame in enumerate(detail['frame_requests']):
        rebuilt.expect(index)
        rebuilt.deliver(index, replay(screen._canvas.image, frame), frame['detection_box'])
    found = rebuilt.outcome()
    pasted, _ = engine._paste_region_objects(np.zeros((96,96), np.uint16), found.labels,
        found.origin, overlap=detail['overlap'], min_area=detail['paste_min_area'])
    np.testing.assert_array_equal(screen._canvas.mask, pasted)
    screen._on_undo()
    assert not screen._canvas.mask.any()


def test_fallback_records_the_actual_algorithm_and_reason(screen):
    request = request_now(screen, 'region')._replace(mode='cellpose', otsu_classes=4)
    detail = mm._magnifier_provenance(request, 'otsu', 'weights unavailable')
    assert detail['mode'] == 'otsu' and detail['requested_mode'] == 'cellpose'
    assert detail['fallback_note'] == 'weights unavailable'
    assert detail['otsu_classes'] == 2
    assert detail['otsu_local'] is False


def test_normalization_percentiles_are_in_the_request_key(screen):
    before = request_now(screen, 'region')
    screen._norm_hi.setValue(95)
    after = request_now(screen, 'region')
    assert before.key != after.key
    assert before.detection_percentiles == (2, 98)
    assert after.detection_percentiles == (2, 95)


def test_duplicate_or_unsolicited_drag_deliveries_do_not_add_provenance():
    stroke = _DragStroke((8,8), (2,2), step=2)
    labels = np.ones((8,8), np.int32)
    stroke.expect('first')
    assert stroke.deliver('first', labels, (0,0,8,8), provenance={'mode':'niblack'})
    assert not stroke.deliver('first', labels, (0,0,8,8), provenance={'mode':'wrong'})
    assert stroke.outcome().provenance['frame_requests'] == [{'mode':'niblack'}]
