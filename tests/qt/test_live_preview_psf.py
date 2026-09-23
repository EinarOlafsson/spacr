"""PSF preview intensities, cancellation, result identity and thread ownership."""
from copy import deepcopy
from threading import Event

import numpy as np
import pytest

from spacr.cancellation import PipelineCancelled
from spacr.psf_pipeline import prepare_psf
from spacr.qt.widgets import live_preview as lp


def settings(operation='convolve'):
    return dict(psf_operation=operation, psf_source='gaussian',
                psf_image_sampling_um=[1., 1.], psf_fwhm_um=[2., 2.],
                psf_iterations=3)


def source():
    image = np.zeros((32, 32, 3), np.uint16)
    image[10:20, 10:20, 0] = 1000
    image[15:25, 15:25, 1] = 2000
    image[..., 2] = 65535
    return image


class Model:
    def __init__(self):
        self.inputs = []

    def eval(self, image, **kwargs):
        self.inputs.append(image.copy())
        mask = np.zeros(image.shape, np.int32)
        mask[10:20, 10:20] = 1
        image[:] = 0
        return mask, []


@pytest.mark.parametrize('operation', ['none', 'convolve', 'deconvolve'])
def test_exact_processed_planes_before_background_and_shared_channel_isolation(
        monkeypatch, operation):
    raw = source()
    original = raw.copy()
    config = settings(operation)
    config.update(remove_background_cell=True, cell_background=200.)
    model = Model()
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda name: model)
    request = lp.PreviewRequest(raw, object_types=('cell', 'nucleus', 'pathogen'),
                               channels={'cell': 0, 'nucleus': 0, 'pathogen': 1},
                               preprocess_settings=config)
    expected = raw[..., :2].copy()
    if operation != 'none':
        expected = prepare_psf(config).apply(expected)
    lp._segment_multi(request)
    cell = expected[..., 0].copy()
    cell[cell < 200] = 0
    for actual, target in zip(model.inputs,
                              (cell, expected[..., 0], expected[..., 1])):
        np.testing.assert_allclose(actual, target, atol=1e-5)
    np.testing.assert_array_equal(raw, original)
    assert request.provenance['channels'] == {'cell': 0, 'nucleus': 0, 'pathogen': 1}
    assert request.provenance['processing']['operation'] == operation


def test_bad_calibration_fails_before_loading_model(monkeypatch):
    monkeypatch.setattr(lp, 'preview_cellpose_model',
                        lambda name: pytest.fail('model should not load'))
    config = settings()
    config['psf_image_sampling_um'] = None
    with pytest.raises(ValueError, match='psf_image_sampling_um'):
        lp._segment_multi(lp.PreviewRequest(source(), preprocess_settings=config))


def test_classical_uses_processed_pixels_without_constructing_cellpose(monkeypatch):
    monkeypatch.setattr(lp, 'preview_cellpose_model',
                        lambda name: pytest.fail('classical needs no model'))
    config = settings()
    config['organelle_method'] = 'otsu'
    captured = []
    def classical(image, role, config):
        captured.append(image.copy())
        return (image > 100).astype(np.int32)
    monkeypatch.setattr(lp, '_classical_organelle_mask', classical)
    request = lp.PreviewRequest(source(), object_types=('organelle',),
                               preprocess_settings=config)
    lp._segment_multi(request)
    np.testing.assert_allclose(captured[0], prepare_psf(config).apply(
        request.image[..., :1])[..., 0])
    assert request.provenance['methods'] == {'organelle': 'otsu'}


def test_measured_kernel_is_captured_once_for_multiple_channels(monkeypatch, tmp_path):
    path = tmp_path / 'kernel.npy'
    np.save(path, np.ones((3, 3)))
    config = settings()
    config.update(psf_source='measured', psf_path=str(path),
                  psf_kernel_sampling_um=[1, 1])
    expected = prepare_psf(config).apply(source()[..., :2])
    model = Model()
    def factory(name):
        path.unlink()
        return model
    monkeypatch.setattr(lp, 'preview_cellpose_model', factory)
    request = lp.PreviewRequest(source(), object_types=('cell', 'nucleus'),
                               channels={'cell': 0, 'nucleus': 1},
                               preprocess_settings=config)
    lp._segment_multi(request)
    for channel in (0, 1):
        np.testing.assert_allclose(model.inputs[channel], expected[..., channel])
    assert request.provenance['processing']['kernel']


def test_real_rl_stops_between_convolutions_before_model(monkeypatch):
    from spacr.point_spread import _ReflectOperator
    request = lp.PreviewRequest(source(), preprocess_settings=settings('deconvolve'))
    original = _ReflectOperator.forward
    calls = []
    def forward(operator, image):
        result = original(operator, image)
        calls.append(1)
        request.cancel.set()
        return result
    monkeypatch.setattr(_ReflectOperator, 'forward', forward)
    monkeypatch.setattr(lp, 'preview_cellpose_model',
                        lambda name: pytest.fail('cancelled before inference'))
    with pytest.raises(PipelineCancelled):
        lp._segment_multi(request)
    assert len(calls) == 1


@pytest.fixture
def panel(qtbot):
    widget = lp.LivePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    widget._image = source()
    yield widget
    widget.shutdown()


def test_request_snapshots_nested_settings(panel):
    panel._settings.update(settings())
    request = panel._build_request()
    panel._settings['psf_fwhm_um'][0] = 12
    assert request.preprocess_settings['psf_fwhm_um'] == [2, 2]
    request.preprocess_settings['psf_fwhm_um'][0] = 8
    assert panel._settings['psf_fwhm_um'] == [12, 2]


def test_cancel_between_metadata_and_masks_keeps_previous_identity(panel):
    old = {'model': 'old', 'processing': {'operation': 'none'}}
    panel._on_processing_provenance(old, panel._run_token)
    masks = {'cell': np.ones((32, 32), np.int32)}
    panel._on_worker_done(masks, '', panel._run_token)
    token = panel._run_token
    panel._on_processing_provenance({'model': 'abandoned'}, token)
    panel.cancel_preview()
    panel._on_worker_done(masks, '', token)
    assert panel._processing_provenance == old
    assert len(panel._history) == 1


def test_classical_history_names_algorithm_without_claiming_cellpose(panel):
    record = {'model': 'cpsam', 'methods': {'organelle': 'otsu'}}
    panel._on_processing_provenance(record, -1)
    panel._on_worker_done({'organelle': np.ones((32, 32), np.int32)}, '', -1)
    assert 'otsu' in panel._status.text()
    assert 'cpsam' not in panel._status.text()
    assert panel._history[-1]['model'] == 'otsu'


def test_success_history_and_failed_rerun_keep_actual_provenance(panel, qtbot, monkeypatch):
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda name: Model())
    panel._settings.update(settings())
    panel.run_preview()
    qtbot.waitUntil(lambda: bool(panel._history), timeout=5000)
    qtbot.waitUntil(lambda: not panel.preview_running())
    record = deepcopy(panel._processing_provenance)
    assert record['processing']['operation'] == 'convolve'
    assert panel._history[-1]['processing_provenance'] == record
    assert 'PSF:' in panel._status.text()
    assert 'batch normalization' in panel._status.toolTip()
    panel._settings['psf_image_sampling_um'] = None
    panel.run_preview()
    qtbot.waitUntil(lambda: 'psf_image_sampling_um' in panel._status.text())
    qtbot.waitUntil(lambda: not panel.preview_running())
    assert panel._processing_provenance == record
    assert len(panel._history) == 1
    panel._on_processing_provenance({'model': 'wrong'}, panel._run_token - 1)
    assert panel._processing_provenance == record
    panel._on_compare_scrub(0)
    assert 'convolve' in panel._compare_label.toolTip()


def test_intensity_filters_keep_original_pixels(panel, monkeypatch):
    panel._settings.update(settings())
    mask = np.zeros((32, 32), np.int32)
    mask[10:20, 10:20] = 1
    seen = []
    def filter_mask(mask, config, role, intensity_img=None):
        seen.append(intensity_img.copy())
        return mask
    monkeypatch.setattr(lp, '_apply_size_filter', filter_mask)
    panel._on_worker_done({'cell': mask}, '')
    np.testing.assert_array_equal(seen[0], lp._select_channel(
        panel._image, panel._obj_channel('cell')))


@pytest.mark.parametrize('close', [False, True])
def test_cancel_or_close_native_work_discards_result_without_destroying_thread(
        panel, qtbot, monkeypatch, close):
    entered, release = Event(), Event()
    def segment(request):
        entered.set()
        assert release.wait(5)
        return {'cell': np.ones(request.image.shape[:2], np.int32)}
    monkeypatch.setattr(lp, '_segment_multi', segment)
    panel.run_preview()
    qtbot.waitUntil(entered.is_set)
    worker = panel._worker
    try:
        if close:
            panel.shutdown()
            assert worker.parent() is None
            assert panel._worker is None
        else:
            panel.cancel_preview()
            panel.run_preview()
            assert panel._worker is worker
        assert worker._request.cancel.is_set()
        assert worker.isRunning()
    finally:
        release.set()
        qtbot.waitUntil(lambda: not worker.isRunning())
        qtbot.wait(20)
        from spacr.qt.bridge import prune_parked_threads
        prune_parked_threads()
    assert not panel._history
    assert not panel._processing_provenance
