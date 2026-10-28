"""Organized-stack Live Preview uses the batch spectral correction."""
from copy import deepcopy
import threading

import numpy as np
import pytest

from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS
from tests.conftest import check_cellpose_eval_call

from spacr.cancellation import PipelineCancelled
from spacr.psf_pipeline import _prepare_unmixing, prepare_psf
from spacr.qt.widgets import live_preview as lp


@pytest.fixture
def plate(tmp_path):
    stack = tmp_path / 'stack'
    stack.mkdir()
    matrix = np.array([[1., .25], [.5, 1.]])
    first = np.zeros((32, 32))
    first[4:12, 4:12] = 1200
    second = np.zeros((32, 32))
    second[19:27, 18:26] = 800
    fields = {}
    for well, dyes in [('A01', (first, first * 0)),
                       ('B01', (second * 0, second)),
                       ('C01', (first, second))]:
        values = np.stack(dyes, axis=-1) @ matrix.T + 100
        path = stack / f'plate1_{well}_1.npy'
        np.save(path, values.astype(np.uint16))
        fields[well] = path
    return fields, np.stack((first, second), axis=-1) + 100


def config(**extra):
    return dict(unmix=True, unmix_controls='0:A01; 1:B01',
                unmix_background_percentile=5., organelle_method='otsu',
                **extra)


def request(plate, settings=None):
    path = plate[0]['C01']
    loaded = lp.load_source_payload(path)
    assert loaded['error'] == ''
    return lp.PreviewRequest(
        loaded['array'], source_path=str(loaded['path']),
        object_types=('organelle', 'organelleb'),
        channels={'organelle': 1, 'organelleb': 0},
        preprocess_settings=settings if settings is not None else config())


@pytest.mark.parametrize('psf', [False, True])
def test_actual_loader_to_segmentation_matches_batch_pixels_and_keeps_source(
        plate, monkeypatch, psf):
    settings = config()
    if psf:
        settings.update(psf_operation='convolve', psf_source='gaussian',
                        psf_image_sampling_um=[1., 1.], psf_fwhm_um=[2., 2.])
    req = request(plate, settings)
    original = req.image.copy()
    before = {p: p.read_bytes() for p in plate[0].values()}
    batch = _prepare_unmixing(settings, plate[0]['C01'].parent)
    expected = batch.apply(original)
    np.testing.assert_allclose(expected, plate[1], atol=1e-4)
    if psf:
        expected = prepare_psf(settings).apply(expected)
    inputs = []
    actual_segment = lp._classical_organelle_mask

    def inspect_pixels(image, role, options):
        inputs.append(image.copy())
        return actual_segment(image, role, options)

    monkeypatch.setattr(lp, '_classical_organelle_mask', inspect_pixels)
    monkeypatch.setattr(lp, 'preview_cellpose_model',
                        lambda name: pytest.fail('classical preview loaded a model'))
    masks, flows = lp._segment_multi(req)
    assert not flows
    assert all(np.any(mask) for mask in masks.values())
    for actual, channel in zip(inputs, [1, 0]):
        np.testing.assert_allclose(actual, expected[..., channel], atol=1e-4)
    np.testing.assert_array_equal(req.image, original)
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert req.provenance['unmixing']['matrix'] == batch.provenance()['matrix']
    assert req.provenance['unmixing']['fields'] == batch.provenance()['fields']
    assert req.provenance['filter_intensity_source'] == 'original loaded preview field'
    assert req.provenance['input_modified'] is False


def test_disabled_unmixing_is_exact_identity_without_control_reads(plate):
    req = request(plate, {'unmix': False})
    req.source_path = 'nonexistent.tif'
    actual, provenance = lp._unmix_preview_field(req)
    assert actual is req.image and provenance is None


@pytest.mark.parametrize('failure', ['missing', 'flat', 'channels', 'raw', 'merged', 'shape'])
def test_invalid_sources_fail_before_segmentation(plate, monkeypatch, failure):
    req = request(plate)
    if failure == 'missing':
        req.preprocess_settings['unmix_controls'] = '0:D01'
    elif failure == 'flat':
        np.save(plate[0]['A01'], np.ones_like(req.image))
    elif failure == 'channels':
        np.save(plate[0]['A01'], np.ones((*req.image.shape[:2], 3)))
    elif failure == 'raw':
        req.source_path = str(plate[0]['C01'].with_suffix('.tif'))
    elif failure == 'merged':
        path = plate[0]['C01'].parent.parent / 'merged'
        path.mkdir()
        target = path / plate[0]['C01'].name
        np.save(target, req.image)
        req.source_path = str(target)
    else:
        req.image = req.image[..., 0]
    monkeypatch.setattr(lp, '_classical_organelle_mask',
                        lambda *a: pytest.fail('invalid source reached segmentation'))
    with pytest.raises(ValueError):
        lp._segment_multi(req)


def test_cancel_during_control_read_stops_before_next_stage(plate, monkeypatch):
    req = request(plate)
    real_load = np.load
    reads = []

    def read(path, **kwargs):
        reads.append(path)
        result = real_load(path, **kwargs)
        req.cancel.set()
        return result

    monkeypatch.setattr(np, 'load', read)
    with pytest.raises(PipelineCancelled):
        lp._segment_multi(req)
    assert len(reads) == 1


def test_request_settings_not_modified_by_unmix_plan(plate):
    req = request(plate)
    original = deepcopy(req.preprocess_settings)
    lp._unmix_preview_field(req)
    assert req.preprocess_settings == original


@pytest.mark.parametrize('array', [np.zeros((2, 3, 4, 5)), np.zeros((0, 4)),
                                   np.array([['not pixels']])])
def test_bad_npy_layout_is_reported_by_actual_loader(tmp_path, array):
    path = tmp_path / 'bad.npy'
    np.save(path, array)
    payload = lp.load_source_payload(path)
    assert payload['array'] is None and payload['error']


def test_stack_directory_discovery_can_load_numpy_fields(plate):
    payload = lp.load_source_payload(plate[0]['C01'].parent)
    assert payload['error'] == ''
    assert payload['path'].suffix == '.npy'
    assert payload['array'].shape == (32, 32, 2)


@pytest.fixture
def calibrated_plate(plate, tmp_path):
    import json

    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({
        'version': 1, 'axes': 'YX', 'pixel_size_um': [1., 1.], 'reference_channel': 0,
        'channels': [{'channel': channel, 'shift_to_reference_px': [0, 0],
                      'psf': [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                      'psf_origin_px': [1, 1], 'psf_pixel_size_um': [1., 1.]}
                     for channel in (0, 1)]}))
    settings = config(unmix_calibration=str(path), unmix_pixel_size_um=[1., 1.],
                      organelle_min_size=0)
    return plate, settings


def test_calibrated_classical_preview_infers_only_valid_nonnegative_crop_and_restores_canvas(
        calibrated_plate, monkeypatch):
    plate, settings = calibrated_plate
    path = plate[0]['C01']
    first = np.zeros((32, 32), dtype=np.float32)
    first[0:8, 4:12] = 1200
    first[16:24, 18:26] = 900
    dyes = np.stack((first, first * 0), axis=-1)
    measured = dyes @ np.array([[1., .25], [.5, 1.]]).T
    camera_noise_readout = np.array([0., 40.])
    measured[12, 12] = camera_noise_readout
    assert np.all(measured >= 0)
    np.save(path, measured.astype(np.float32))
    req = request(plate, settings)
    req.object_types = ('organelle',)
    req.channels = {'organelle': 0}
    originals = {file: file.read_bytes() for file in plate[0].values()}
    raw = req.image.copy()
    plan = _prepare_unmixing(settings, path.parent, allow_calibrated=True)
    expected = plan.apply_calibrated(raw)
    assert expected.image.dtype == np.float32 and np.any(expected.image < 0)
    assert expected.image.shape[0] < raw.shape[0] and expected.image.shape[1] < raw.shape[1]
    observed, inferred = [], []
    actual_classical = lp._classical_organelle_mask

    def segment(image, role, options):
        observed.append(image.copy())
        mask = actual_classical(image, role, options)
        inferred.append(mask.copy())
        return mask

    monkeypatch.setattr(lp, '_classical_organelle_mask', segment)
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda *_: pytest.fail('loaded model'))
    masks, flows = lp._segment_multi(req)
    assert not flows and len(observed) == 1
    np.testing.assert_allclose(observed[0], np.maximum(expected.image[..., 0], 0), atol=1e-4)
    labels = inferred[0]
    boundary = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    boundary = boundary[boundary != 0]
    assert len(boundary) > 0
    kept = np.where(np.isin(labels, boundary), 0, labels)
    assert np.any(kept)
    restored = np.zeros(raw.shape[:2], dtype=labels.dtype)
    restored[expected.valid_slices] = kept
    np.testing.assert_array_equal(masks['organelle'], restored)
    record = req.provenance['unmixing']
    assert record['mode'] == 'calibrated_live_preview'
    assert record['valid_roi_yx'] == expected.record['valid_roi_yx']
    assert record['source_shape'] == list(raw.shape)
    assert record['negative_fraction_per_channel'] == expected.record['negative_fraction_per_channel']
    assert record['excluded_boundary_label_ids']['organelle'] == boundary.tolist()
    assert record['segmentation_view'] == 'nonnegative clipped view; not quantitative correction'
    assert req.provenance['filter_intensity_source'] == 'unavailable in calibrated reference frame'
    np.testing.assert_array_equal(req.image, raw)
    assert all(file.read_bytes() == payload for file, payload in originals.items())


@pytest.mark.parametrize('extra', [dict(psf_operation='convolve', psf_source='gaussian',
                                       psf_image_sampling_um=[1., 1.], psf_fwhm_um=[2., 2.]),
                                   dict(enhance_gamma=2.)])
def test_calibrated_preview_refuses_extra_processing_before_inference(calibrated_plate, monkeypatch, extra):
    plate, settings = calibrated_plate
    req = request(plate, {**settings, **extra})
    monkeypatch.setattr(lp, '_classical_organelle_mask', lambda *_: pytest.fail('segmented'))
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda *_: pytest.fail('loaded model'))
    with pytest.raises(ValueError, match='additional PSF or enhancement'):
        lp._segment_multi(req)


def test_calibrated_preview_refuses_volume_before_inference(calibrated_plate, monkeypatch):
    plate, settings = calibrated_plate
    req = request(plate, settings)
    req.image = np.stack((req.image, req.image))
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda *_: pytest.fail('loaded model'))
    with pytest.raises(ValueError, match='nonempty'):
        lp._segment_multi(req)


@pytest.mark.parametrize('whole_canvas', [False, True])
def test_calibrated_preview_refuses_supplied_cell_masks_without_current_pass_cells(calibrated_plate, monkeypatch, whole_canvas):
    plate, settings = calibrated_plate
    req = request(plate, {**settings, 'organelle_mask_within_cells': True})
    req.object_types = ('organelle',)
    req.channels = {'organelle': 0}
    expected = _prepare_unmixing(settings, plate[0]['C01'].parent,
                                allow_calibrated=True).apply_calibrated(req.image)
    req.cell_mask = np.ones(req.image.shape[:2] if whole_canvas else expected.image.shape[:2], dtype=np.int32)
    original = req.cell_mask.copy()
    monkeypatch.setattr(lp, '_classical_organelle_mask', lambda *_: pytest.fail('segmented'))
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda *_: pytest.fail('loaded model'))
    with pytest.raises(ValueError, match='cells generated in the same preview pass'):
        lp._segment_multi(req)
    np.testing.assert_array_equal(req.cell_mask, original)


def test_calibrated_model_adapter_keeps_crop_for_generated_cells_then_restores_overlays(
        calibrated_plate, monkeypatch):
    plate, settings = calibrated_plate
    req = request(plate, {**settings, 'organelle_mask_within_cells': True})
    req.object_types = ('cell', 'organelle')
    req.channels = {'cell': 0, 'organelle': 0}
    expected = _prepare_unmixing(settings, plate[0]['C01'].parent,
                                allow_calibrated=True).apply_calibrated(req.image)
    shape = expected.image.shape[:2]
    cell = np.zeros(shape, dtype=np.int32)
    cell[2:-2, 2:-2] = 1
    cell[0:2, :6] = 2
    flow = np.arange(np.prod(shape) * 3, dtype=np.float32).reshape(*shape, 3)
    probability = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    observed = []

    class CPUAdapter:
        def eval(self, image, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 invert=False, rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
            check_cellpose_eval_call(image, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D, stitch_threshold=stitch_threshold)
            observed.append(image.copy())
            return cell.copy(), [flow.copy(), np.zeros((2,) + shape), probability.copy()]

    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda *_: CPUAdapter())
    actual_classical = lp._classical_organelle_mask
    organelle_inputs = []

    def segment(image, role, options):
        organelle_inputs.append(image.copy())
        return actual_classical(image, role, options)

    monkeypatch.setattr(lp, '_classical_organelle_mask', segment)
    masks, flows = lp._segment_multi(req)
    assert len(observed) == len(organelle_inputs) == 1
    np.testing.assert_allclose(observed[0], np.maximum(expected.image[..., 0], 0), atol=1e-4)
    assert np.all(organelle_inputs[0][cell != 1] == 0)
    retained = np.where(cell == 2, 0, cell)
    cell_canvas = np.zeros(req.image.shape[:2], dtype=np.int32)
    cell_canvas[expected.valid_slices] = retained
    np.testing.assert_array_equal(masks['cell'], cell_canvas)
    flow_canvas = np.zeros(req.image.shape[:2] + (3,), dtype=np.float32)
    flow_canvas[expected.valid_slices] = flow
    np.testing.assert_array_equal(flows['cell'], flow_canvas)
    probability_canvas = np.zeros(req.image.shape[:2], dtype=np.float32)
    probability_canvas[expected.valid_slices] = probability
    np.testing.assert_array_equal(req.cellprob_maps['cell'], probability_canvas)
    assert req.provenance['unmixing']['excluded_boundary_label_ids']['cell'] == [2]



def test_configured_calibration_cannot_be_silently_disabled_in_preview(calibrated_plate, monkeypatch):
    plate, settings = calibrated_plate
    req = request(plate, {**settings, 'unmix': False})
    monkeypatch.setattr(lp, '_classical_organelle_mask', lambda *_: pytest.fail('segmented'))
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda *_: pytest.fail('loaded model'))
    with pytest.raises(ValueError, match='explicit 2D live preview'):
        lp._segment_multi(req)


def test_real_calibrated_gui_uses_aligned_display_and_refuses_postrun_intensity_edits(
        calibrated_plate, qtbot, monkeypatch):
    import json

    plate, settings = calibrated_plate
    panel = lp.LivePreviewPanel(threaded=True)
    qtbot.addWidget(panel)
    panel.apply_settings({**settings, 'organelle_channel': 0})
    index = panel._object_box.findData('organelle')
    assert index >= 0
    panel._object_box.setCurrentIndex(index)
    assert panel.load_image(plate[0]['C01']), panel._status.text()
    raw = panel._image.copy()
    original_bytes = {path: path.read_bytes() for path in plate[0].values()}
    result = _prepare_unmixing(settings, plate[0]['C01'].parent,
                              allow_calibrated=True).apply_calibrated(raw)
    expected = np.zeros(raw.shape, dtype=np.float32)
    expected[result.valid_slices] = np.maximum(result.image, 0)
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda *_: pytest.fail('loaded model'))
    filter_started, filter_release = threading.Event(), threading.Event()
    panel.show()
    try:
        panel.run_preview()
        qtbot.waitUntil(lambda: bool(panel._masks) and panel._worker is not None and not panel._worker.isRunning(), timeout=15000)
        assert panel._processing_provenance['unmixing']['mode'] == 'calibrated_live_preview'
        np.testing.assert_array_equal(panel._calibrated_display, expected)
        post, intensities = panel._filter_inputs()
        assert intensities == {'organelle': None}
        assert 'outside the valid ROI' in panel._status.toolTip()
        json.dumps(panel._processing_provenance)
        np.testing.assert_array_equal(panel._overlay_arguments()['image'], panel._display_image())
        for channel in (0, 1):
            panel._select_display_channel(channel)
            np.testing.assert_array_equal(panel._display_image(), expected[..., channel])
        panel._composite_roles = ('organelle',)
        np.testing.assert_array_equal(panel._composite_view(), expected[..., 0])
        panel._composite_roles = ()
        panel._on_hover(0, 0)
        assert 'display_channels=' in panel._hover_label.text()
        assert 'outside the valid ROI' in panel._hover_label.text()
        assert panel._history and panel._history[-1]['image'] is panel._calibrated_display
        layers = panel.comparison_layers()
        image_layers = [layer for layer in layers if getattr(layer, 'kind', None) == 'image']
        assert len(image_layers) == 3
        for channel in (0, 1):
            layer = next(item for item in image_layers if item.name == lp.tr('Channel {n}', n=channel + 1))
            np.testing.assert_array_equal(layer.array, lp._to_uint8(expected[..., channel],
                normalise=panel._normalise_check.isChecked(), lo_pct=float(panel._lo_pct.value()),
                hi_pct=float(panel._hi_pct.value())))
        assert len(panel.comparable_masks()) == 1
        edited = panel._compartment_widgets['organelle']['min_intensity']
        edited.setValue(1.)
        qtbot.waitUntil(lambda: not panel._masks and 'do not support intensity' in panel._status.text(),
                        timeout=10000)
        assert panel._raw_masks
        edited.setValue(0.)
        qtbot.waitUntil(lambda: bool(panel._masks), timeout=10000)
        np.testing.assert_array_equal(panel._calibrated_display, expected)
        qtbot.waitUntil(lambda: panel._filter_jobs.active_jobs() == 0, timeout=10000)
        original_filter = lp._filter_preview_masks

        def filter_before_failure(*args, **kwargs):
            result = original_filter(*args, **kwargs)
            filter_started.set()
            assert filter_release.wait(10)
            return result

        monkeypatch.setattr(lp, '_filter_preview_masks', filter_before_failure)
        panel._start_filter_job()
        qtbot.waitUntil(filter_started.is_set, timeout=10000)
        accepted_record = deepcopy(panel._processing_provenance)
        accepted_masks = dict(panel._raw_masks)
        rejected_token = panel._run_token
        panel._receive_calibrated_display({'image': expected, 'source_image_id': id(panel._image),
            'source_path': panel._path_full}, rejected_token)
        channel_index = next(index for index in range(panel._channel_box.count())
                             if panel._channel_box.itemData(index) == 'Ch 0')
        assert panel._channel_box.currentIndex() != channel_index
        panel._channel_box.setCurrentIndex(channel_index)
        assert panel.display_channel() == 0
        panel._on_processing_provenance(accepted_record, rejected_token)
        panel._on_worker_done(accepted_masks, '', rejected_token)
        assert panel._stale(rejected_token)
        assert panel._calibrated_display is None and not panel._masks
        filter_release.set()
        qtbot.waitUntil(lambda: panel._filter_jobs.active_jobs() == 0, timeout=10000)
        assert not panel._masks
        panel._on_flows_ready({'organelle': lp._to_uint8(expected)}, rejected_token)
        panel._on_cellprob_ready({'organelle': expected[..., 0]}, rejected_token)
        assert not panel._flows and not panel._cellprob
        cell_index = panel._object_box.findData('cell')
        assert cell_index >= 0
        panel._object_box.setCurrentIndex(cell_index)
        panel._compartment_widgets['cell']['min_area'].setValue(3)
        adapted = panel._build_request()
        assert adapted.preprocess_settings['cell_min_area'] == 3
        assert not adapted.preprocess_settings['object_filters'].get('cell')
        panel._settings['object_filters'] = {'cell': [{'property': 'area', 'min': 3, 'max': None}]}
        with pytest.raises(ValueError, match='custom object filters'):
            panel._build_request()
        panel._settings.pop('object_filters')
        panel._object_box.setCurrentIndex(index)
        old_token = panel._run_token
        stale_payload = {'image': expected, 'source_image_id': id(panel._image),
                         'source_path': panel._path_full}
        normal_source = plate[0]['B01']
        assert panel.load_image(normal_source), panel._status.text()
        assert panel._calibrated_display is None and panel._pending_calibrated_display is None
        assert not panel._masks and not panel.comparable_masks()
        current = panel._display_image().copy()
        panel._receive_calibrated_display(stale_payload, old_token)
        assert panel._pending_calibrated_display is None
        panel._receive_calibrated_display(stale_payload, panel._run_token)
        assert panel._pending_calibrated_display is None
        np.testing.assert_array_equal(panel._display_image(), current)
        before_history = panel._mask_view.picture().toImage()
        panel._on_compare_scrub(len(panel._history) - 1)
        assert 'different source or calibration' in panel._status.text()
        assert panel._mask_view.picture().toImage() == before_history
        np.testing.assert_array_equal(panel._display_image(), current)
        np.testing.assert_array_equal(raw, lp.load_source_payload(plate[0]['C01'])['array'])
        assert all(path.read_bytes() == payload for path, payload in original_bytes.items())
    finally:
        filter_release.set()
        panel.shutdown()



def test_gui_refuses_legacy_cells_when_switching_to_calibrated_organelle_restriction(
        calibrated_plate, qtbot, monkeypatch):
    plate, settings = calibrated_plate
    panel = lp.LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.apply_settings({'unmix': False, 'cell_channel': 0, 'organelle_channel': 0})
    assert panel.load_image(plate[0]['C01']), panel._status.text()
    raw = panel._image.copy()
    actual_classical = lp._classical_organelle_mask

    class CPUCellAdapter:
        def eval(self, image, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 invert=False, rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
            check_cellpose_eval_call(image, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D, stitch_threshold=stitch_threshold)
            return actual_classical(image, 'organelle', config()), []

    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda *_: CPUCellAdapter())
    try:
        cell_index = panel._object_box.findData('cell')
        assert cell_index >= 0
        panel._object_box.setCurrentIndex(cell_index)
        panel.run_preview()
        qtbot.waitUntil(lambda: bool(panel._masks.get('cell', np.array([])).any())
                        and panel._worker is not None and not panel._worker.isRunning(), timeout=15000)
        legacy_cells = panel._masks['cell'].copy()
        panel.apply_settings({**settings, 'organelle_channel': 0, 'organelle_mask_within_cells': True})
        organelle_index = panel._object_box.findData('organelle')
        assert organelle_index >= 0
        panel._object_box.setCurrentIndex(organelle_index)
        req = panel._build_request()
        assert req.preprocess_settings['organelle_mask_within_cells']
        np.testing.assert_array_equal(req.cell_mask, legacy_cells)
        monkeypatch.setattr(lp, '_classical_organelle_mask', lambda *_: pytest.fail('segmented restricted organelles'))
        panel.run_preview()
        qtbot.waitUntil(lambda: panel._worker is not None and not panel._worker.isRunning() and
            'cells generated in the same preview pass' in panel._status.text(), timeout=15000)
        assert panel._calibrated_display is None
        assert panel._processing_provenance.get('unmixing', {}).get('mode') != 'calibrated_live_preview'
        np.testing.assert_array_equal(panel._image, raw)
    finally:
        panel.shutdown()


@pytest.mark.parametrize('unmix', [False, True])
def test_per_plate_controls_refuse_preview_before_disabled_bypass_or_model(plate, monkeypatch, unmix):
    """A saved plate-scope request must not silently become an ordinary preview."""
    settings = config(unmix_per_plate=True, unmix_calibration='')
    settings['unmix'] = unmix
    req = request(plate, settings)
    before = {path: path.read_bytes() for path in plate[0].values()}
    original = req.image.copy()

    def forbidden(*args, **kwargs):
        """Keep the unsupported request from reaching inference."""
        pytest.fail('per-plate controls reached preview inference')

    monkeypatch.setattr(lp, 'preview_cellpose_model', forbidden)
    monkeypatch.setattr(lp, '_classical_organelle_mask', forbidden)
    with pytest.raises(ValueError, match='Per-plate spectral controls require calibrated organized 2D cell segmentation through V2'):
        lp._segment_multi(req)
    np.testing.assert_array_equal(req.image, original)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_real_preview_refuses_per_plate_controls_before_worker_dispatch(plate, qtbot, monkeypatch):
    """Show the shared refusal on the actual panel before model selection or worker creation."""
    panel = lp.LivePreviewPanel(threaded=True)
    qtbot.addWidget(panel)
    panel.apply_settings(dict(unmix_per_plate=True, unmix=False, unmix_calibration='',
                              cell_channel=0, organelle_channel=1))
    assert panel.load_image(plate[0]['C01']), panel._status.text()
    before = {path: path.read_bytes() for path in plate[0].values()}
    original = panel._image.copy()

    def forbidden(*args, **kwargs):
        """Fail on either model resolution or actual background worker construction."""
        pytest.fail('unsupported per-plate request was dispatched')

    monkeypatch.setattr(panel, '_model_for_this_pass', forbidden)
    monkeypatch.setattr(lp, '_PreviewWorker', forbidden)
    try:
        panel.run_preview()
        assert 'Per-plate spectral controls require calibrated organized 2D cell segmentation through V2' in panel._status.text()
        assert panel._settings['unmix_per_plate'] is True
        np.testing.assert_array_equal(panel._image, original)
        assert all(path.read_bytes() == content for path, content in before.items())
    finally:
        panel.shutdown()


@pytest.mark.parametrize('completion', ['stale-error', 'current-error', 'empty'])
def test_calibrated_preview_completion_keeps_only_current_valid_state_and_recovers(
        calibrated_plate, qtbot, monkeypatch, completion):
    plate, settings = calibrated_plate
    panel = lp.LivePreviewPanel(threaded=True)
    qtbot.addWidget(panel)
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda *a: pytest.fail('loaded a model'))
    panel.apply_settings({**settings, 'organelle_channel': 0})
    panel._object_box.setCurrentIndex(panel._object_box.findData('organelle'))
    assert panel.load_image(plate[0]['C01'])
    source = {p: p.read_bytes() for p in plate[0].values()}
    panel.show()
    try:
        panel.run_preview()
        qtbot.waitUntil(lambda: bool(panel._masks) and panel._worker is not None
                        and not panel._worker.isRunning(), timeout=15000)
        token = panel._run_token
        display = panel._calibrated_display.copy()
        provenance = deepcopy(panel._processing_provenance)
        masks = {key: value.copy() for key, value in panel._masks.items()}
        ready = []
        panel.preview_ready.connect(ready.append)
        if completion == 'stale-error':
            panel._on_worker_done(None, 'obsolete worker failure', token=token - 1)
            assert panel._processing_provenance == provenance
            np.testing.assert_array_equal(panel._calibrated_display, display)
            assert not ready
            for key, value in masks.items():
                np.testing.assert_array_equal(panel._masks[key], value)
        else:
            panel._on_worker_done(None if completion == 'current-error' else {},
                                  'controlled worker failure' if completion == 'current-error' else '',
                                  token=token)
            assert not panel._masks and not panel._raw_masks
            assert panel._calibrated_display is None
            assert panel._processing_provenance == {}
            assert panel._pending_calibrated_display is None and panel._pending_provenance is None
            assert not panel._flows and not panel._cellprob
            assert not panel._filter_timer.isActive()
            if completion == 'current-error':
                assert ready == [None]
                assert 'controlled worker failure' in panel._status.text()
            else:
                assert not ready
                assert 'no masks' in panel._status.text()
            panel.run_preview()
            qtbot.waitUntil(lambda: bool(panel._masks) and panel._worker is not None
                            and not panel._worker.isRunning(), timeout=15000)
            np.testing.assert_array_equal(panel._calibrated_display, display)
            assert panel._processing_provenance['unmixing'] == provenance['unmixing']
        assert {p: p.read_bytes() for p in source} == source
    finally:
        panel.close()


@pytest.mark.parametrize('morphology,method,visible,hidden', [
    ('spots', 'dog', ('dog_sigma_low', 'dog_sigma_high'), ('ridge_filter', 'hysteresis_low')),
    ('network', 'ridge', ('ridge_filter', 'ridge_sigmas'), ('dog_sigma_low', 'hysteresis_low')),
    ('network', 'hysteresis', ('hysteresis_low', 'hysteresis_high'), ('ridge_filter', 'dog_sigma_low')),
    ('irregular', 'adaptive', ('adaptive_block_size', 'morph_radius'), ('ridge_filter', 'dog_sigma_low')),
])
def test_detector_changes_retain_hidden_tuning_and_offer_only_pipeline_supported_methods(
        qtbot, morphology, method, visible, hidden):
    from spacr.organelle_types import LEGAL_METHODS

    panel = lp.LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    settings = {'organelle_morphology': morphology, 'organelle_method': method,
                'organelle_ridge_sigmas': [1, 3], 'organelle_dog_sigma_low': 2,
                'organelle_hysteresis_low': .2, 'organelle_hysteresis_high': .8}
    panel.apply_settings(settings)
    box = panel._organelle_widgets['method']
    offered = [box.itemData(i) or box.itemText(i) for i in range(box.count())]
    assert offered == list(LEGAL_METHODS[morphology]) and box.currentData() == method
    for suffix in visible:
        assert panel._organelle_field_visible(suffix, None, morphology)
    for suffix in hidden:
        assert not panel._organelle_field_visible(suffix, None, morphology)
    values = panel._organelle_settings()
    assert values['organelle_ridge_sigmas'] == [1, 3]
    assert values['organelle_hysteresis_low'] == .2
    assert values['organelle_hysteresis_high'] == .8
    assert values['organelle_dog_sigma_low'] == 2
    assert panel._worker is None
    panel.close()


def test_actual_channel_cycle_keeps_each_plane_and_the_two_object_composite_distinct(qtbot, plate):
    panel = lp.LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.apply_settings({'cell_channel': 0, 'nucleus_channel': 1})
    panel._object_box.setCurrentIndex(panel._object_box.findData('cell + nucleus'))
    assert panel.load_image(plate[0]['C01'])
    source = plate[0]['C01'].read_bytes()
    panel._cycle_index = 0
    panel._apply_cycle_stop()
    assert panel._composite_roles == () and panel.display_channel() == 0
    panel._cycle_view(1)
    assert panel._composite_roles == () and panel.display_channel() == 1
    panel._cycle_view(1)
    assert panel._composite_roles == ('cell', 'nucleus')
    composite = panel._composite_view()
    reference = panel._reference_display_image()
    np.testing.assert_array_equal(composite[..., 0], lp._select_channel(reference, 1))
    np.testing.assert_array_equal(composite[..., 1], lp._select_channel(reference, 0))
    np.testing.assert_array_equal(composite[..., 2], composite[..., 0])
    panel._object_box.setCurrentIndex(panel._object_box.findData('cell'))
    panel._apply_cycle_stop()
    assert panel._composite_roles == () and panel._cycle_stops() == []
    assert panel._composite_view() is None
    assert plate[0]['C01'].read_bytes() == source and panel._worker is None
    panel.close()


def test_invalid_ridge_scales_remain_visible_for_validation_and_blank_model_path_stays_unselected(qtbot):
    panel = lp.LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    field = panel._organelle_widgets['ridge_sigmas']
    for text in ('1, bad', '', '0, 2', '-1, 2'):
        field.setText(text)
        assert panel._widget_value(field) == text
        assert panel._organelle_settings()['organelle_ridge_sigmas'] == text
    field.setText('1, 2.5')
    assert panel._widget_value(field) == [1., 2.5]
    path = panel._organelle_widgets['unet_model_path']
    path.setText('   ')
    assert panel._widget_value(path) is None
    channel = panel.display_channel()
    panel._select_display_channel(None)
    assert panel.display_channel() == channel and panel._worker is None
    panel.close()


def test_auto_diameter_spelling_normalizes_without_changing_a_manual_measurement(qtbot):
    box = lp._DiameterSpinBox()
    qtbot.addWidget(box)
    for written in ('Auto', ' AUTO ', 'auto'):
        assert box.fixup(written) == 'auto'
        assert box.valueFromText(written) == 0
    box.setValue(12)
    before = box.text()
    box.fixup('12')
    assert box.value() == 12 and box.text() == before
