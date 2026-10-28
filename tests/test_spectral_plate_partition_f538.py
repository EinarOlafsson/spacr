"""Mixed calibrated plates remain independent datasets and never raw crop sources."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest

from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS, assert_declares_installed_eval_signature
from tests.conftest import check_cellpose_eval_call
from tests.test_pipeline_v2_illumination import plate_qualified_calibrated_sources
from tests.qt.test_measure_unmix_preview_f538 import plate as raw_preview_plate, panel

PARTITION_ACCESS = (hasattr(os, 'O_DIRECTORY') and hasattr(os, 'O_NOFOLLOW')
                    and hasattr(os, 'O_NONBLOCK') and os.open in os.supports_dir_fd)
needs_partition_access = pytest.mark.skipif(
    not PARTITION_ACCESS,
    reason='plate partitioning refuses hosts without descriptor-relative nonblocking access')


@pytest.fixture
def mixed_plate_inputs(plate_qualified_calibrated_sources, tmp_path, monkeypatch):
    from spacr import accelerator

    sample = plate_qualified_calibrated_sources
    source = tmp_path / 'mixed'
    source.mkdir()
    optical_map = {}
    for plate, spacing in zip(sample['plates'], ([.4, .3], [.6, .5])):
        for path, raw in plate['before'].items():
            (source / path.name).write_bytes(raw)
        calibration = json.loads(sample['calibration_bytes'])
        calibration['pixel_size_um'] = spacing
        for channel in calibration['channels']:
            channel['psf_pixel_size_um'] = spacing
        path = source / (plate['name'] + '_optics.json')
        path.write_text(json.dumps(calibration), encoding='utf-8')
        optical_map[plate['name']] = dict(calibration_file=path.name,
            pixel_size_um=spacing, source_context=dict(plate=plate['name'], time=2, z=1),
            acquisition_compatibility='user_asserts_same_channel_gains_exposures_and_linear_response')
    merged = source / 'merged'
    merged.mkdir()
    np.save(merged / 'original_camera.npy', np.arange(24, dtype=np.uint16).reshape(3, 4, 2))
    before = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
    observed = []
    constructed = []

    class CPUCellposeAdapter:
        def __init__(self, *, pretrained_model, gpu, device, use_bfloat16):
            constructed.append((gpu, str(device), use_bfloat16))
            assert gpu is False and str(device) == 'cpu' and use_bfloat16 is False
            self.pretrained_model = None

        def eval(self, images, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
                 do_3D=False, anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=None, compute_masks=True, progress=None):
            check_cellpose_eval_call(images, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D, stitch_threshold=stitch_threshold)
            masks = []
            for image in images:
                observed.append(np.asarray(image).copy())
                mask = np.zeros(np.asarray(image).shape[:2], np.uint16)
                mask[3:6, 3:6] = 7
                masks.append(mask)
            return masks, None, None

    assert_declares_installed_eval_signature(CPUCellposeAdapter.eval)
    monkeypatch.setattr('cellpose.models.CellposeModel', CPUCellposeAdapter)
    cpu = accelerator.Accelerator(kind='cpu', device='cpu', label='CPU partition oracle')
    monkeypatch.setattr(accelerator, 'resolve', lambda: cpu)
    settings = dict(sample['kwargs']['postprocess_settings'], unmix_partition_plates=True,
        unmix_plate_calibrations=json.dumps(optical_map), unmix_calibration='',
        unmix_pixel_size_um=None, unmix_matrix_file='', seg_qc='report')
    return dict(sample=sample, source=source, optical_map=optical_map, before=before,
        settings=settings, kwargs=dict(sample['kwargs'], postprocess_settings=settings),
        observed=observed, constructed=constructed)


def _unchanged(inputs):
    assert all(path.read_bytes() == raw for path, raw in inputs['before'].items())
    assert sorted(p.name for p in (inputs['source'] / 'merged').iterdir()) == ['original_camera.npy']


def _run(inputs):
    from spacr.pipeline_v2 import run_v2
    return run_v2(inputs['source'], **inputs['kwargs'])


@pytest.fixture
def completed_partition(mixed_plate_inputs):
    if not PARTITION_ACCESS:
        pytest.skip('plate partitioning refuses hosts without descriptor-relative nonblocking access')
    return mixed_plate_inputs, _run(mixed_plate_inputs)


@pytest.mark.skipif(PARTITION_ACCESS, reason='this host admits plate partitioning')
def test_partition_refuses_a_host_without_descriptor_relative_access(mixed_plate_inputs):
    inputs = mixed_plate_inputs
    before_paths = set(inputs['source'].rglob('*'))
    with pytest.raises(ValueError, match='unsupported hosts refuse'):
        _run(inputs)
    assert inputs['constructed'] == inputs['observed'] == []
    assert set(inputs['source'].rglob('*')) == before_paths
    assert not tuple(inputs['source'].glob('calibrated_plate_batch*'))
    _unchanged(inputs)


def test_real_mixed_plates_preserve_signed_fields_physical_units_and_plural_outputs(
        completed_partition, tmp_path):
    from spacr import pipeline_v2
    from spacr.measure import measure_calibrated_cells
    from spacr.tabular import read_table

    inputs, result = completed_partition
    assert result['partitioned'] is True
    assert not {'output_root', 'mapper', 'stacks', 'dst'}.intersection(result)
    assert [child['plate_id'] for child in result['plate_results']] == ['plate1', 'plate2']
    assert result['output_roots'] == [child['output_root'] for child in result['plate_results']]
    manifest_path = Path(result['manifest_path'])
    collection = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    assert (manifest['artifact'], manifest['version'], manifest['stage']) == (
        'spacr_calibrated_plate_batch', 1, 'calibrated_segmentation')
    assert manifest['segmentation_complete'] is True
    assert manifest['aggregate_complete'] is manifest['complete'] is False
    assert manifest['status'] == 'segmented'
    assert manifest['source_dataset'] == str(inputs['source'].resolve())
    marker = json.loads((collection / 'psf/segmentation_application.json').read_text())
    assert marker['configuration']['collection_only'] is True and marker['complete'] is False
    assert marker['fields'] == [] and marker['archives'] == {}
    original_mapper = pipeline_v2.FilenameMapper.discover(inputs['source'], metadata_type='custom',
        custom_regex=inputs['kwargs']['custom_regex'])
    original_ids = {(rows[0].plate, rows[0].well, rows[0].field, rows[0].time, rows[0].z): key
                    for key, rows in original_mapper.by_field().items()}
    all_field_ids = []
    for index, child in enumerate(result['plate_results']):
        plate = inputs['sample']['plates'][index]
        root = Path(child['output_root'])
        assert root == collection / f'plate_{index + 1:06d}'
        assert child['child_name'] == root.name
        record_path = root / 'psf/segmentation_application.json'
        record_bytes = record_path.read_bytes()
        record = json.loads(record_bytes)
        assert record['complete'] is True and len(record['fields']) == 3
        configuration = record['configuration']
        assert configuration['source_dataset'] == str(inputs['source'].resolve())
        assert configuration['physical_channel_order'] == [3, 1]
        np.testing.assert_allclose(configuration['unmixing']['matrix'], plate['matrix'], atol=1e-6, rtol=0)
        assert configuration['unmixing']['controls'] == inputs['sample']['master'][plate['name']]
        spacing = inputs['optical_map'][plate['name']]['pixel_size_um']
        assert configuration['unmixing']['calibration']['pixel_size_um'] == spacing
        manifest_child = manifest['children'][index]
        assert manifest_child['plate_id'] == plate['name'] and manifest_child['status'] == 'complete'
        assert manifest_child['record_sha256'] == hashlib.sha256(record_bytes).hexdigest()
        assert manifest_child['matrix'] == configuration['unmixing']['matrix']
        assert manifest_child['control_fields'] == configuration['unmixing']['fields']
        assert manifest_child['acquisition_compatibility_verified'] is False
        assert manifest_child['pixel_size_um'] == spacing
        original_optics = inputs['source'] / inputs['optical_map'][plate['name']]['calibration_file']
        assert manifest_child['calibration_source_sha256'] == hashlib.sha256(original_optics.read_bytes()).hexdigest()
        assert child['settings']['unmix_pixel_size_um'] == spacing
        assert Path(child['settings']['src']) == root
        assert json.loads(Path(child['settings']['unmix_calibration']).read_text()) == json.loads(original_optics.read_text())
        assert manifest_child['time_id'] == 2 and manifest_child['z_id'] == 1
        for stack in child['stacks']:
            field = record['calibrated_fields'][stack.field_id]
            identity = field['source_identity']
            key = tuple(identity[name] for name in ('plate', 'well', 'field', 'time', 'z'))
            assert stack.field_id == original_ids[key]
            all_field_ids.append(stack.field_id)
            assert identity['source_dataset'] == str(inputs['source'].resolve())
            packed = np.load(stack.path)
            labels = np.zeros((16, 18), np.uint16)
            labels[3:6, 3:6] = 7
            np.testing.assert_array_equal(packed[..., 2], labels)
            if identity['well'] == 'C01':
                np.testing.assert_allclose(np.load(root / field['signed_crop']['path']),
                    inputs['sample']['truth'], atol=1e-5, rtol=0)
        measured = measure_calibrated_cells(root, dst=tmp_path / (plate['name'] + '_cells'), channels=[1, 3])
        rows = read_table(measured['csv_path'], dtype={'measurement_field_id': str})
        assert measured['fields'] == 3 and measured['rows'] == 6
        target = rows[rows['wellID'] == 'C01']
        for channel, position in ((1, 1), (3, 0)):
            row = target[target['physical_channel_id'] == channel].iloc[0]
            values = inputs['sample']['truth'][3:6, 3:6, position].astype(np.float64)
            for statistic, expected in (('mean', values.mean()), ('min', values.min()),
                    ('max', values.max()), ('sum', values.sum()), ('std', values.std(ddof=0))):
                assert float(row['intensity_' + statistic]) == pytest.approx(expected, abs=1e-5)
            assert float(row['area_um2']) == pytest.approx(9 * spacing[0] * spacing[1])
            assert row['measurement_field_id'] == original_ids[(plate['name'], 'C01', 10, 2, 1)]
        provenance = json.loads(Path(measured['provenance_path']).read_text())
        assert provenance['source_provenance'] == record
        assert provenance['pixel_size_um'] == spacing and provenance['physical_channel_order'] == [1, 3]
    assert len(set(all_field_ids)) == len(all_field_ids) == 6
    assert len(inputs['observed']) == 6 and all(np.min(image) >= 0 for image in inputs['observed'])
    with pytest.raises(ValueError):
        measure_calibrated_cells(collection, dst=tmp_path / 'invalid_parent_measurements')
    assert not (tmp_path / 'invalid_parent_measurements').exists()
    _unchanged(inputs)
    _assert_captured_optics_are_independent(inputs, result)
    _unchanged(inputs)


def _assert_captured_optics_are_independent(inputs, result):
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    children = result['plate_results']
    external_paths = [Path(inputs['source']) / inputs['optical_map'][child['plate_id']]['calibration_file']
                      for child in children]
    originals = [path.read_bytes() for path in external_paths]
    try:
        external_paths[0].unlink()
        changed = json.loads(originals[1])
        changed['pixel_size_um'] = [.9, .9]
        for channel in changed['channels']:
            channel['psf_pixel_size_um'] = [.9, .9]
        external_paths[1].write_text(json.dumps(changed), encoding='utf-8')
        assert not external_paths[0].exists()
        assert external_paths[1].read_bytes() != originals[1]
        for child, original in zip(children, originals):
            root = Path(child['output_root'])
            record_path = root / 'psf/segmentation_application.json'
            record_bytes = record_path.read_bytes()
            with _open_calibrated_cell_dataset(root) as dataset:
                assert dataset.physical_channel_order == (3, 1)
                calibration = dataset.configuration['unmixing']['calibration']
                assert calibration['pixel_size_um'] == inputs['optical_map'][child['plate_id']]['pixel_size_um']
                assert calibration['source_sha256'] == hashlib.sha256(original).hexdigest()
                count = 0
                for field in dataset.fields():
                    count += 1
                    assert np.isfinite(field['signed']).all()
                    assert set(np.unique(field['labels'])) == {0, 7}
                    if field['source_identity']['well'] == 'C01':
                        assert float(np.min(field['signed'])) == pytest.approx(-2., abs=1e-5)
                assert count == 3
            assert record_path.read_bytes() == record_bytes
    finally:
        for path, raw in zip(external_paths, originals):
            path.write_bytes(raw)
    assert all(path.read_bytes() == raw for path, raw in zip(external_paths, originals))


@needs_partition_access
def test_actual_partition_cancellation_preserves_the_first_completed_child(mixed_plate_inputs, monkeypatch):
    from spacr import psf_pipeline
    from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    inputs = mixed_plate_inputs
    token = CancellationToken()
    original = psf_pipeline._SegmentationPSFSession.finish
    finished = []

    def finish_then_request_stop(session, expected_fields):
        result = original(session, expected_fields)
        finished.append(Path(session.root))
        if len(finished) == 1:
            token.cancel('stop after the first durable plate')
        return result

    monkeypatch.setattr(psf_pipeline._SegmentationPSFSession, 'finish', finish_then_request_stop)
    with installed_token(token), pytest.raises(PipelineCancelled, match='first durable plate'):
        _run(inputs)
    assert len(finished) == 1
    manifest_path = next(inputs['source'].glob('calibrated_plate_batch*/plate_batch.json'))
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'cancelled'
    assert not manifest['complete'] and not manifest['aggregate_complete'] and not manifest['segmentation_complete']
    assert [child['status'] for child in manifest['children']] == ['complete', 'pending']
    with _open_calibrated_cell_dataset(finished[0]) as dataset:
        assert len(list(dataset.fields())) == 3
    _unchanged(inputs)


@needs_partition_access
def test_bad_later_plate_optics_refuse_before_any_collection_or_model(mixed_plate_inputs):
    inputs = mixed_plate_inputs
    optical_map = deepcopy(inputs['optical_map'])
    optical_map['plate2']['source_context']['time'] = 99
    inputs['settings']['unmix_plate_calibrations'] = json.dumps(optical_map)
    before_paths = set(inputs['source'].rglob('*'))
    with pytest.raises(ValueError):
        _run(inputs)
    assert inputs['constructed'] == inputs['observed'] == []
    assert set(inputs['source'].rglob('*')) == before_paths
    assert not tuple(inputs['source'].glob('calibrated_plate_batch*'))
    _unchanged(inputs)


@needs_partition_access
def test_later_plate_rank_deficient_real_controls_refuse_before_any_collection_or_model(mixed_plate_inputs):
    import tifffile

    inputs = mixed_plate_inputs
    for channel, well in inputs['sample']['master']['plate2'].items():
        amplitude = 80. if channel == '3' else 100.
        pixels = np.zeros((16, 18), np.float32)
        pixels[3:12, 3:14] = amplitude
        for physical_id in (3, 1):
            path = inputs['source'] / f'plate2_{well[0]}_F10_T2_Z1_C{physical_id}.tif'
            tifffile.imwrite(path, pixels)
    inputs['before'] = {path: path.read_bytes() for path in inputs['source'].rglob('*') if path.is_file()}
    before_paths = set(inputs['source'].rglob('*'))
    with pytest.raises(ValueError, match='too close to singular'):
        _run(inputs)
    assert inputs['constructed'] == inputs['observed'] == []
    assert set(inputs['source'].rglob('*')) == before_paths
    assert not tuple(inputs['source'].glob('calibrated_plate_batch*'))
    _unchanged(inputs)


def _assert_parent_refusals(root, array_path, cached, destination):
    from spacr.errors import ConfigurationError
    from spacr import io, measure
    from spacr.settings import get_measure_crop_settings
    from spacr.qt.widgets import measure_preview as preview

    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    for source in (root, root / 'merged'):
        with pytest.raises(ConfigurationError, match='not supported by Measure'):
            options = get_measure_crop_settings(dict(src=str(source), channels=[0, 1], unmix=False,
                unmix_per_plate=False, unmix_partition_plates=False, save_png=True,
                cell_mask_dim=2, nucleus_mask_dim=None, pathogen_mask_dim=None, n_jobs=1))
            measure.measure_crop(options)
        with pytest.raises(ConfigurationError, match='not supported by Measure'):
            io.generate_dataset(dict(src=str(source), crop_source='merged', unmix=False))
    for path in (root, root / 'merged', root / array_path.name, array_path):
        loaded = preview.load_merged_array(str(path))
        assert loaded['data'] is None and 'not supported by Measure' in loaded['error']
    kwargs = dict(mask_dim=2, channels=[1, 0, 1], normalize=False,
                  mask_background=False, limit=10)
    categories = dict(object='cell', cell_dim=2, dims={}, minima={}, uninfected=True)
    checked = preview._compute_checked_crops([array_path], array_path, cached, kwargs,
        categories, threading.Event(), {'unmix': False, 'unmix_partition_plates': False})
    assert checked['crops'] == [] and 'not supported by Measure' in checked['error']
    entry = dict(crop=np.arange(60, dtype=np.uint8).reshape(4, 5, 3), source_path=str(array_path),
        object_key=(str(array_path), 'cell', 7), label=7, area=9, bbox=(3, 3, 6, 6), included=True)
    exported = preview._write_preview_crop_export(destination, [entry], {}, 'rgb', threading.Event())
    assert 'not supported by Measure' in exported['error']
    assert not destination.exists() and not tuple(destination.parent.glob('.spacr-preview-*'))
    assert {path: path.read_bytes() for path in root.rglob('*') if path.is_file()} == before


def test_populated_collection_and_direct_parent_arrays_never_become_raw_crop_sources(
        completed_partition, tmp_path, qtbot):
    from spacr.qt.widgets import measure_preview as preview

    inputs, result = completed_partition
    root = Path(result['manifest_path']).parent
    actual = np.load(result['plate_results'][0]['stacks'][0].path)
    (root / 'merged').mkdir()
    path = root / 'merged' / 'copied_display.npy'
    np.save(path, actual)
    np.save(root / path.name, actual)
    _assert_parent_refusals(root, path, actual.copy(), tmp_path / 'parent_crops')
    widget = preview.MeasurePreviewPanel(threaded=True)
    qtbot.addWidget(widget)
    try:
        widget.load_array(str(root))
        qtbot.waitUntil(lambda: 'not supported by Measure' in widget._status.text(), timeout=10000)
        assert widget._data is None and not widget._crops
    finally:
        widget._crop_size_timer.stop()
        widget.shutdown()
    _unchanged(inputs)


@pytest.mark.parametrize('marker', ['malformed', pytest.param('dangling', marks=pytest.mark.needs_symlink)])
def test_present_unreadable_collection_manifest_cannot_fall_back_to_raw(tmp_path, marker):
    root = tmp_path / 'collection'
    (root / 'merged').mkdir(parents=True)
    data = np.zeros((16, 18, 3), np.float32)
    data[3:6, 3:6, 2] = 7
    path = root / 'merged' / 'field.npy'
    np.save(path, data)
    np.save(root / path.name, data)
    manifest = root / 'plate_batch.json'
    if marker == 'malformed':
        manifest.write_bytes(b'{unfinished JSON')
    else:
        manifest.symlink_to(root / 'absent-manifest.json')
    _assert_parent_refusals(root, path, data.copy(), tmp_path / 'refused_crops')
    if marker == 'dangling':
        assert manifest.is_symlink() and not manifest.exists()
    else:
        assert manifest.read_bytes() == b'{unfinished JSON'


def test_saved_partition_scope_cannot_disappear_from_the_native_crop_preview(qtbot, monkeypatch, tmp_path, raw_preview_plate):
    from spacr.qt.widgets import measure_preview as preview

    fixture = raw_preview_plate
    widget = panel(qtbot, fixture, monkeypatch)
    before = {path: path.read_bytes() for path in fixture[0].values()}
    saved = {'plate1': {'calibration_file': 'optics.json', 'pixel_size_um': [.4, .3],
        'source_context': {'plate': 'plate1', 'time': 2, 'z': 1},
        'acquisition_compatibility': 'user_asserts_same_channel_gains_exposures_and_linear_response'}}
    try:
        widget.apply_settings(dict(unmix=False, unmix_per_plate=False,
            unmix_partition_plates=True, unmix_plate_calibrations=json.dumps(saved)))
        assert widget._unmix_settings['unmix_partition_plates'] is True
        assert json.loads(widget._unmix_settings['unmix_plate_calibrations']) == saved
        assert not widget._unmix_btn.isChecked()
        widget.run_preview()
        widget._start_crop_export(tmp_path / 'unsupported_partition_crops')
        assert not widget._crops
        assert 'partition' in widget._status.text().lower()
        assert not (tmp_path / 'unsupported_partition_crops').exists()
        assert all(path.read_bytes() == raw for path, raw in before.items())
    finally:
        widget._crop_size_timer.stop()
        widget.shutdown()


def test_png_only_dataset_export_stays_supported_without_any_merged_folder(tmp_path):
    import tarfile
    from PIL import Image
    from spacr.io import generate_dataset
    from spacr.utils import filepaths_to_database

    root = tmp_path / 'raw_png_project'
    crops = root / 'cell_png'
    crops.mkdir(parents=True)
    (root / 'measurements').mkdir()
    path = crops / 'plate1_A01_5_7.png'
    pixels = np.arange(108, dtype=np.uint8).reshape(6, 6, 3)
    Image.fromarray(pixels).save(path)
    filepaths_to_database([str(path)], {'timelapse': False}, str(root), 'cell')
    before = path.read_bytes()
    output = generate_dataset(dict(src=str(root), crop_source='png',
        file_metadata='cell_png', sample=None, experiment='legacy_png_only'))
    with tarfile.open(output) as archive:
        member = next(member for member in archive.getmembers() if member.name.endswith('.png'))
        with Image.open(archive.extractfile(member)) as image:
            np.testing.assert_array_equal(np.asarray(image), pixels)
    assert not (root / 'merged').exists() and path.read_bytes() == before


def _core_settings(inputs):
    return dict(deepcopy(inputs['settings']), src=str(inputs['source']), pipeline_style='v2',
        cell_channel=3, nucleus_channel=1, pathogen_channel=None, organelle_channel=None,
        metadata_type='custom', custom_regex=inputs['kwargs']['custom_regex'], verbose=False)


def test_plural_core_qc_uses_each_child_settings_without_mutating_its_siblings(
        completed_partition, monkeypatch):
    from spacr import core, pipeline_v2, seg_qc

    inputs, result = completed_partition
    for index, child in enumerate(result['plate_results']):
        child['settings']['seg_qc'] = 'report' if index == 0 else 'off'
        child['settings']['oracle_nested'] = {'values': [index]}
    before = [deepcopy(child['settings']) for child in result['plate_results']]
    original_settings = deepcopy(inputs['settings'])
    invoked = []
    modes = []
    actual_mode = seg_qc.qc_mode

    def return_completed(src, **options):
        assert Path(src) == inputs['source']
        return result

    def mode(settings):
        if 'oracle_nested' in settings:
            modes.append(deepcopy(settings))
        return actual_mode(settings)

    def score(root, settings, object_type='cell', *, strict=False):
        assert strict is True and object_type == 'cell'
        assert Path(root) == Path(result['plate_results'][0]['output_root'])
        assert Path(settings['src']) == Path(root)
        invoked.append(Path(root))
        settings['unmix_pixel_size_um'][0] = 99
        settings['oracle_nested']['values'].append('mutated by QC')
        return {'mode': 'report'}

    monkeypatch.setattr(pipeline_v2, 'run_v2', return_completed)
    monkeypatch.setattr(seg_qc, 'qc_mode', mode)
    monkeypatch.setattr(core, '_score_v2_masks', score)
    core.preprocess_generate_masks(_core_settings(inputs))
    assert invoked == [Path(result['plate_results'][0]['output_root'])]
    assert len(modes) == 2
    for index, settings in enumerate(modes):
        child = result['plate_results'][index]
        assert Path(settings['src']) == Path(child['output_root'])
        assert settings['unmix_calibration'] == before[index]['unmix_calibration']
        assert settings['unmix_pixel_size_um'] == before[index]['unmix_pixel_size_um']
        assert settings['oracle_nested'] == before[index]['oracle_nested']
    assert [child['settings'] for child in result['plate_results']] == before
    assert inputs['settings'] == original_settings
    manifest = json.loads(Path(result['manifest_path']).read_text())
    assert manifest['aggregate_complete'] is manifest['complete'] is True
    assert [child['qc_status'] for child in manifest['children']] == ['complete', 'off']
    _unchanged(inputs)


@pytest.mark.parametrize('cancel_after', [1, 2], ids=['between-child-qc', 'after-last-child-qc'])
def test_real_qc_cancellation_cannot_leave_a_collection_claiming_aggregate_success(
        completed_partition, monkeypatch, cancel_after):
    from spacr import core, pipeline_v2
    from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    inputs, result = completed_partition
    token = CancellationToken()
    scored = []
    original = core._score_v2_masks

    def score_then_request_stop(root, settings, object_type='cell', *, strict=False):
        score = original(root, settings, object_type=object_type, strict=strict)
        assert score is not None and score['mode'] == 'report'
        scored.append(Path(root))
        if len(scored) == cancel_after:
            token.cancel('stop after durable child QC')
        return score

    monkeypatch.setattr(pipeline_v2, 'run_v2', lambda *args, **options: result)
    monkeypatch.setattr(core, '_score_v2_masks', score_then_request_stop)
    with installed_token(token), pytest.raises(PipelineCancelled, match='durable child QC'):
        core.preprocess_generate_masks(_core_settings(inputs))
    assert len(scored) == cancel_after
    manifest = json.loads(Path(result['manifest_path']).read_text())
    assert manifest['segmentation_complete'] is True
    assert manifest['status'] == 'cancelled'
    assert manifest['aggregate_complete'] is manifest['complete'] is False
    assert [child['status'] for child in manifest['children']] == ['complete', 'complete']
    expected_qc = ['complete', 'pending'] if cancel_after == 1 else ['complete', 'complete']
    assert [child['qc_status'] for child in manifest['children']] == expected_qc
    for child in result['plate_results']:
        with _open_calibrated_cell_dataset(child['output_root']) as dataset:
            assert len(list(dataset.fields())) == 3
    _unchanged(inputs)


@pytest.mark.parametrize('failure', ['write-error', 'cancel'])
def test_real_inner_qc_failures_propagate_only_on_the_strict_partition_route(
        completed_partition, monkeypatch, failure):
    from spacr import core, pipeline_v2, seg_qc
    from spacr.cancellation import CancellationToken, PipelineCancelled, checkpoint, installed_token

    inputs, result = completed_partition
    root = Path(result['plate_results'][0]['output_root'])
    settings = dict(result['plate_results'][0]['settings'], seg_qc='report', verbose=False)
    actual_qc = seg_qc.run_segmentation_qc
    if failure == 'write-error':
        (root / 'qc').write_bytes(b'an existing file owns the QC destination')
        expected_type = FileExistsError
    else:
        expected_type = PipelineCancelled

        def cancelled_inner(*args, **kwargs):
            value = actual_qc(*args, **kwargs)
            assert value is not None
            from spacr.cancellation import current_token
            current_token().cancel('cancel inside the real scorecard callback')
            checkpoint()
            return value

        monkeypatch.setattr(seg_qc, 'run_segmentation_qc', cancelled_inner)
    with installed_token(CancellationToken()):
        assert core._score_v2_masks(root, settings) is None
    with installed_token(CancellationToken()), pytest.raises(expected_type):
        core._score_v2_masks(root, settings, strict=True)
    if failure == 'write-error':
        class LegacyFailure(FileExistsError):
            add_note = None

        original_failure = LegacyFailure('real QC write refused, without exception notes')

        def legacy_failure(*args, **kwargs):
            try:
                return actual_qc(*args, **kwargs)
            except FileExistsError as exc:
                raise original_failure from exc

        actual_record = pipeline_v2._record_partition_qc

        def refuse_failed_status(result, child, status, error=None):
            if status == 'failed':
                raise OSError('status destination became unavailable')
            return actual_record(result, child, status, error=error)

        monkeypatch.setattr(seg_qc, 'run_segmentation_qc', legacy_failure)
        monkeypatch.setattr(pipeline_v2, 'run_v2', lambda *args, **options: result)
        monkeypatch.setattr(pipeline_v2, '_record_partition_qc', refuse_failed_status)
        with pytest.raises(LegacyFailure) as caught:
            core.preprocess_generate_masks(_core_settings(inputs))
        assert caught.value is original_failure
        assert (root / 'qc').read_bytes() == b'an existing file owns the QC destination'
    _unchanged(inputs)


@needs_partition_access
def test_later_plate_oversized_source_headers_refuse_before_any_pixel_decode(
        mixed_plate_inputs, monkeypatch):
    """All plates' real header bounds precede control decoding and output creation."""
    import tifffile
    from spacr import pipeline_v2

    inputs = mixed_plate_inputs
    source = inputs['source']
    shape = (4097, 2048)
    assert shape[0] * shape[1] * len(inputs['kwargs']['channels']) > 16 * 1024 ** 2
    replaced = {source / f'plate2_C01_F10_T2_Z1_C{channel}.tif'
                for channel in inputs['kwargs']['channels']}
    for path in replaced:
        assert path in inputs['before']
        image = tifffile.memmap(path, shape=shape, dtype=np.uint16,
                               photometric='minisblack')
        try:
            image[0, 0] = 17
            image[-1, -1] = 93
            image.flush()
        finally:
            image._mmap.close()
        with tifffile.TiffFile(path) as header:
            assert len(header.series) == 1
            assert header.series[0].shape == shape
            assert header.series[0].dtype == np.dtype(np.uint16)
    before_paths = set(source.rglob('*'))
    before_stats = {path: path.stat() for path in replaced}

    def forbid_pixel_decode(*args, **kwargs):
        pytest.fail('Oversized acquired headers must refuse before any control or target pixel decode')

    monkeypatch.setattr(pipeline_v2, '_read_plane', forbid_pixel_decode)
    with pytest.raises(ValueError):
        _run(inputs)
    assert inputs['constructed'] == inputs['observed'] == []
    assert set(source.rglob('*')) == before_paths
    assert not tuple(source.glob('calibrated_plate_batch*'))
    assert all(path.read_bytes() == raw for path, raw in inputs['before'].items()
               if path not in replaced)
    for path in replaced:
        before, after = before_stats[path], path.stat()
        assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        image = tifffile.memmap(path, mode='r')
        try:
            assert image.shape == shape and image.dtype == np.dtype(np.uint16)
            assert int(image[0, 0]) == 17 and int(image[-1, -1]) == 93
            assert int(image[shape[0] // 2, shape[1] // 2]) == 0
        finally:
            image._mmap.close()


@needs_partition_access
@pytest.mark.parametrize('damage', ['missing-channel', 'duplicate-channel', 'shape',
                                    'nonfinite-control', 'multiple-times'])
def test_later_plate_acquisition_refuses_without_parent_or_model(
        mixed_plate_inputs, damage):
    """A bad later plate cannot leave even the earlier valid plate's outputs."""
    import tifffile

    inputs = mixed_plate_inputs
    source = inputs['source']
    target = source / 'plate2_C01_F10_T2_Z1_C3.tif'
    if damage == 'missing-channel':
        target.unlink()
    elif damage == 'duplicate-channel':
        target.with_suffix('.tiff').write_bytes(target.read_bytes())
        inputs['kwargs']['custom_regex'] = inputs['kwargs']['custom_regex'].replace(
            r'\.tif$', r'\.tiff?$')
    elif damage == 'shape':
        tifffile.imwrite(target, np.ones((17, 18), np.float32))
    elif damage == 'nonfinite-control':
        control = source / 'plate2_B01_F10_T2_Z1_C3.tif'
        pixels = tifffile.imread(control)
        pixels[4, 4] = np.nan
        tifffile.imwrite(control, pixels)
    else:
        target.rename(target.with_name(target.name.replace('_T2_', '_T3_')))
    inputs['before'] = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
    paths = set(source.rglob('*'))
    with pytest.raises(ValueError):
        _run(inputs)
    assert inputs['constructed'] == inputs['observed'] == []
    assert set(source.rglob('*')) == paths
    assert not tuple(source.glob('calibrated_plate_batch*'))
    _unchanged(inputs)


@needs_partition_access
def test_an_occupied_collection_is_preserved_and_never_resumed(mixed_plate_inputs):
    """A new run chooses a fresh parent without touching an existing run marker."""
    inputs = mixed_plate_inputs
    occupied = inputs['source'] / 'calibrated_plate_batch'
    occupied.mkdir()
    marker = occupied / 'plate_batch.json'
    original = b'{"artifact":"previous independent run","complete":false}\n'
    marker.write_bytes(original)
    result = _run(inputs)
    assert Path(result['manifest_path']).parent.name == 'calibrated_plate_batch_0001'
    assert marker.read_bytes() == original
    assert sorted(path.name for path in occupied.iterdir()) == ['plate_batch.json']
    record = json.loads(Path(result['manifest_path']).read_text())
    assert record['segmentation_complete'] is True
    assert record['complete'] is record['aggregate_complete'] is False
    assert [child['status'] for child in record['children']] == ['complete', 'complete']
    _unchanged(inputs)


@needs_partition_access
@pytest.mark.parametrize('failure', ['model-error', 'cancel', 'status-write-error'])
def test_active_child_failure_preserves_the_completed_signed_sibling(
        mixed_plate_inputs, monkeypatch, failure):
    """Original child failures survive journaling, with no aggregate success claim."""
    from functools import wraps
    from cellpose import models
    from spacr import psf_pipeline
    from spacr.cancellation import PipelineCancelled
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    inputs = mixed_plate_inputs
    original_eval = models.CellposeModel.eval
    original_write = psf_pipeline._write_json
    error = (PipelineCancelled('stop the second plate') if failure == 'cancel'
             else RuntimeError('private model detail must not enter the manifest'))
    evaluated = []

    @wraps(original_eval)
    def fail_second_plate(model, *args, **kwargs):
        evaluated.append(model)
        if len(evaluated) == 2:
            raise error
        return original_eval(model, *args, **kwargs)

    def write_status(path, record):
        if (failure == 'status-write-error' and Path(path).name == 'plate_batch.json'
                and record.get('status') == 'failed'):
            raise OSError('collection status destination unavailable')
        return original_write(path, record)

    monkeypatch.setattr(models.CellposeModel, 'eval', fail_second_plate)
    monkeypatch.setattr(psf_pipeline, '_write_json', write_status)
    with pytest.raises(type(error)) as caught:
        _run(inputs)
    assert caught.value is error and len(evaluated) == 2
    manifest_path = next(inputs['source'].glob('calibrated_plate_batch*/plate_batch.json'))
    record = json.loads(manifest_path.read_text())
    assert record['complete'] is record['aggregate_complete'] is False
    assert record['segmentation_complete'] is False
    assert record['children'][0]['status'] == 'complete'
    if failure == 'status-write-error':
        assert record['status'] == record['children'][1]['status'] == 'running'
        if callable(getattr(error, 'add_note', None)):
            assert any('OSError' in note for note in error.__notes__)
    else:
        expected = 'cancelled' if failure == 'cancel' else 'failed'
        assert record['status'] == record['children'][1]['status'] == expected
        assert record['error_type'] == record['children'][1]['error_type'] == type(error).__name__
        assert str(error) not in manifest_path.read_text()
    first = manifest_path.parent / record['children'][0]['path']
    application = first / 'psf/segmentation_application.json'
    raw = application.read_bytes()
    assert record['children'][0]['record_sha256'] == hashlib.sha256(raw).hexdigest()
    with _open_calibrated_cell_dataset(first) as dataset:
        assert len(list(dataset.fields())) == 3
    assert application.read_bytes() == raw
    second = manifest_path.parent / record['children'][1]['path']
    assert json.loads((second / 'psf/segmentation_application.json').read_text())['complete'] is False
    _unchanged(inputs)


@pytest.mark.parametrize('damage', ['status', 'artifact', 'unknown-child', 'child-path'])
def test_qc_cannot_rewrite_an_untrusted_collection_binding(
        completed_partition, damage):
    """Bad QC authority leaves both collection bytes and signed children intact."""
    from spacr import pipeline_v2

    inputs, result = completed_partition
    path = Path(result['manifest_path'])
    child = dict(result['plate_results'][0])
    if damage == 'artifact':
        record = json.loads(path.read_text())
        record['artifact'] = 'an unrelated run'
        path.write_text(json.dumps(record))
    elif damage == 'unknown-child':
        child['child_name'] = 'not-a-returned-child'
    elif damage == 'child-path':
        child['output_root'] = inputs['source']
    before = path.read_bytes()
    signed = {Path(entry['output_root']) / 'psf/segmentation_application.json':
              (Path(entry['output_root']) / 'psf/segmentation_application.json').read_bytes()
              for entry in result['plate_results']}
    with pytest.raises(ValueError):
        pipeline_v2._record_partition_qc(result, child,
            'made-up-success' if damage == 'status' else 'complete')
    assert path.read_bytes() == before
    assert all(application.read_bytes() == raw for application, raw in signed.items())
    _unchanged(inputs)


def test_qc_atomic_replace_failure_preserves_the_previous_truthful_manifest(
        completed_partition, monkeypatch):
    """A failed rename cannot publish success or leave a temporary QC journal."""
    from spacr import pipeline_v2, psf_pipeline

    inputs, result = completed_partition
    path = Path(result['manifest_path'])
    before = path.read_bytes()
    original_replace = psf_pipeline.os.replace
    attempted = []

    def refuse_collection_replace(source, destination):
        if Path(destination) == path:
            proposed = json.loads(Path(source).read_text())
            assert proposed['children'][0]['qc_status'] == 'complete'
            attempted.append(Path(source))
            raise OSError('atomic collection replace refused')
        return original_replace(source, destination)

    monkeypatch.setattr(psf_pipeline.os, 'replace', refuse_collection_replace)
    with pytest.raises(OSError, match='atomic collection replace refused'):
        pipeline_v2._record_partition_qc(result, result['plate_results'][0], 'complete')
    assert len(attempted) == 1 and not attempted[0].exists()
    assert path.read_bytes() == before
    record = json.loads(before)
    assert record['status'] == 'segmented' and record['complete'] is False
    assert [entry['qc_status'] for entry in record['children']] == ['pending', 'pending']
    _unchanged(inputs)


@needs_partition_access
def test_refused_packed_control_closes_its_actual_mapping_before_model_use(
        mixed_plate_inputs, monkeypatch):
    """A changed child control is rejected and its real mmap handle is retired."""
    from spacr import pipeline_v2

    inputs = mixed_plate_inputs
    original_stream = pipeline_v2.stream_originals_to_stack
    original_load = pipeline_v2.np.load
    mapped = []

    def changed_control(*args, **kwargs):
        stacks = original_stream(*args, **kwargs)
        values = original_load(stacks[0].path, allow_pickle=False)
        values[4, 4, 0] += 1
        np.save(stacks[0].path, values)
        return stacks

    def observe_mapping(path, *args, **kwargs):
        values = original_load(path, *args, **kwargs)
        if kwargs.get('mmap_mode') == 'r' and Path(path).parent.name == 'merged':
            mapped.append(values)
        return values

    monkeypatch.setattr(pipeline_v2, 'stream_originals_to_stack', changed_control)
    monkeypatch.setattr(pipeline_v2.np, 'load', observe_mapping)
    with pytest.raises(ValueError):
        _run(inputs)
    assert mapped and all(value._mmap.closed for value in mapped)
    assert inputs['constructed'] == inputs['observed'] == []
    path = next(inputs['source'].glob('calibrated_plate_batch*/plate_batch.json'))
    record = json.loads(path.read_text())
    assert record['status'] == record['children'][0]['status'] == 'failed'
    assert record['children'][1]['status'] == 'pending'
    assert record['complete'] is record['segmentation_complete'] is False
    _unchanged(inputs)
