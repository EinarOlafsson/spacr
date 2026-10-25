"""Supplied matrices remain explicit bounded artifacts on the calibrated V2 route."""

import copy
import hashlib
import json
import os

import numpy as np
import pytest

from spacr import pipeline_v2 as PV
from spacr import psf_pipeline as PP
from tests.test_pipeline_v2_illumination import (
    plate_qualified_calibrated_sources,
)
from tests.test_spectral_unmixing import (
    _rewrite_calibrated_cell_record, calibrated_batch_inputs, calibrated_plate,
)

SUPPLIED_MATRIX_ACCESS = (hasattr(os, 'O_DIRECTORY') and hasattr(os, 'O_NOFOLLOW')
                          and hasattr(os, 'O_NONBLOCK') and os.open in os.supports_dir_fd
                          and os.mkdir in os.supports_dir_fd)
needs_supplied_matrix_access = pytest.mark.skipif(
    not SUPPLIED_MATRIX_ACCESS,
    reason='a supplied matrix refuses hosts without descriptor-relative no-follow access')


@pytest.fixture
def supplied_matrix(plate_qualified_calibrated_sources, tmp_path):
    sample = plate_qualified_calibrated_sources
    plate = sample['plates'][0]
    payload = dict(
        artifact='spacr_spectral_matrix', version=1,
        source_context=dict(plate=plate['name'], time=2, z=1), channels=[3, 1],
        orientation='rows_readings_columns_dyes', matrix=plate['matrix'].tolist(),
        background_percentile=0., background_operator='per_field_percentile_subtract_solve_restore',
        calibration_source_sha256=hashlib.sha256(sample['calibration_bytes']).hexdigest(),
        acquisition_compatibility='user_asserts_same_channel_gains_exposures_and_linear_response')
    path = tmp_path / 'supplied_matrix.json'
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    settings = dict(sample['kwargs']['postprocess_settings'], unmix_per_plate=False,
                    unmix_controls='', unmix_matrix_file=str(path))
    kwargs = dict(sample['kwargs'], postprocess_settings=settings)
    return dict(sample=sample, plate=plate, payload=payload, path=path,
                settings=settings, kwargs=kwargs)


def _run_matrix(sample):
    return PV.run_v2(sample['plate']['source'], **sample['kwargs'])


def _source_unchanged(sample):
    assert all(path.read_bytes() == before for path, before in sample['plate']['before'].items())
    assert sample['sample']['calibration_path'].read_bytes() == sample['sample']['calibration_bytes']


@pytest.fixture
def completed_supplied_session(supplied_matrix, tmp_path):
    import tifffile

    sample = supplied_matrix
    identity = dict(plate='plate1', well='C01', field=10, time=2, z=1,
                    source_dataset=str(sample['plate']['source'].resolve()))
    identities = {'target': identity, 'stack_target.npy': identity}
    plan = PP._prepare_unmixing(sample['settings'], sample['plate']['source'],
        channels=(3, 1), allow_calibrated=True, channel_positions={3: 0, 1: 1},
        field_identities=identities)
    raw = sample['path'].read_bytes()
    sample['path'].unlink()
    root = tmp_path / 'session-output'
    PP._initialize_calibrated_output(root, sample['settings'])
    session = PP._prepare_segmentation_psf(sample['settings'], root, [0, 1],
        pipeline_style='v2', source_channels=(3, 1), field_identities=identities,
        prepared_unmix=plan)
    camera = np.stack([tifffile.imread(sample['plate']['source'] /
        f'plate1_C01_F10_T2_Z1_C{channel}.tif') for channel in (3, 1)], axis=-1)
    original = camera.copy()
    result = session.prepare_calibrated(camera, 'target')
    labels = np.zeros(result.image.shape[:2], dtype=np.uint16)
    labels[3:6, 3:6] = 7
    mask = session.restore_calibrated(labels, result, 'target')
    display = session.calibrated_display(camera, result)
    merged = root / 'merged/stack_target.npy'
    signed = root / 'signed/stack_target.npy'
    merged.parent.mkdir(); signed.parent.mkdir()
    np.save(merged, np.concatenate((display, mask[..., None]), axis=-1).astype(np.float32))
    np.save(signed, result.image)
    session.mark_completed('target', output_path=merged, signed_path=signed, mask_role='cell')
    session.finish({'target'})
    np.testing.assert_array_equal(camera, original)
    return dict(sample=sample, root=root, raw=raw, plan=plan)


@pytest.mark.skipif(SUPPLIED_MATRIX_ACCESS, reason='this host admits a supplied matrix')
def test_supplied_matrix_refuses_a_host_without_descriptor_relative_access(supplied_matrix, monkeypatch):
    sample = supplied_matrix
    before = sample['path'].read_bytes()

    def unexpected_model(*args, **kwargs):
        pytest.fail('a refused host reached model construction')

    monkeypatch.setattr('cellpose.models.CellposeModel', unexpected_model)
    with pytest.raises(ValueError, match='supplied spectral matrix'):
        _run_matrix(sample)
    assert sample['path'].read_bytes() == before
    assert not tuple(sample['plate']['source'].glob('calibrated_segmentation*'))
    _source_unchanged(sample)


@needs_supplied_matrix_access
def test_supplied_matrix_session_copies_exact_bytes_without_external_file(completed_supplied_session):
    sample = completed_supplied_session['sample']
    root = completed_supplied_session['root']
    raw = completed_supplied_session['raw']
    assert not sample['path'].exists()
    assert (root / 'psf/spectral_matrix_input.json').read_bytes() == raw
    assert not (sample['plate']['source'] / 'merged').exists()
    record = json.loads((root / 'psf/segmentation_application.json').read_text())
    assert record['complete'] is True and record['fields'] == ['target']
    unmixing = record['configuration']['unmixing']
    assert unmixing['method'] == 'supplied_matrix'
    assert unmixing['controls'] == unmixing['fields'] == {}
    assert 'control_selection' not in unmixing
    assert unmixing['matrix'] == sample['payload']['matrix']
    source = unmixing['matrix_source']
    assert source == dict(kind='supplied_matrix', version=1,
        path='psf/spectral_matrix_input.json', sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw), artifact=sample['payload'], acquisition_compatibility_verified=False)
    with PP._open_calibrated_cell_dataset(root) as dataset:
        assert dataset.physical_channel_order == (3, 1)
        field = next(dataset.fields())
        assert field['source_shape'] == (16, 18, 2)
        assert field['source_origin_px'] == (0, 0)
        assert field['source_identity']['plate'] == 'plate1'
        assert set(np.unique(field['labels'])) == {0, 7}
        np.testing.assert_allclose(field['signed'], sample['sample']['truth'], atol=1e-5, rtol=0)
        assert field['signed'][4, 4, 0] == pytest.approx(-2., abs=1e-5)
    packed = np.load(root / 'merged/stack_target.npy')
    assert packed.shape == (16, 18, 3) and packed.dtype == np.float32
    assert np.all(packed[..., :2] >= 0)
    _source_unchanged(sample)


@needs_supplied_matrix_access
@pytest.mark.parametrize('defect', [
    'channel-order', 'bool-channel', 'orientation', 'plate', 'time-type',
    'background', 'calibration', 'compatibility', 'condition', 'diagonal',
    'negative', 'duplicate', 'nan', 'extra', 'oversize', 'mixed-source',
])
def test_supplied_matrix_refuses_bad_artifacts_before_outputs_or_model(
        supplied_matrix, monkeypatch, defect):
    sample = supplied_matrix
    payload = copy.deepcopy(sample['payload'])
    if defect == 'channel-order':
        payload['channels'].reverse()
    elif defect == 'bool-channel':
        payload['channels'][1] = True
    elif defect == 'orientation':
        payload['orientation'] = 'rows_dyes_columns_readings'
    elif defect == 'plate':
        payload['source_context']['plate'] = 'plate2'
    elif defect == 'time-type':
        payload['source_context']['time'] = '2'
    elif defect == 'background':
        payload['background_percentile'] = 5
    elif defect == 'calibration':
        payload['calibration_source_sha256'] = '0' * 64
    elif defect == 'compatibility':
        payload['acquisition_compatibility'] = 'verified'
    elif defect == 'condition':
        payload['matrix'] = [[1., 1.], [1., 1.]]
    elif defect == 'diagonal':
        payload['matrix'][0][0] = 2
    elif defect == 'negative':
        payload['matrix'][0][1] = -.1
    elif defect == 'extra':
        payload['gain_metadata_matches'] = True
    elif defect == 'mixed-source':
        for path, raw in sample['sample']['plates'][1]['before'].items():
            target = sample['plate']['source'] / path.name
            target.write_bytes(raw)
            sample['plate']['before'][target] = raw
    text = json.dumps(payload)
    if defect == 'duplicate':
        text = text.replace('"version": 1', '"version": 1, "version": 1')
    elif defect == 'nan':
        text = text.replace('"background_percentile": 0.0', '"background_percentile": NaN')
    elif defect == 'oversize':
        text += ' ' * (1024 ** 2 + 1)
    sample['path'].write_text(text)
    before = sample['path'].read_bytes()

    def unexpected_model(*args, **kwargs):
        pytest.fail('invalid matrix reached model construction')

    monkeypatch.setattr('cellpose.models.CellposeModel', unexpected_model)
    with pytest.raises(ValueError, match='supplied spectral matrix'):
        _run_matrix(sample)
    assert sample['path'].read_bytes() == before
    assert not tuple(sample['plate']['source'].glob('calibrated_segmentation*'))
    _source_unchanged(sample)


@pytest.mark.posix_only
def test_supplied_matrix_nonregular_and_unsupported_access_refuse_before_outputs(
        supplied_matrix, monkeypatch):
    sample = supplied_matrix
    sample['path'].unlink()
    os.mkfifo(sample['path'])
    with pytest.raises(ValueError, match='supplied spectral matrix'):
        _run_matrix(sample)
    assert not tuple(sample['plate']['source'].glob('calibrated_segmentation*'))
    sample['path'].unlink()
    sample['path'].write_text(json.dumps(sample['payload']))
    with monkeypatch.context() as patch:
        patch.setattr(PP.os, 'supports_dir_fd', set())
        with pytest.raises(ValueError, match='supplied spectral matrix'):
            _run_matrix(sample)
    assert not tuple(sample['plate']['source'].glob('calibrated_segmentation*'))
    _source_unchanged(sample)


@needs_supplied_matrix_access
def test_supplied_matrix_reader_refuses_rebound_evidence_and_changed_bundle(
        completed_supplied_session):
    sample = completed_supplied_session['sample']
    root = completed_supplied_session['root']
    record_path = root / 'psf/segmentation_application.json'
    original = json.loads(record_path.read_text())
    bundle = root / 'psf/spectral_matrix_input.json'
    raw = bundle.read_bytes()
    for defect in ('missing-source', 'matrix', 'verified', 'context', 'bundle'):
        record = copy.deepcopy(original)
        unmixing = record['configuration']['unmixing']
        if defect == 'missing-source':
            del unmixing['matrix_source']
        elif defect == 'matrix':
            unmixing['matrix'][0][1] += .1
        elif defect == 'verified':
            unmixing['matrix_source']['acquisition_compatibility_verified'] = True
        elif defect == 'context':
            artifact = unmixing['matrix_source']['artifact']
            artifact['source_context']['plate'] = 'plate2'
            changed = json.dumps(artifact).encode()
            bundle.write_bytes(changed)
            unmixing['matrix_source']['sha256'] = hashlib.sha256(changed).hexdigest()
            unmixing['matrix_source']['size_bytes'] = len(changed)
        else:
            bundle.write_bytes(raw + b' ')
        _rewrite_calibrated_cell_record(record_path, record)
        with pytest.raises(ValueError, match='Signed cell measurements require'):
            with PP._open_calibrated_cell_dataset(root):
                pytest.fail('inconsistent supplied matrix reached a consumer')
        bundle.write_bytes(raw)
    _rewrite_calibrated_cell_record(record_path, original)
    with pytest.raises(ValueError, match='Signed cell measurements require'):
        with PP._open_calibrated_cell_dataset(root) as dataset:
            field = next(dataset.fields())
            assert isinstance(field['signed'], np.memmap)
            bundle.write_bytes(raw + b' ')
    _source_unchanged(sample)


def test_blank_matrix_path_keeps_legacy_plan_numbers_and_provenance(calibrated_batch_inputs):
    sample = calibrated_batch_inputs
    options = dict(channels=(3, 1), allow_calibrated=True,
        channel_positions={3: 0, 1: 1}, field_identities=sample['identities'])
    baseline = PP._prepare_unmixing(sample['settings'], sample['stack'], **options)
    for value in (None, '', '  '):
        settings = dict(sample['settings'], unmix_matrix_file=value)
        plan = PP._prepare_unmixing(settings, sample['stack'], **options)
        assert plan.provenance() == baseline.provenance()
        assert 'matrix_source' not in plan.provenance()
        np.testing.assert_array_equal(plan.apply_calibrated(sample['target']).image,
                                      baseline.apply_calibrated(sample['target']).image)
