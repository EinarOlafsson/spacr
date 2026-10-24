"""A supplied camera matrix carries signed pixels through V2 and cell measurement."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tests.cellpose_api_contract import (
    MISSING_CHANNEL_AXIS, assert_declares_installed_eval_signature,
)
from tests.conftest import check_cellpose_eval_call


def test_supplied_matrix_v2_measures_signed_truth_after_external_matrix_removal(
        tmp_path, monkeypatch):
    """Real camera TIFFs retain signed zero-sum cells and their physical channel IDs."""
    from spacr import accelerator, pipeline_v2
    from spacr.measure import measure_calibrated_cells
    from spacr.tabular import read_table

    observed = []

    class CPUCellposeAdapter:
        def __init__(self, *, pretrained_model, gpu, device, use_bfloat16):
            assert gpu is False and str(device) == 'cpu' and use_bfloat16 is False
            self.pretrained_model = None

        def eval(self, images, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
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
    cpu = accelerator.Accelerator(kind='cpu', device='cpu', label='CPU oracle')
    monkeypatch.setattr(accelerator, 'resolve', lambda: cpu)
    source = tmp_path / 'plate1'
    source.mkdir()
    matrix = np.array([[1., .25], [.5, 1.]], np.float64)
    latent = np.zeros((16, 18, 2), np.float32)
    latent[3:6, 3:6, 1] = 10.
    latent[4, 5, 0] = 2.
    camera_perturbation = np.zeros_like(latent)
    camera_perturbation[4, 4] = -2. * matrix[:, 0]
    camera = (latent @ matrix.T + camera_perturbation).astype(np.float32)
    signed_truth = latent.copy()
    signed_truth[4, 4, 0] = -2.
    assert latent.min() >= 0 and camera.min() >= 0 and signed_truth.min() == -2.
    np.testing.assert_array_equal(np.percentile(camera, 0, axis=(0, 1)), [0., 0.])
    for position, physical_id in enumerate((3, 1)):
        tifffile.imwrite(source / f'plate1_C01_F10_T2_Z1_C{physical_id}.tif',
                         camera[..., position])
    raw_merged = source / 'merged'
    raw_merged.mkdir()
    np.save(raw_merged / 'original_camera.npy', camera)
    original_bytes = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
    calibration = dict(version=1, axes='YX', pixel_size_um=[.4, .3],
        reference_channel=3, channels=[dict(channel=channel, shift_to_reference_px=[0, 0],
        psf=[[1.]], psf_origin_px=[0, 0], psf_pixel_size_um=[.4, .3]) for channel in (3, 1)])
    calibration_path = tmp_path / 'optics.json'
    calibration_path.write_text(json.dumps(calibration), encoding='utf-8')
    calibration_bytes = calibration_path.read_bytes()
    artifact = dict(artifact='spacr_spectral_matrix', version=1,
        source_context=dict(plate='plate1', time=2, z=1), channels=[3, 1],
        orientation='rows_readings_columns_dyes', matrix=matrix.tolist(),
        background_percentile=0., background_operator='per_field_percentile_subtract_solve_restore',
        calibration_source_sha256=hashlib.sha256(calibration_bytes).hexdigest(),
        acquisition_compatibility='user_asserts_same_channel_gains_exposures_and_linear_response')
    matrix_path = tmp_path / 'supplied_matrix.json'
    matrix_path.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')
    matrix_bytes = matrix_path.read_bytes()
    settings = dict(unmix=True, unmix_per_plate=False, unmix_controls='',
        unmix_matrix_file=str(matrix_path), unmix_background_percentile=0.,
        unmix_calibration=str(calibration_path), unmix_pixel_size_um=[.4, .3],
        lower_percentile=0., cell_background=0., cell_signal_to_noise=0.,
        remove_background_cell=False, nucleus_background=0.,
        nucleus_signal_to_noise=0., remove_background_nucleus=False)
    result = pipeline_v2.run_v2(source, channels=(3, 1), channels_for_cellpose=(0, 1),
        channel_names=('cell', 'nucleus'), metadata_type='custom',
        custom_regex=r'^(?P<plateID>plate\d+)_(?P<wellID>[A-Z]\d+)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)_Z(?P<sliceID>\d+)_C(?P<chanID>\d+)\.tif$',
        batch_fields=1, postprocess_settings=settings)
    root = result['output_root']
    assert root.parent == source and root != source
    assert len(observed) == len(result['stacks']) == 1
    assert np.isfinite(observed[0]).all() and np.min(observed[0]) >= 0
    record_path = root / 'psf/segmentation_application.json'
    record_bytes = record_path.read_bytes()
    record = json.loads(record_bytes)
    assert record['complete'] is True and len(record['fields']) == 1
    configuration = record['configuration']
    assert configuration['physical_channel_order'] == [3, 1]
    assert configuration['coordinate_system'] == 'calibrated_reference_canvas'
    unmixing = configuration['unmixing']
    assert unmixing['method'] == 'supplied_matrix'
    assert unmixing['matrix'] == matrix.tolist()
    assert unmixing['controls'] == unmixing['fields'] == {}
    assert 'control_selection' not in unmixing
    assert unmixing['matrix_source'] == dict(kind='supplied_matrix', version=1,
        path='psf/spectral_matrix_input.json', sha256=hashlib.sha256(matrix_bytes).hexdigest(),
        size_bytes=len(matrix_bytes), artifact=artifact, acquisition_compatibility_verified=False)
    assert (root / 'psf/spectral_matrix_input.json').read_bytes() == matrix_bytes
    field_id = result['stacks'][0].field_id
    field = record['calibrated_fields'][field_id]
    assert field['source_identity'] == dict(plate='plate1', well='C01', field=10,
        time=2, z=1, source_dataset=str(source.resolve()))
    assert field['valid_roi_yx'] == [[0, 16], [0, 18]]
    assert field['source_origin_px'] == [0, 0]
    assert field['measurement_schema_version'] == 1 and field['mask_role'] == 'cell'
    signed_path = root / field['signed_crop']['path']
    assert field['signed_crop']['sha256'] == hashlib.sha256(signed_path.read_bytes()).hexdigest()
    np.testing.assert_allclose(np.load(signed_path), signed_truth, atol=1e-6, rtol=0)
    packed = np.load(result['stacks'][0].path)
    assert packed.shape == (16, 18, 3) and packed.dtype == np.float32
    np.testing.assert_allclose(packed[..., :2], np.maximum(signed_truth, 0), atol=1e-6, rtol=0)
    expected_labels = np.zeros((16, 18), np.uint16)
    expected_labels[3:6, 3:6] = 7
    np.testing.assert_array_equal(packed[..., 2], expected_labels)
    matrix_path.unlink()
    measured = measure_calibrated_cells(root, dst=tmp_path / 'signed_cells', channels=[1, 3])
    assert not matrix_path.exists()
    assert measured['fields'] == 1 and measured['rows'] == 2
    rows = read_table(measured['csv_path'])
    assert list(rows['physical_channel_id']) == [1, 3]
    for position, physical_id in ((1, 1), (0, 3)):
        row = rows[rows['physical_channel_id'] == physical_id].iloc[0]
        values = signed_truth[3:6, 3:6, position].astype(np.float64)
        for statistic, expected in (('mean', values.mean()), ('min', values.min()),
                ('max', values.max()), ('sum', values.sum()), ('std', values.std(ddof=0))):
            assert float(row['intensity_' + statistic]) == pytest.approx(expected, abs=1e-6)
        assert int(row['label_id']) == 7 and int(row['area_px']) == 9
        assert float(row['area_um2']) == pytest.approx(1.08)
        assert (float(row['centroid_y_px']), float(row['centroid_x_px'])) == (4., 4.)
        assert float(row['centroid_y_um']) == pytest.approx(1.6)
        assert float(row['centroid_x_um']) == pytest.approx(1.2)
        assert tuple(int(row[key]) for key in ('bbox_min_y_px', 'bbox_min_x_px',
            'bbox_max_y_px', 'bbox_max_x_px')) == (3, 3, 6, 6)
        assert (row['plateID'], row['wellID'], int(row['fieldID']), int(row['timeID']),
                int(row['sliceID'])) == ('plate1', 'C01', 10, 2, 1)
    provenance = json.loads(Path(measured['provenance_path']).read_text())
    assert provenance['source_provenance'] == record
    assert provenance['source_record_sha256'] == hashlib.sha256(record_bytes).hexdigest()
    assert provenance['csv_sha256'] == hashlib.sha256(
        Path(measured['csv_path']).read_bytes()).hexdigest()
    assert provenance['physical_channel_order'] == [1, 3]
    assert provenance['pixel_size_um'] == [.4, .3]
    assert provenance['standard_deviation_ddof'] == 0
    assert all(path.read_bytes() == before for path, before in original_bytes.items())
    assert calibration_path.read_bytes() == calibration_bytes
