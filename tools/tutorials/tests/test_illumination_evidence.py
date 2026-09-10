"""The tutorial's numeric acceptance must reject the defect it names."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from illumination_evidence import corrected_reference, verify_corrected, verify_plane, inspect_model, require_preserved, digest, qc_reference


@pytest.mark.parametrize('defect', ['shape', 'dtype', 'nan', 'zero', 'normalisation'])
def test_saved_plane_guard(defect):
    field = np.array([[[.5, 1.5], [.5, 1.5]]], dtype=np.float32)
    assert verify_plane(field, channels=[0], shape=(2, 2))['pixels_checked'] == 4
    changed = field.copy()
    if defect == 'shape':
        changed = changed[:, :1]
    elif defect == 'dtype':
        changed = changed.astype(np.float64)
    elif defect == 'nan':
        changed[0, 0, 0] = np.nan
    elif defect == 'zero':
        changed[0, 0, 0] = 0
    else:
        changed *= 2
    message = ('shape or dtype' if defect in ('shape', 'dtype') else
               'finite and positive' if defect in ('nan', 'zero') else 'mean-one')
    with pytest.raises(ValueError, match=message):
        verify_plane(changed, channels=[0], shape=(2, 2))


def test_integer_formula_rounds_clips_and_preserves_input():
    field = np.array([[[.5, 1.5], [.5, 1.5]]], dtype=np.float32)
    image = np.array([[[40000], [5]], [[1], [4]]], dtype=np.uint16)
    original = image.copy()
    result = corrected_reference(image, field)
    assert result.tolist() == [[[65535], [3]], [[2], [3]]]
    assert np.array_equal(image, original)
    assert verify_corrected(result, result.copy()) == dict(pixels_checked=4, unequal_pixels=0)


@pytest.mark.parametrize('defect', ['shape', 'dtype', 'pixel'])
def test_corrected_values_are_actually_checked(defect):
    expected = np.array([[[7], [8]], [[9], [10]]], dtype=np.uint16)
    assert verify_corrected(expected, expected.copy())['unequal_pixels'] == 0
    changed = expected.copy()
    if defect == 'shape':
        changed = changed[:1]
    elif defect == 'dtype':
        changed = changed.astype(np.float32)
    else:
        changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match='shape or dtype' if defect != 'pixel' else 'pixels differ'):
        verify_corrected(changed, expected)


def test_reference_rejects_a_different_sensor_geometry():
    field = np.ones((1, 2, 2), dtype=np.float32)
    assert corrected_reference(np.ones((2, 2, 1), dtype=np.uint16), field).shape == (2, 2, 1)
    with pytest.raises(ValueError, match='reference geometry'):
        corrected_reference(np.ones((3, 2, 1), dtype=np.uint16), field)


@pytest.mark.parametrize('defect', ['identity', 'field provenance', 'model provenance'])
def test_saved_model_metadata_is_checked(tmp_path, defect):
    import copy
    import json
    record = dict(plate='plate1', key='plate1', channels=[0], dark=[0.0],
                  n_fields=16, estimator='polynomial', degree=4)
    meta = dict(src=[str(tmp_path.resolve())], channels=[0], per_plate=True,
                estimator='polynomial', degree=4, max_fields=16, dark=0.0,
                application_contract_version=1, channel_index_space='persisted-intensity-axis',
                estimated_from_intensity_state='raw')
    manifest = dict(format=1, index={'field0': record}, meta=meta)
    path = tmp_path / 'model.npz'

    def save(value):
        np.savez(path, manifest=np.asarray(json.dumps(value)),
                 field0=np.ones((1, 2, 2), dtype=np.float32))

    save(manifest)
    plane, proof = inspect_model(path, tmp_path, shape=(2, 2))
    assert plane.shape == (1, 2, 2) and proof['pixels_checked'] == 4
    changed = copy.deepcopy(manifest)
    if defect == 'identity':
        changed['format'] = 2
    elif defect == 'field provenance':
        changed['index']['field0']['n_fields'] = 15
    else:
        changed['meta']['estimated_from_intensity_state'] = 'already corrected'
    save(changed)
    with pytest.raises(ValueError, match={'identity':'model identity',
                           'field provenance':'field provenance',
                           'model provenance':'model provenance'}[defect]):
        inspect_model(path, tmp_path, shape=(2, 2))


@pytest.mark.parametrize('which', ['source', 'copy'])
def test_input_preservation_checks_both_real_files(tmp_path, which):
    source = tmp_path / 'original.npy'
    private = tmp_path / 'private.npy'
    np.save(source, np.arange(4, dtype=np.uint16))
    np.save(private, np.arange(4, dtype=np.uint16))
    record = dict(source=str(source), copy=str(private), sha256=digest(source))
    assert require_preserved([record]) == 2
    np.save(Path(record[which]), np.arange(4, dtype=np.uint16) + 1)
    with pytest.raises(ValueError, match='input field changed'):
        require_preserved([record])


def test_qc_radial_slope_has_an_analytic_positive_counterpart():
    coordinate = np.linspace(-1, 1, 8)
    radius = np.sqrt(coordinate[:, None] ** 2 + coordinate[None, :] ** 2) / np.sqrt(2)
    image = (100 * (1 + .4 * radius)).astype(np.float32)
    field = (image / image.mean())[None].astype(np.float32)
    result = qc_reference([image, image * 2], field, 1)
    assert result['slope_before'] == pytest.approx(.4 / (1 + .4 * radius.mean()), abs=1e-6)
    assert abs(result['slope_after']) < 1e-6
    assert result['bias_removed_pct'] == pytest.approx(100, abs=.001)
    assert result['n_fields'] == 2
    # With no correction field, the original radial dependence must survive.
    unchanged = qc_reference([image], np.ones((1, 8, 8), dtype=np.float32), 1)
    assert unchanged['slope_after'] == pytest.approx(unchanged['slope_before'])
    assert unchanged['bias_removed_pct'] == pytest.approx(0)
    with pytest.raises(ValueError, match='positive finite field medians'):
        qc_reference([np.zeros((8, 8))], field, 1)
