"""PSF calibration, operator mathematics and immutable image provenance."""
import io
import json
import threading
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import convolve

from spacr.point_spread import (
    PSF, ProcessingCancelled, _ReflectOperator, apply_psf, gaussian_psf,
    load_psf, measured_psf,
)


def test_gaussian_calibration_matches_physical_second_moments():
    kernel = gaussian_psf(fwhm_um=(1, 2), sampling_um=(.1, .2))
    data = kernel.array()
    yy, xx = np.meshgrid((np.arange(data.shape[0]) - data.shape[0]//2)*.1,
                         (np.arange(data.shape[1]) - data.shape[1]//2)*.2, indexing='ij')
    assert data.sum() == pytest.approx(1)
    assert np.sqrt(np.sum(data*yy**2)) == pytest.approx(1/np.sqrt(8*np.log(2)), rel=.001)
    assert np.sqrt(np.sum(data*xx**2)) == pytest.approx(2/np.sqrt(8*np.log(2)), rel=.001)
    assert not data.flags.writeable
    assert kernel.provenance()['source'] == 'gaussian approximation'
    assert kernel.provenance()['details']['optical_model'] is False
    assert hash(kernel)


@pytest.mark.parametrize('data', [np.zeros((3, 3)), np.ones((2, 3)),
                                    np.full((3, 3), -1), np.full((3, 3), np.nan),
                                    np.full((3, 3), np.inf), np.ones((3, 3), complex),
                                    np.ones((3,)), np.ones((3, 3, 3, 3))])
def test_invalid_measured_kernels_fail(data):
    with pytest.raises(ValueError):
        measured_psf(data, sampling_um=.1)


@pytest.mark.parametrize('spacing', [0, -1, np.nan, np.inf, (.1,), (.1, .2, .3)])
def test_invalid_kernel_sampling_fails(spacing):
    with pytest.raises(ValueError):
        measured_psf(np.ones((3, 3)), sampling_um=spacing)


def test_measured_kernel_normalizes_without_overflow_or_mutation():
    data = np.full((3, 3), 1e308)
    data[0, 0] = 0
    before = data.copy()
    kernel = measured_psf(data, sampling_um=(.2, .2))
    np.testing.assert_array_equal(data, before)
    assert np.isfinite(kernel.array()).all()
    assert kernel.array()[0, 0] == 0
    assert kernel.array().sum() == pytest.approx(1)


@pytest.mark.parametrize('suffix', ['.npy', '.tif'])
def test_loaded_file_identity_and_kernel_are_fixed_after_source_changes(tmp_path, suffix):
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    path = tmp_path / ('psf' + suffix)
    if suffix == '.npy':
        np.save(path, data)
    else:
        import tifffile
        tifffile.imwrite(path, data, metadata={'axes': 'YX'})
    raw = path.read_bytes()
    kernel = load_psf(path, sampling_um=.1)
    before = kernel.provenance()
    path.write_bytes(b'changed')
    assert before == kernel.provenance()
    from hashlib import sha256
    assert before['details']['file_sha256'] == sha256(raw).hexdigest()
    np.testing.assert_allclose(kernel.array(), data/data.sum())
    assert json.loads(json.dumps(before)) == before


def test_rgb_tiff_is_not_silently_treated_as_a_3d_psf(tmp_path):
    import tifffile
    path = tmp_path/'rgb.tif'
    tifffile.imwrite(path, np.ones((3, 3, 3), np.uint8), photometric='rgb')
    with pytest.raises(ValueError, match='spatial axes only'):
        load_psf(path, sampling_um=(.1, .1, .1))


def test_npy_dimensions_are_checked_before_array_allocation(tmp_path, monkeypatch):
    path = tmp_path/'huge.npy'
    with path.open('wb') as stream:
        np.lib.format.write_array_header_1_0(stream, {
            'descr': '<f4', 'fortran_order': False, 'shape': (1000001, 1000001)})
    def unexpected(*args, **kwargs):
        raise AssertionError('attempted allocation before validating dimensions')
    monkeypatch.setattr(np, 'load', unexpected)
    with pytest.raises(ValueError, match='kernel values'):
        load_psf(path, sampling_um=.1)


@pytest.mark.parametrize('ndim', [2, 3])
def test_asymmetric_reflect_convolution_and_adjoint_match_independent_definition(ndim):
    rng = np.random.default_rng(47)
    shape = (5, 6) if ndim == 2 else (4, 5, 6)
    kernel = measured_psf(rng.random((3,)*ndim), sampling_um=.1)
    operator = _ReflectOperator(shape, kernel.array())
    x, y = rng.random(shape).astype(np.float32), rng.random(shape).astype(np.float32)
    np.testing.assert_allclose(operator.forward(x), convolve(x, kernel.array(), mode='reflect'), atol=2e-7)
    assert np.sum(operator.forward(x)*y) == pytest.approx(np.sum(x*operator.adjoint(y)), rel=1e-6)
    np.testing.assert_allclose(operator.forward(np.ones(shape, np.float32)), 1, atol=3e-7)


@pytest.mark.parametrize('operation', ['convolve', 'deconvolve'])
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_channels_source_units_and_shape_survive_processing(operation, channel_axis):
    image = np.stack([np.zeros((9, 11)), np.full((9, 11), 40000), np.full((9, 11), 123)], axis=channel_axis).astype(np.uint16)
    before = image.copy()
    kernel = gaussian_psf(fwhm_um=.3, sampling_um=.1)
    result = apply_psf(image, kernel, operation=operation, image_sampling_um=.1,
                       iterations=5, channel_axis=channel_axis)
    assert result.image.dtype == np.float32
    assert result.image.shape == image.shape
    np.testing.assert_array_equal(image, before)
    np.testing.assert_allclose(result.image, image, rtol=2e-6, atol=1e-4)
    assert result.provenance['input_dtype'] == 'uint16'
    assert json.loads(json.dumps(result.provenance)) == result.provenance


def test_sampling_mismatch_cannot_silently_change_physical_blur_width():
    kernel = gaussian_psf(fwhm_um=.4, sampling_um=.1)
    with pytest.raises(ValueError, match='sampling differ'):
        apply_psf(np.ones((10, 10)), kernel, operation='convolve', image_sampling_um=.2)


def test_known_blur_is_restored_with_lower_error_and_preserved_signal():
    image = np.zeros((65, 65), np.float32)
    image[12:22, 14:24] = 20
    image[40:48, 38:50] = 40
    kernel = gaussian_psf(fwhm_um=.6, sampling_um=.1)
    blurred = apply_psf(image, kernel, operation='convolve', image_sampling_um=.1).image
    restored = apply_psf(blurred, kernel, operation='deconvolve', image_sampling_um=.1, iterations=40).image
    assert np.mean((restored-image)**2) < .5*np.mean((blurred-image)**2)
    assert restored.sum() == pytest.approx(image.sum(), rel=.001)
    assert restored.max() > 1, 'physical intensities were clipped to 0..1'
    assert restored.min() >= 0


def test_richardson_lucy_matches_an_independently_formed_dense_operator():
    rng = np.random.default_rng(77)
    kernel = measured_psf(rng.random((3, 3)), sampling_um=.1)
    shape = (5, 7)
    basis = np.eye(np.prod(shape), dtype=np.float64)
    matrix = np.column_stack([convolve(row.reshape(shape), kernel.array(), mode='reflect').ravel()
                              for row in basis])
    data = rng.random(shape)*200
    scaled = data.ravel()/data.max()
    expected = np.full_like(scaled, scaled.mean())
    sensitivity = matrix.T @ np.ones_like(scaled)
    for _ in range(12):
        expected *= (matrix.T @ (scaled/np.maximum(matrix @ expected, 1e-7))) / sensitivity
    result = apply_psf(data, kernel, operation='deconvolve', image_sampling_um=.1, iterations=12)
    np.testing.assert_allclose(result.image, expected.reshape(shape)*data.max(), rtol=5e-6)


def test_asymmetric_rl_improves_the_matching_forward_likelihood_at_edges():
    rng = np.random.default_rng(3)
    kernel = measured_psf(rng.random((3, 5)), sampling_um=.1)
    image = np.zeros((25, 31), np.float32)
    image[:5, :7] = 20
    image[-7:, -3:] = 10
    data = apply_psf(image, kernel, operation='convolve', image_sampling_um=.1).image
    restored = apply_psf(data, kernel, operation='deconvolve', image_sampling_um=.1, iterations=40).image
    restored_prediction = apply_psf(restored, kernel, operation='convolve', image_sampling_um=.1).image
    baseline_prediction = apply_psf(data, kernel, operation='convolve', image_sampling_um=.1).image
    assert np.mean((restored_prediction-data)**2) < .1*np.mean((baseline_prediction-data)**2)


def test_cancellation_between_iterations_returns_no_partial_result():
    cancel = threading.Event()
    calls = []
    kernel = gaussian_psf(fwhm_um=.4, sampling_um=.1)
    def progress(channel, done, total):
        calls.append((channel, done, total))
        if done == 3:
            cancel.set()
    with pytest.raises(ProcessingCancelled):
        apply_psf(np.ones((31, 31)), kernel, operation='deconvolve',
                  image_sampling_um=.1, cancel=cancel, progress=progress)
    assert calls == [(0, 1, 20), (0, 2, 20), (0, 3, 20)]


@pytest.mark.parametrize('image', [np.full((4, 4), np.nan), np.full((4, 4), -1),
                                     np.ones((2, 4, 4)), np.zeros((0, 4)),
                                     np.ones((4, 4), complex)])
def test_invalid_image_fails_instead_of_inventing_a_channel_layout(image):
    kernel = gaussian_psf(fwhm_um=.4, sampling_um=.1)
    with pytest.raises(ValueError):
        apply_psf(image, kernel, operation='convolve', image_sampling_um=.1)


def test_3d_volume_with_anisotropic_sampling():
    kernel = gaussian_psf(fwhm_um=(1, .3, .3), sampling_um=(.5, .1, .1), ndim=3)
    image = np.zeros((7, 15, 15), np.float32)
    image[3, 7, 7] = 500
    result = apply_psf(image, kernel, operation='convolve', image_sampling_um=(.5, .1, .1))
    assert result.image.shape == image.shape
    assert np.unravel_index(result.image.argmax(), image.shape) == (3, 7, 7)
    assert result.image.sum() == pytest.approx(500, rel=.001)


def test_kernel_identity_includes_sampling_and_rejects_invalid_direct_construction():
    kernel = measured_psf(np.ones((3, 3)), sampling_um=.1)
    assert kernel.provenance()['kernel_sha256'] != replace(kernel, sampling_um=(.2, .2)).provenance()['kernel_sha256']
    with pytest.raises(ValueError):
        PSF((3, 3), (.1, .1), b'bad', 'measured')
    with pytest.raises(ValueError):
        replace(kernel, values=np.zeros((3, 3), '<f4').tobytes())


@pytest.mark.parametrize('fwhm', [0, -1, np.nan, np.inf, 1e300])
def test_gaussian_invalid_or_unbounded_extent_is_rejected(fwhm):
    with pytest.raises(ValueError):
        gaussian_psf(fwhm_um=fwhm, sampling_um=.1)


@pytest.mark.parametrize('iterations', [0, 201, 1.5, True])
def test_deconvolution_iteration_range_is_explicit(iterations):
    with pytest.raises(ValueError, match='iterations'):
        apply_psf(np.ones((5, 5)), gaussian_psf(fwhm_um=.3, sampling_um=.1),
                  operation='deconvolve', image_sampling_um=.1, iterations=iterations)


def test_loading_is_cancellable_and_size_bounded(tmp_path, monkeypatch):
    import spacr.point_spread as psf
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ProcessingCancelled):
        load_psf(tmp_path/'missing.tif', sampling_um=.1, cancel=cancel)
    monkeypatch.setattr(psf, 'MAX_FILE_BYTES', 512)
    path = tmp_path/'large.tif'
    path.write_bytes(b'x'*513)
    with pytest.raises(ValueError, match='64 MiB'):
        load_psf(path, sampling_um=.1)


@pytest.mark.parametrize('axis', [True, np.nan, np.inf, 'wrong', 4, -5])
def test_invalid_channel_axis_is_actionable(axis):
    with pytest.raises(ValueError, match='Channel axis'):
        apply_psf(np.ones((5,5)), gaussian_psf(fwhm_um=.3, sampling_um=.1),
                  operation='convolve', image_sampling_um=.1, channel_axis=axis)


@pytest.mark.parametrize('value', [None, np.inf, np.nan, 'many'])
def test_malformed_iteration_values_are_rejected(value):
    with pytest.raises(ValueError, match='iterations'):
        apply_psf(np.ones((5,5)), gaussian_psf(fwhm_um=.3, sampling_um=.1),
                  operation='deconvolve', image_sampling_um=.1, iterations=value)


def test_no_implicit_operation_or_uncalibrated_kernel():
    image = np.ones((5,5))
    with pytest.raises(TypeError, match='calibrated'):
        apply_psf(image, np.ones((3,3)), operation='convolve', image_sampling_um=.1)
    with pytest.raises(ValueError, match='operation'):
        apply_psf(image, gaussian_psf(fwhm_um=.3, sampling_um=.1), operation='magic', image_sampling_um=.1)
    with pytest.raises(ValueError):
        gaussian_psf(fwhm_um=.3, sampling_um=True)


def test_serialized_kernel_metadata_must_be_a_finite_object():
    kernel = measured_psf(np.ones((3,3)), sampling_um=.1)
    with pytest.raises(ValueError):
        replace(kernel, details_json='[]')
    with pytest.raises(ValueError):
        replace(kernel, details_json='{"gain": NaN}')
    with pytest.raises(ValueError):
        replace(kernel, source='unknown')
