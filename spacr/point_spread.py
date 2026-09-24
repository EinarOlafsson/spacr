"""Calibrated point-spread kernels and reproducible CPU image processing.

Calculated kernels are sampled Gaussian approximations with explicitly supplied
FWHM and pixel/voxel spacing, not estimates of a microscope's optical PSF.
Measured kernels retain their supplied centre pixel as the optical origin.
Processing uses half-sample symmetric boundaries, independently per channel.
Richardson–Lucy uses the transpose of that same boundary operator, including
its sensitivity normalization for asymmetric kernels. More iterations may
amplify noise; this is not a claim of recovered biological structure.

The original image is never modified. Results are float32 in the input's
intensity units, with no percentile stretch, integer rounding or clipping to
0..1. Kernels, sampling and settings are included in a JSON-safe receipt.

References: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html
and https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.richardson_lucy
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

__all__ = ['PSF', 'PSFResult', 'ProcessingCancelled', 'measured_psf',
           'gaussian_psf', 'load_psf', 'apply_psf']

MAX_KERNEL_VALUES = 2_000_000
MAX_FILE_BYTES = 64 * 1024 * 1024


class ProcessingCancelled(RuntimeError):
    """A caller cancelled processing; no partial image should be applied."""


def _cancelled(cancel):
    """Raise ProcessingCancelled when a callback or event requests cancellation."""
    if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
        raise ProcessingCancelled('PSF processing cancelled')


def _spacing(values, ndim, name):
    """Validate positive finite physical lengths and expand a scalar across spatial axes."""
    try:
        values = (values,) * ndim if np.isscalar(values) else tuple(values)
        if any(isinstance(v, (bool, np.bool_)) for v in values):
            raise ValueError('Boolean values are not physical lengths')
        values = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain positive finite values in micrometers') from exc
    if len(values) != ndim or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError(f'{name} needs {ndim} positive finite values in spatial axis order')
    return values


def _shape(shape):
    """Validate a bounded 2D or 3D kernel shape with positive odd axis lengths."""
    shape = tuple(shape)
    if len(shape) not in (2, 3) or any(int(n) != n or n < 1 or n % 2 != 1 for n in shape):
        raise ValueError('A PSF must have two or three spatial axes with positive odd lengths')
    if math.prod(shape) > MAX_KERNEL_VALUES:
        raise ValueError(f'PSF exceeds {MAX_KERNEL_VALUES} kernel values; reduce its extent')
    return tuple(int(n) for n in shape)


@dataclass(frozen=True)
class PSF:
    """An immutable, normalized kernel whose bytes identify the actual PSF.

    :param shape: odd spatial dimensions, YX or ZYX.
    :param sampling_um: pixel/voxel spacing in the same axis order, in µm.
    :param values: C-ordered little-endian float32 normalized kernel bytes.
    :param source: ``measured`` or ``gaussian approximation``.
    :param details_json: JSON object holding acquisition/file or calculation
        provenance. Use :func:`measured_psf`, :func:`gaussian_psf` or
        :func:`load_psf` to construct a kernel from ordinary arrays/files.
    """

    shape: tuple
    sampling_um: tuple
    values: bytes
    source: str
    details_json: str = '{}'

    def __post_init__(self):
        """Validate serialized kernel geometry, calibration and immutable float32 values."""
        shape = _shape(self.shape)
        spacing = _spacing(self.sampling_um, len(shape), 'PSF sampling')
        values = bytes(self.values)
        if len(values) != math.prod(shape) * 4:
            raise ValueError('PSF byte count does not match its shape')
        data = np.frombuffer(values, dtype='<f4')
        if not np.isfinite(data).all() or (data < 0).any() or not np.isclose(data.sum(dtype=np.float64), 1, rtol=1e-6):
            raise ValueError('PSF values must be finite, nonnegative and sum to one')
        if self.source not in ('measured', 'gaussian approximation'):
            raise ValueError('PSF source must be measured or gaussian approximation')
        details = json.loads(self.details_json)
        if not isinstance(details, dict):
            raise ValueError('PSF details must be a JSON object')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'sampling_um', spacing)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'details_json', json.dumps(details, sort_keys=True, allow_nan=False))

    def array(self):
        """Return a read-only float32 view of the normalized kernel."""
        return np.frombuffer(self.values, dtype='<f4').reshape(self.shape)

    def provenance(self):
        """Return a fresh JSON-safe record of the kernel and its identity."""
        identity = json.dumps([self.shape, self.sampling_um], separators=(',', ':')).encode()
        return {'source': self.source, 'shape': list(self.shape),
                'sampling_um': list(self.sampling_um),
                'kernel_sha256': sha256(identity + self.values).hexdigest(),
                'kernel_dtype': 'float32 little-endian', 'normalization': 'sum=1',
                'origin': 'centre pixel', 'details': json.loads(self.details_json)}


def _kernel(data, sampling_um, source, details):
    """Capture a nonnegative kernel with unit sum and reproducible calibration provenance."""
    data = np.asarray(data)
    shape = _shape(data.shape)
    if data.dtype.kind not in 'uif':
        raise ValueError('PSF values must be real numbers')
    data = np.array(data, dtype=np.float64, copy=True)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError('PSF values must be finite and nonnegative')
    peak = float(data.max())
    if peak <= 0:
        raise ValueError('PSF must contain positive signal')
    data /= peak
    data /= data.sum()
    return PSF(shape, _spacing(sampling_um, data.ndim, 'PSF sampling'),
               data.astype('<f4').tobytes(), source,
               json.dumps(details, sort_keys=True, allow_nan=False))


def measured_psf(data, *, sampling_um, source_name='array'):
    """Normalize a measured 2-D/3-D kernel, preserving its centre as origin.

    :param data: real nonnegative kernel with positive signal and odd axes.
        Background must already be removed; negative values are rejected.
    :param sampling_um: calibrated YX or ZYX spacing in µm, or one isotropic
        spacing. Sampling must match the image when processing it.
    :param source_name: acquisition identifier retained in provenance.
    :returns: immutable normalized :class:`PSF`; input data stay unchanged.
    """
    return _kernel(data, sampling_um, 'measured', {'source_name': str(source_name)})


def gaussian_psf(*, fwhm_um, sampling_um, ndim=2, truncate=4.0):
    """Calculate a sampled Gaussian approximation from declared physical widths.

    :param fwhm_um: full width at half maximum per spatial axis, or one value.
    :param sampling_um: pixel/voxel spacing per axis in µm, or one value.
    :param ndim: two (YX) or three (ZYX) spatial dimensions.
    :param truncate: radius in standard deviations, between two and eight.
    :returns: normalized :class:`PSF`. Sigma is FWHM/sqrt(8*ln(2)); radius is
        ceil(truncate*sigma/sampling). No optical parameters are inferred.
    """
    if ndim not in (2, 3):
        raise ValueError('Gaussian PSFs support two or three spatial axes')
    spacing = _spacing(sampling_um, ndim, 'Image sampling')
    widths = _spacing(fwhm_um, ndim, 'FWHM')
    truncate = float(truncate)
    if not math.isfinite(truncate) or not 2 <= truncate <= 8:
        raise ValueError('Gaussian truncation must be between two and eight sigma')
    sigma = tuple(w / math.sqrt(8 * math.log(2)) / s for w, s in zip(widths, spacing))
    if not all(math.isfinite(s) and 0 < s <= MAX_KERNEL_VALUES / (2 * truncate) for s in sigma):
        raise ValueError('Gaussian extent is too large; check FWHM and sampling units')
    shape = _shape(tuple(2 * max(1, math.ceil(truncate * s)) + 1 for s in sigma))
    data = np.ones(shape, np.float64)
    for axis, (length, scale) in enumerate(zip(shape, sigma)):
        coord = np.arange(length, dtype=np.float64) - length // 2
        weights = np.exp(-0.5 * (coord / scale) ** 2)
        broadcast = [1] * ndim
        broadcast[axis] = length
        data *= weights.reshape(broadcast)
    return _kernel(data, spacing, 'gaussian approximation',
                   {'fwhm_um': list(widths), 'truncate_sigma': truncate,
                    'formula': 'sigma=FWHM/sqrt(8*ln(2))', 'optical_model': False})


def load_psf(path, *, sampling_um, cancel=None):
    """Read a calibrated measured kernel from NPY or a single TIFF series.

    :param path: .npy, .tif or .tiff file, at most 64 MiB. Pickled/object
        arrays, RGB/channel axes, multiple TIFF series and even axes are
        rejected. TIFF YX/ZYX (or a plane stack) is supported.
    :param sampling_um: explicitly supplied measured kernel spacing in µm;
        TIFF metadata is not silently assumed to describe the target image.
    :param cancel: optional callable or threading.Event checked around reads.
    :returns: immutable :class:`PSF`, including SHA-256 of the exact file bytes
        decoded. Subsequent edits to the source file do not change this PSF.
    """
    _cancelled(cancel)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ('.npy', '.tif', '.tiff'):
        raise ValueError('Measured PSF files must be NPY or TIFF')
    with path.open('rb') as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('Measured PSF file exceeds 64 MiB')
    _cancelled(cancel)
    if suffix == '.npy':
        header = BytesIO(raw)
        version = np.lib.format.read_magic(header)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(header)
        elif version in ((2, 0), (3, 0)):
            shape, _, dtype = np.lib.format.read_array_header_2_0(header)
        else:
            raise ValueError('Unsupported NPY kernel format')
        _shape(shape)
        if dtype.kind not in 'uif':
            raise ValueError('PSF values must be real numbers')
        data = np.load(BytesIO(raw), allow_pickle=False)
    else:
        import tifffile
        with tifffile.TiffFile(BytesIO(raw)) as tif:
            if len(tif.series) != 1:
                raise ValueError('Measured PSF TIFF must have exactly one spatial series')
            series = tif.series[0]
            _shape(series.shape)
            if any(axis in series.axes for axis in 'CST'):
                raise ValueError('Measured PSF TIFF must have spatial axes only; export one channel/time point')
            data = series.asarray()
    _cancelled(cancel)
    return _kernel(data, sampling_um, 'measured',
                   {'path': str(path), 'file_sha256': sha256(raw).hexdigest()})


class PSFResult(NamedTuple):
    """Processed float32 image and a JSON-safe processing receipt."""

    image: np.ndarray
    provenance: dict


class _ReflectOperator:
    """Convolution after symmetric extension, with its exact transpose."""

    def __init__(self, shape, kernel):
        """Prepare symmetric boundary padding and index maps for a fixed image shape."""
        self.kernel = kernel
        self.shape = shape
        self.pad = tuple((n // 2, n // 2) for n in kernel.shape)
        self.maps = [np.pad(np.arange(n), p, mode='symmetric')
                     for n, p in zip(shape, self.pad)]

    def forward(self, image):
        """Convolve a symmetrically extended image and return its original spatial extent."""
        from scipy.signal import fftconvolve
        return fftconvolve(np.pad(image, self.pad, mode='symmetric'),
                           self.kernel, mode='valid')

    def adjoint(self, image):
        """Apply the transpose convolution, folding boundary contributions back onto image pixels."""
        from scipy.signal import fftconvolve
        out = fftconvolve(image, np.flip(self.kernel), mode='full')
        for axis, (length, indices) in enumerate(zip(self.shape, self.maps)):
            moved = np.moveaxis(out, axis, 0)
            folded = np.zeros((length, *moved.shape[1:]), dtype=out.dtype)
            np.add.at(folded, indices, moved)
            out = np.moveaxis(folded, 0, axis)
        return out


def apply_psf(image, kernel, *, operation, image_sampling_um,
              iterations=20, channel_axis=None, cancel=None,
              progress: Callable | None = None):
    """Convolve or Richardson–Lucy deconvolve each channel with a known PSF.

    :param image: finite nonnegative real YX/ZYX data, optionally with one
        explicitly identified channel axis. Input pixels are never changed.
    :param kernel: calibrated immutable :class:`PSF`.
    :param operation: ``convolve`` (blur) or ``deconvolve`` (Richardson–Lucy).
    :param image_sampling_um: image spacing matching kernel spacing in spatial
        axis order. Mismatches raise; no implicit kernel resampling is done.
    :param iterations: deconvolution iterations, integer 1..200, default20.
    :param channel_axis: None for a spatial image, otherwise the channel axis;
        channels are processed independently and returned in their original order.
    :param cancel: callable or Event; checked between channels, convolutions
        and iterations. Cancellation raises :class:`ProcessingCancelled`.
    :param progress: optional worker-thread callback(channel, completed, total),
        with zero-based channel and one-based completed iteration.
    :returns: :class:`PSFResult`, float32 image in original intensity units and
        provenance. Half-sample symmetric boundaries do not wrap opposite edges.
        Richardson–Lucy assumes nonnegative Poisson-like intensities; it is
        unregularized and can amplify noise. Output is not clipped to the input
        range. Dimensionality/channel identity are preserved.
    """
    _cancelled(cancel)
    if not isinstance(kernel, PSF):
        raise TypeError('kernel must be a calibrated PSF')
    if operation not in ('convolve', 'deconvolve'):
        raise ValueError('PSF operation must be convolve or deconvolve')
    spacing = _spacing(image_sampling_um, len(kernel.shape), 'Image sampling')
    if not np.allclose(spacing, kernel.sampling_um, rtol=1e-6, atol=0):
        raise ValueError('Image and PSF sampling differ; supply a kernel sampled at the image spacing')
    original = np.asarray(image)
    if original.dtype.kind not in 'uif' or not original.size:
        raise ValueError('Image must contain real nonnegative pixel values')
    if channel_axis is not None:
        try:
            valid_axis = not isinstance(channel_axis, (bool, np.bool_)) and int(channel_axis) == channel_axis and -original.ndim <= channel_axis < original.ndim
        except (ValueError, OverflowError, TypeError):
            valid_axis = False
        if not valid_axis:
            raise ValueError('Channel axis is outside the image dimensions')
        channel_axis = int(channel_axis) % original.ndim
        channels = np.moveaxis(original, channel_axis, 0)
    else:
        channels = original[None]
    if channels.ndim - 1 != len(kernel.shape):
        raise ValueError('Image spatial dimensions must match PSF dimensions; specify channel_axis for multichannel data')
    if not np.isfinite(original).all() or (original < 0).any() or np.max(original) > np.finfo(np.float32).max:
        raise ValueError('Image values must be finite, nonnegative and representable as float32')
    if operation == 'deconvolve':
        try:
            valid_iterations = not isinstance(iterations, (bool, np.bool_)) and int(iterations) == iterations and 1 <= iterations <= 200
        except (ValueError, OverflowError, TypeError):
            valid_iterations = False
        if not valid_iterations:
            raise ValueError('Richardson–Lucy iterations must be an integer from 1 to 200')
        iterations = int(iterations)
    operator = _ReflectOperator(channels.shape[1:], kernel.array())
    result = np.empty(channels.shape, dtype=np.float32)
    sensitivity = None
    for index, channel in enumerate(channels):
        _cancelled(cancel)
        data = np.array(channel, dtype=np.float32, copy=True)
        scale = float(data.max())
        data /= scale or 1.0
        if operation == 'convolve':
            out = np.maximum(operator.forward(data), 0)
            _cancelled(cancel)
            if progress:
                progress(index, 1, 1)
        else:
            if sensitivity is None:
                sensitivity = np.maximum(operator.adjoint(np.ones_like(data)), 1e-7)
            out = np.full_like(data, float(data.mean()))
            for iteration in range(iterations):
                _cancelled(cancel)
                prediction = np.maximum(operator.forward(out), 1e-7)
                _cancelled(cancel)
                out *= np.maximum(operator.adjoint(data / prediction), 0) / sensitivity
                if not np.isfinite(out).all():
                    raise ValueError('Richardson–Lucy produced nonfinite values; check the image and PSF')
                if progress:
                    progress(index, iteration + 1, iterations)
        result[index] = out * scale
    _cancelled(cancel)
    if not np.isfinite(result).all():
        raise ValueError('Processed intensities exceed float32 range')
    output = result[0] if channel_axis is None else np.moveaxis(result, 0, channel_axis)
    receipt = {'operation': operation, 'kernel': kernel.provenance(),
               'image_sampling_um': list(spacing), 'channel_axis': channel_axis,
               'input_dtype': str(original.dtype), 'output_dtype': 'float32',
               'input_shape': list(original.shape), 'boundary': 'half-sample symmetric',
               'intensity_units': 'unchanged', 'input_modified': False,
               'algorithm': 'spacr.reflect_psf.v1'}
    if operation == 'deconvolve':
        receipt.update(iterations=iterations, method='Richardson-Lucy', regularization=None,
                       epsilon_relative=1e-7, initialization='channel mean',
                       sensitivity_normalization=True)
    return PSFResult(output, receipt)
