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

When the calibration is not known, :func:`infer_optics` supplies it from the
image's own metadata, a chosen objective from :data:`OBJECTIVES` or common
defaults, recording where each value came from: pixel size is the camera
pixel divided by the magnification and the lateral FWHM is 0.51 λ/NA.

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
           'gaussian_psf', 'load_psf', 'apply_psf', 'Objective', 'OpticalValue',
           'OBJECTIVES', 'CAMERAS', 'FLUOROPHORES', 'IMMERSION_INDEX', 'objective',
           'pixel_size_um', 'lateral_fwhm_um', 'image_optics_metadata',
           'infer_optics', 'describe_optics', 'fill_psf_settings']

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
    """Processed float32 image and a JSON-safe processing receipt.

    :param image: processed image with the input's spatial shape, as float32.
    :param provenance: JSON-safe kernel and processing settings for this result.
    """

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


class Objective(NamedTuple):
    """One microscope objective: nominal magnification, numerical aperture and immersion."""

    name: str
    magnification: float
    numerical_aperture: float
    immersion: str


class OpticalValue(NamedTuple):
    """An inferred optical quantity and where it came from.

    ``source`` is one of ``metadata`` (OME-XML), ``imagej`` (ImageJ TIFF
    calibration), ``tiff_resolution`` (TIFF resolution tags in centimetres),
    ``file_name``, ``image`` (the file's own dimensions), ``chosen`` (supplied
    by the caller), ``objective`` (the objective table), ``default`` or
    ``calculated``. ``detail`` names the file, table row or formula.
    """

    value: object
    source: str
    detail: str = ''


IMMERSION_INDEX = {'air': 1.0, 'water': 1.33, 'oil': 1.515}
OBJECTIVES = tuple(Objective(*row) for row in (
    ('10x/0.30 air', 10.0, 0.30, 'air'),
    ('10x/0.45 air', 10.0, 0.45, 'air'),
    ('20x/0.45 air', 20.0, 0.45, 'air'),
    ('20x/0.75 air', 20.0, 0.75, 'air'),
    ('40x/0.95 air', 40.0, 0.95, 'air'),
    ('40x/1.30 oil', 40.0, 1.30, 'oil'),
    ('60x/1.20 water', 60.0, 1.20, 'water'),
    ('60x/1.40 oil', 60.0, 1.40, 'oil'),
    ('63x/1.40 oil', 63.0, 1.40, 'oil'),
    ('100x/1.40 oil', 100.0, 1.40, 'oil'),
    ('100x/1.45 oil', 100.0, 1.45, 'oil'),
))
PREFERRED_OBJECTIVE = {10.0: '10x/0.30 air', 20.0: '20x/0.75 air', 40.0: '40x/0.95 air',
                       60.0: '60x/1.40 oil', 63.0: '63x/1.40 oil', 100.0: '100x/1.40 oil'}
CAMERAS = (('sCMOS, 6.5 µm pixels', 6.5), ('CMOS, 4.54 µm pixels', 4.54),
           ('CMOS, 3.45 µm pixels', 3.45), ('sCMOS, 11 µm pixels', 11.0),
           ('EMCCD, 16 µm pixels', 16.0))
FLUOROPHORES = (('DAPI / Hoechst', 461.0), ('GFP / FITC / Alexa 488', 520.0),
                ('Cy3 / TRITC / Alexa 555', 600.0), ('Cy5 / Alexa 647', 670.0))
DEFAULT_OBJECTIVE = '20x/0.75 air'
DEFAULT_CAMERA_PIXEL_UM = 6.5
DEFAULT_EMISSION_NM = 520.0
AIRY_FWHM_FACTOR = 0.51
_TIFF_SUFFIXES = ('.tif', '.tiff')


def objective(name):
    """Return the :class:`Objective` table row called ``name``.

    :param name: a row name such as ``'60x/1.40 oil'``; case and surrounding
        spaces are ignored.
    :returns: the matching :class:`Objective`.
    :raises ValueError: when the table has no such objective.
    """
    wanted = str(name or '').strip().lower()
    for row in OBJECTIVES:
        if row.name.lower() == wanted:
            return row
    raise ValueError(f'Unknown objective {name!r}; choose one of '
                     + ', '.join(row.name for row in OBJECTIVES))


def pixel_size_um(camera_pixel_um, magnification):
    """Sample spacing at the specimen: camera pixel pitch divided by total magnification.

    :param camera_pixel_um: physical camera pixel pitch in micrometers.
    :param magnification: total magnification between specimen and camera,
        including any tube-lens or camera-adapter factor (binning multiplies
        the camera pixel instead).
    :returns: pixel size in micrometers.
    :raises ValueError: for a nonpositive or nonfinite input.
    """
    camera_pixel_um, magnification = float(camera_pixel_um), float(magnification)
    if not (math.isfinite(camera_pixel_um) and camera_pixel_um > 0
            and math.isfinite(magnification) and magnification > 0):
        raise ValueError('Camera pixel size and magnification must be positive')
    return camera_pixel_um / magnification


def lateral_fwhm_um(emission_nm, numerical_aperture, refractive_index=None):
    """Lateral FWHM of a diffraction-limited widefield PSF, 0.51 λ/NA.

    The intensity full width at half maximum of the Airy pattern is
    ``0.514 λ / NA`` (Born and Wolf, *Principles of Optics*, 7th ed., §8.5.2,
    whose Rayleigh radius is ``0.61 λ / NA``). A Gaussian fitted to the
    widefield PSF has ``σ ≈ 0.21 λ / NA``, a FWHM of ``≈ 0.49 λ / NA`` (Zhang,
    Zerubia and Olivo-Marin 2007, Appl. Opt. 46:1819–1829), so a Gaussian of
    this width approximates the ideal emission PSF. Aberrations, confocal
    pinholes, thick specimens and camera binning are not modelled.

    :param emission_nm: emission wavelength in nanometers.
    :param numerical_aperture: objective NA.
    :param refractive_index: immersion index; when given, NA must not exceed it.
    :returns: FWHM in micrometers.
    :raises ValueError: for an implausible wavelength or aperture.
    """
    emission_nm, numerical_aperture = float(emission_nm), float(numerical_aperture)
    if not (math.isfinite(emission_nm) and 100 <= emission_nm <= 2000):
        raise ValueError('Emission wavelength must be between 100 and 2000 nm')
    if not (math.isfinite(numerical_aperture) and 0 < numerical_aperture <= 1.7):
        raise ValueError('Numerical aperture must be between 0 and 1.7')
    if refractive_index is not None and numerical_aperture > float(refractive_index):
        raise ValueError('Numerical aperture cannot exceed the immersion refractive index')
    return AIRY_FWHM_FACTOR * emission_nm / 1000.0 / numerical_aperture


def _micrometers(value, unit):
    """Convert an OME length to micrometers, or None for an unusable value."""
    scale = {'µm': 1.0, 'um': 1.0, 'micron': 1.0, 'nm': 1e-3, 'mm': 1e3,
             'm': 1e6, 'cm': 1e4, 'Å': 1e-4}.get(str(unit or 'µm').strip())
    try:
        value = float(value) * scale
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0.001 <= value <= 100 else None


def _ome_values(xml, name):
    """Read calibration from OME-XML: pixel sizes, objective and emission."""
    import xml.etree.ElementTree as ElementTree

    found = {}
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return found
    elements = {}
    for element in root.iter():
        elements.setdefault(element.tag.rsplit('}', 1)[-1], element)
    pixels = elements.get('Pixels')
    if pixels is not None:
        sizes = [_micrometers(pixels.get(f'PhysicalSize{axis}'),
                              pixels.get(f'PhysicalSize{axis}Unit')) for axis in 'YX']
        if sizes[1] is not None:
            sizes[0] = sizes[0] or sizes[1]
            found['pixel_size_um'] = OpticalValue(tuple(sizes), 'metadata', name)
    lens = elements.get('Objective')
    if lens is not None:
        for key, attribute in (('numerical_aperture', 'LensNA'),
                               ('magnification', 'CalibratedMagnification'),
                               ('magnification', 'NominalMagnification')):
            try:
                value = float(lens.get(attribute))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0 and key not in found:
                found[key] = OpticalValue(value, 'metadata', name)
        immersion = str(lens.get('Immersion') or '').lower()
        if immersion in IMMERSION_INDEX:
            found['refractive_index'] = OpticalValue(IMMERSION_INDEX[immersion], 'metadata', name)
    objective_settings = elements.get('ObjectiveSettings')
    if objective_settings is not None:
        try:
            index = float(objective_settings.get('RefractiveIndex'))
        except (TypeError, ValueError):
            index = math.nan
        if 1 <= index <= 2:
            found['refractive_index'] = OpticalValue(index, 'metadata', name)
    channel = elements.get('Channel')
    if channel is not None:
        try:
            emission = float(channel.get('EmissionWavelength'))
        except (TypeError, ValueError):
            emission = math.nan
        unit = str(channel.get('EmissionWavelengthUnit') or 'nm')
        emission *= {'nm': 1.0, 'µm': 1000.0, 'um': 1000.0}.get(unit, math.nan)
        if 100 <= emission <= 2000:
            found['emission_nm'] = OpticalValue(emission, 'metadata', name)
    return found


def _resolution_values(tif, tags, name):
    """Pixel size from an ImageJ micron calibration or centimetre resolution tags."""
    x_tag = tags.get('XResolution')
    if x_tag is None:
        return {}
    y_tag = tags.get('YResolution') or x_tag
    try:
        per_unit = [float(tag.value[0]) / float(tag.value[1]) for tag in (y_tag, x_tag)]
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        return {}
    unit, source = None, ''
    if tif.is_imagej and tif.imagej_metadata:
        label = str(tif.imagej_metadata.get('unit', '')).replace('\\u00B5', 'µ')
        if label in ('micron', 'um', 'µm'):
            unit, source = 1.0, 'imagej'
    elif 'ResolutionUnit' in tags and int(tags['ResolutionUnit'].value) == 3:
        unit, source = 1e4, 'tiff_resolution'
    if not unit or not all(v > 0 for v in per_unit):
        return {}
    sizes = tuple(unit / v for v in per_unit)
    if not all(0.001 <= s <= 100 for s in sizes):
        return {}
    return {'pixel_size_um': OpticalValue(sizes, source, name)}


def _tiff_values(path):
    """Read image dimensions and any calibration a TIFF header carries, without pixels."""
    import tifffile

    name = Path(path).name
    found = {}
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(n) for n in series.shape)
        if 'Y' in axes and 'X' in axes:
            found['image_shape'] = OpticalValue(
                (shape[axes.index('Y')], shape[axes.index('X')]), 'image', name)
        elif len(shape) >= 2:
            found['image_shape'] = OpticalValue(shape[-2:], 'image', name)
        if tif.is_ome and tif.ome_metadata:
            found.update(_ome_values(tif.ome_metadata, name))
        if 'pixel_size_um' not in found:
            found.update(_resolution_values(tif, tif.pages[0].tags, name))
    return found


def _first_image(source):
    """The first TIFF named by ``source``: a file, a folder or an iterable of either."""
    if source is None:
        return None
    if not isinstance(source, (str, Path)):
        for item in source:
            found = _first_image(item)
            if found is not None:
                return found
        return None
    path = Path(source)
    if path.is_file():
        return path if path.suffix.lower() in _TIFF_SUFFIXES else None
    for folder in (path, path / 'orig'):
        if folder.is_dir():
            files = sorted(p for p in folder.iterdir()
                           if p.is_file() and p.suffix.lower() in _TIFF_SUFFIXES)
            if files:
                return files[0]
    return None


def image_optics_metadata(path):
    """Calibration an image file states about itself, each value with its source.

    Reads only the TIFF header: OME-XML ``PhysicalSizeX/Y``, ``Objective``
    ``LensNA``/``NominalMagnification``/``Immersion``, ``ObjectiveSettings``
    ``RefractiveIndex`` and ``Channel`` ``EmissionWavelength``; an ImageJ
    calibration in microns; or TIFF resolution tags in centimetres. A bare
    dots-per-inch tag says nothing about the specimen and is ignored. A
    magnification token such as ``_40x_`` in the file name is used when no
    header states one.

    :param path: an image file; formats other than TIFF contribute only the
        file-name token.
    :returns: mapping of field name to :class:`OpticalValue`; empty when the
        file states nothing or cannot be read.
    """
    import re

    path = Path(path)
    found = {}
    if path.suffix.lower() in _TIFF_SUFFIXES:
        try:
            found = _tiff_values(path)
        except Exception:
            found = {}
    if 'magnification' not in found:
        match = re.search(r'(?:^|[_\-\s.])(\d{1,3})[xX](?=$|[_\-\s.])', path.stem)
        if match and 1 < int(match.group(1)) <= 150:
            found['magnification'] = OpticalValue(float(match.group(1)), 'file_name', path.name)
    return found


def _table_objective(stated):
    """The table row closest to a stated magnification and aperture."""
    mag = stated['magnification'].value
    if 'numerical_aperture' in stated:
        na = stated['numerical_aperture'].value
        return min(OBJECTIVES, key=lambda row: (abs(row.magnification - mag),
                                                abs(row.numerical_aperture - na)))
    if mag in PREFERRED_OBJECTIVE:
        return objective(PREFERRED_OBJECTIVE[mag])
    return min(OBJECTIVES, key=lambda row: abs(row.magnification - mag))


def infer_optics(source=None, *, objective_name=None, camera_pixel_um=None,
                 emission_nm=None, magnification=None, numerical_aperture=None,
                 refractive_index=None, image_pixel_um=None):
    """Infer PSF calibration from image metadata, a chosen objective and defaults.

    Every returned value carries its source. Precedence, highest first: a
    value passed here (``chosen``), the image's own metadata (ignored for the
    objective's own magnification, NA and immersion once an objective is
    chosen), the objective table (the chosen row, else the row nearest a
    stated magnification), then the defaults: a 20x/0.75 air objective, a
    6.5 µm sCMOS pixel and 520 nm (GFP) emission. Immersion refractive
    indices are air 1.0, water 1.33 and oil 1.515.

    Formulae: pixel size = camera pixel / magnification (:func:`pixel_size_um`);
    lateral Gaussian FWHM = 0.51 λ / NA (:func:`lateral_fwhm_um`; Born and
    Wolf §8.5.2; Zhang, Zerubia and Olivo-Marin 2007). A pixel size stated
    in metadata wins over the calculated one because it already includes
    binning and adapters.

    :param source: an image file, a folder (its first TIFF, or ``orig/``'s)
        or an iterable of paths; None uses the table and defaults only.
    :param objective_name: an :data:`OBJECTIVES` row name, or None/``'auto'``.
    :param camera_pixel_um: camera pixel pitch in micrometers.
    :param emission_nm: emission wavelength in nanometers.
    :param magnification: total magnification.
    :param numerical_aperture: objective NA.
    :param refractive_index: immersion refractive index.
    :param image_pixel_um: image pixel size in micrometers, one value or (Y, X).
    :returns: dict of :class:`OpticalValue` for ``objective``, ``magnification``,
        ``numerical_aperture``, ``refractive_index``, ``emission_nm``,
        ``camera_pixel_um``, ``pixel_size_um`` (Y, X), ``fwhm_um`` (Y, X) and,
        when an image was read, ``image`` and ``image_shape`` (Y, X).
    :raises ValueError: for an unknown objective or implausible optics.
    """
    image = _first_image(source)
    stated = image_optics_metadata(image) if image is not None else {}
    values = {}
    if image is not None:
        values['image'] = OpticalValue(str(image), 'image', Path(image).name)
    if 'image_shape' in stated:
        values['image_shape'] = stated['image_shape']
    if objective_name and str(objective_name).strip().lower() != 'auto':
        row = objective(objective_name)
        values['objective'] = OpticalValue(row.name, 'chosen', row.name)
    elif 'magnification' in stated:
        row = _table_objective(stated)
        values['objective'] = OpticalValue(row.name, 'objective', 'nearest to the stated magnification')
    else:
        row = objective(DEFAULT_OBJECTIVE)
        values['objective'] = OpticalValue(row.name, 'default', row.name)
    picked = values['objective'].source == 'chosen'
    table = 'default' if values['objective'].source == 'default' else 'objective'

    def pick(key, given, table_value):
        """Resolve one value: the argument, then metadata, then the table row."""
        if given is not None:
            return OpticalValue(float(given), 'chosen', '')
        if key in stated and not picked:
            return stated[key]
        return OpticalValue(float(table_value), table, row.name)

    values['magnification'] = pick('magnification', magnification, row.magnification)
    values['numerical_aperture'] = pick('numerical_aperture', numerical_aperture,
                                        row.numerical_aperture)
    values['refractive_index'] = pick('refractive_index', refractive_index,
                                      IMMERSION_INDEX[row.immersion])
    if emission_nm is not None:
        values['emission_nm'] = OpticalValue(float(emission_nm), 'chosen', '')
    else:
        values['emission_nm'] = stated.get('emission_nm') or OpticalValue(
            DEFAULT_EMISSION_NM, 'default', 'GFP')
    values['camera_pixel_um'] = (
        OpticalValue(float(camera_pixel_um), 'chosen', '') if camera_pixel_um is not None
        else OpticalValue(DEFAULT_CAMERA_PIXEL_UM, 'default', CAMERAS[0][0]))
    if image_pixel_um is not None:
        values['pixel_size_um'] = OpticalValue(
            _spacing(image_pixel_um, 2, 'Image pixel size'), 'chosen', '')
    elif 'pixel_size_um' in stated:
        values['pixel_size_um'] = stated['pixel_size_um']
    else:
        size = pixel_size_um(values['camera_pixel_um'].value, values['magnification'].value)
        values['pixel_size_um'] = OpticalValue(
            (size, size), 'calculated', 'camera pixel / magnification')
    fwhm = lateral_fwhm_um(values['emission_nm'].value, values['numerical_aperture'].value,
                           values['refractive_index'].value)
    values['fwhm_um'] = OpticalValue((fwhm, fwhm), 'calculated', '0.51 × emission / NA')
    return values


def describe_optics(values):
    """One English line per inferred value: ``name = value (source: detail)``.

    :param values: the mapping :func:`infer_optics` returns.
    :returns: list of strings, in the mapping's order.
    """
    lines = []
    for key, item in values.items():
        value = item.value
        if isinstance(value, tuple):
            value = ' × '.join(f'{v:.4g}' if isinstance(v, float) else str(v) for v in value)
        elif isinstance(value, float):
            value = f'{value:.4g}'
        detail = f': {item.detail}' if item.detail else ''
        lines.append(f'{key} = {value} ({item.source}{detail})')
    return lines


def fill_psf_settings(settings, source=None):
    """Fill unset Gaussian PSF calibration in Mask settings, in place.

    Acts only when ``psf_operation`` is convolve or deconvolve, ``psf_source``
    is gaussian and ``psf_image_sampling_um`` or ``psf_fwhm_um`` is None;
    values already set are never replaced. ``psf_objective`` (``'auto'`` or
    an :data:`OBJECTIVES` row name) picks the objective and :func:`infer_optics`
    supplies the rest.

    :param settings: Mask or timelapse settings dict.
    :param source: images to read metadata from; defaults to ``settings['src']``.
    :returns: the :func:`infer_optics` values used, or None when nothing was filled.
    """
    if settings.get('psf_operation', 'none') not in ('convolve', 'deconvolve'):
        return None
    if settings.get('psf_source', 'gaussian') != 'gaussian':
        return None
    if (settings.get('psf_image_sampling_um') is not None
            and settings.get('psf_fwhm_um') is not None):
        return None
    if source is None:
        source = settings.get('src')
        if isinstance(source, (list, tuple)):
            source = [item for item in source if item]
    values = infer_optics(source, objective_name=settings.get('psf_objective', 'auto'))
    if settings.get('psf_image_sampling_um') is None:
        settings['psf_image_sampling_um'] = [round(float(v), 6)
                                             for v in values['pixel_size_um'].value]
    if settings.get('psf_fwhm_um') is None:
        settings['psf_fwhm_um'] = [round(float(v), 6) for v in values['fwhm_um'].value]
    return values
