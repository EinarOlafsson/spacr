"""Opt-in PSF and enhancement-chain preparation and provenance for segmentation inputs.

Kernels are captured once per run. Only private intensity arrays are processed;
raw stacks and mask labels are never rewritten by this module. V1 reuse checks
both the requested kernel/settings and the exact completed archive bytes.

The enhancement chain follows the same route. Make Masks' chain
(:mod:`spacr.qt.detect_chain`) is written into a settings file as the
``enhance_*`` keys and read back here by :func:`prepare_chain`, with the
``psf_*`` settings folded in as the chain's PSF stage, so one plate run
applies exactly the steps a curator tuned, in the chain's own order:
background, PSF, denoise, contrast, sharpen. The session applies it per
selected channel at the stage the PSF already ran -- after illumination
correction and before normalization -- and its provenance goes into the
same ``psf/segmentation_application.json`` record, so a changed chain is
refused on resume the way a changed kernel is. With no chain step on, a
run is byte for byte what it was before the chain existed.

Spectral unmixing comes first. With ``unmix`` on, a bleed-through matrix is
estimated from the single-stain control wells ``unmix_controls`` names and
every raw field is unmixed across all its channels before illumination
correction, the PSF and the chain; the matrix joins the same record. Measure
estimates its own over the measured channels and unmixes each field before
its preprocessing hooks, recording the matrix in the saved run settings and
``measurements/bleed_through.json``.

Self-supervised denoising comes before the chain. With ``n2v_denoise`` on,
a Noise2Void (N2V2) model per segmentation channel is trained by CAREamics,
in its own environment, on a sample of the run's own raw fields -- no clean
images are needed -- or read from ``n2v_model``. Each field's selected
channels are denoised after illumination correction and before the PSF and
the chain, while the noise is still independent from pixel to pixel, which
is what Noise2Void assumes. The checkpoints' hashes and the training record
join the same provenance record.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .point_spread import (PSF, apply_psf, describe_optics, fill_psf_settings,
                           gaussian_psf, load_psf, _spacing)
from .cancellation import checkpoint


__all__ = ['PSFPlan', 'prepare_psf', 'validate_psf_resume', 'prepare_chain',
           'chain_problems', 'apply_chain', 'processing_requested']


def _check_cancel(cancel=None):
    """Bridge the calling pipeline's cancellation token into the PSF engine."""
    checkpoint()
    if cancel is not None and cancel.is_set():
        from .cancellation import PipelineCancelled
        raise PipelineCancelled('PSF measurement cancelled')
    return False


@dataclass(frozen=True)
class PSFPlan:
    """Immutable calibrated processing configuration, safe to pass to workers.

    :param operation: ``convolve`` or ``deconvolve``.
    :param kernel: captured immutable measured or Gaussian kernel.
    :param sampling_um: image spacing in spatial array order, in micrometers.
    :param iterations: Richardson–Lucy iterations; unused for convolution.
    """

    operation: str
    kernel: PSF
    sampling_um: tuple
    iterations: int

    def apply(self, image, *, cancel=None):
        """Process one spatial image with an explicit last channel axis.

        :param image: YXC or ZYXC intensity array, never a merged label stack.
        :param cancel: optional process-safe event shared with a parent worker.
        :returns: float32 processed intensities without source mutation.
        """
        return apply_psf(
            image, self.kernel, operation=self.operation,
            image_sampling_um=self.sampling_um, iterations=self.iterations,
            channel_axis=-1, cancel=lambda: _check_cancel(cancel)).image

    def provenance(self):
        """Return JSON-safe identity of every scientifically relevant setting."""
        record = {
            'operation': self.operation, 'kernel': self.kernel.provenance(),
            'image_sampling_um': list(self.sampling_um),
            'iterations': self.iterations if self.operation == 'deconvolve' else None,
            'algorithm': 'spacr.reflect_psf.v1',
            'boundary': 'half-sample symmetric',
            'output_dtype': 'float32', 'intensity_units': 'input units',
        }
        if self.operation == 'deconvolve':
            record.update(method='Richardson-Lucy', regularization=None,
                          epsilon_relative=1e-7, initialization='channel mean',
                          sensitivity_normalization=True)
        return record


def prepare_psf(settings, *, ndim=2):
    """Capture and validate an optional PSF before processing starts.

    :param settings: ``psf_operation`` (none/convolve/deconvolve), source
        (gaussian/measured), image sampling, Gaussian FWHM or measured path
        and kernel sampling, and iteration count. Lengths are in micrometers,
        YX for two dimensions or ZYX for three. No calibration is inferred.
    :param ndim: number of spatial dimensions, two or three.
    :returns: immutable :class:`PSFPlan`, or None when disabled (the default).
    :raises ValueError: for invalid operations, calibration or kernels.
    """
    operation = settings.get('psf_operation', 'none')
    if operation == 'none':
        return None
    if operation not in ('convolve', 'deconvolve'):
        raise ValueError('psf_operation must be none, convolve or deconvolve')
    spacing = _spacing(settings.get('psf_image_sampling_um'), ndim,
                       'psf_image_sampling_um')
    iterations = settings.get('psf_iterations', 20)
    if (isinstance(iterations, bool) or not isinstance(iterations, int)
            or not 1 <= iterations <= 200):
        raise ValueError('psf_iterations must be an integer from 1 to 200')
    source = settings.get('psf_source', 'gaussian')
    if source == 'gaussian':
        kernel = gaussian_psf(
            fwhm_um=settings.get('psf_fwhm_um'), sampling_um=spacing, ndim=ndim)
    elif source == 'measured':
        path = settings.get('psf_path')
        if not path:
            raise ValueError('psf_path must name a measured TIFF or NPY kernel')
        kernel = load_psf(
            path, sampling_um=settings.get('psf_kernel_sampling_um'),
            cancel=_check_cancel)
    else:
        raise ValueError('psf_source must be gaussian or measured')
    if len(kernel.shape) != ndim or not np.allclose(
            kernel.sampling_um, spacing, rtol=1e-6, atol=0):
        raise ValueError('PSF dimensions and sampling must match the image; '
                         'kernels are not implicitly resampled')
    return PSFPlan(operation, kernel, spacing, iterations)


def _bool(value):
    """A settings value as a bool, reading the words a settings file writes."""
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ('true', '1', 'yes', 'on'):
            return True
        if word in ('false', '0', 'no', 'off', ''):
            return False
        raise ValueError(f'{value!r} is not a yes or a no')
    return bool(value)


def _read_chain_settings(settings):
    """The ``enhance_*`` settings, each coerced to its chain field's type.

    Every key is read here and nowhere else, so the setting's API help lands
    on this function. A missing key means the step is off, which is what a
    settings file written before the chain existed says.

    :returns: ``(fields, problems)``: the :class:`spacr.qt.detect_chain.Chain`
        keyword arguments, and ``(setting, message)`` pairs for values that
        are not the type their field holds.
    """
    from .qt.detect_chain import NO_CHAIN

    raw = {
        'background': settings.get('enhance_background', NO_CHAIN.background),
        'background_radius': settings.get('enhance_background_radius',
                                          NO_CHAIN.background_radius),
        'background_scale': settings.get('enhance_background_scale',
                                         NO_CHAIN.background_scale),
        'denoise': settings.get('enhance_denoise', NO_CHAIN.denoise),
        'denoise_strength': settings.get('enhance_denoise_strength',
                                         NO_CHAIN.denoise_strength),
        'percentile_clip': settings.get('enhance_percentile_clip',
                                        NO_CHAIN.percentile_clip),
        'percentile_low': settings.get('enhance_percentile_low',
                                       NO_CHAIN.percentile_low),
        'percentile_high': settings.get('enhance_percentile_high',
                                        NO_CHAIN.percentile_high),
        'gamma': settings.get('enhance_gamma', NO_CHAIN.gamma),
        'log': settings.get('enhance_log', NO_CHAIN.log),
        'log_gain': settings.get('enhance_log_gain', NO_CHAIN.log_gain),
        'sqrt': settings.get('enhance_sqrt', NO_CHAIN.sqrt),
        'clahe': settings.get('enhance_clahe', NO_CHAIN.clahe),
        'clahe_tile': settings.get('enhance_clahe_tile', NO_CHAIN.clahe_tile),
        'clahe_clip': settings.get('enhance_clahe_clip', NO_CHAIN.clahe_clip),
        'equalize': settings.get('enhance_equalize', NO_CHAIN.equalize),
        'sharpen': settings.get('enhance_sharpen', NO_CHAIN.sharpen),
        'sharpen_radius': settings.get('enhance_sharpen_radius',
                                       NO_CHAIN.sharpen_radius),
        'sharpen_amount': settings.get('enhance_sharpen_amount',
                                       NO_CHAIN.sharpen_amount),
    }
    fields, problems = {}, []
    for field, value in raw.items():
        default = getattr(NO_CHAIN, field)
        if value is None:
            value = default
        try:
            if isinstance(default, bool):
                fields[field] = _bool(value)
            elif isinstance(default, int):
                fields[field] = int(float(value))
            elif isinstance(default, float):
                fields[field] = float(value)
            else:
                fields[field] = str(value).strip().lower()
        except (TypeError, ValueError):
            problems.append((f'enhance_{field}',
                             f'enhance_{field} must be a '
                             f'{type(default).__name__}, not {value!r}'))
            fields[field] = default
    return fields, problems


def chain_problems(settings):
    """Every ``enhance_*`` value the chain cannot run with, as ``(setting, message)``.

    Read by the preflight so a run refuses before any field is touched, and
    by :func:`prepare_chain`, which raises on the first.

    :param settings: the run's settings; absent ``enhance_*`` keys are off.
    :returns: a list of ``(setting, message)`` pairs, empty when all is well.
    """
    from .qt.detect_chain import BACKGROUND_METHODS, DENOISE_METHODS

    fields, problems = _read_chain_settings(settings)
    if fields['background'] not in BACKGROUND_METHODS:
        problems.append(('enhance_background',
                         'enhance_background must be one of '
                         + ', '.join(BACKGROUND_METHODS)))
    if fields['background_radius'] < 1:
        problems.append(('enhance_background_radius',
                         'enhance_background_radius must be at least 1 pixel'))
    if not 0.0 < fields['background_scale'] <= 1.0:
        problems.append(('enhance_background_scale',
                         'enhance_background_scale must be above 0 and at most 1'))
    if fields['denoise'] not in DENOISE_METHODS:
        problems.append(('enhance_denoise',
                         'enhance_denoise must be one of '
                         + ', '.join(DENOISE_METHODS)))
    if fields['denoise_strength'] <= 0.0:
        problems.append(('enhance_denoise_strength',
                         'enhance_denoise_strength must be above 0'))
    if not 0.0 <= fields['percentile_low'] < fields['percentile_high'] <= 100.0:
        problems.append(('enhance_percentile_low',
                         'enhance_percentile_low must be at least 0 and below '
                         'enhance_percentile_high, which must be at most 100'))
    if fields['gamma'] <= 0.0:
        problems.append(('enhance_gamma', 'enhance_gamma must be above 0'))
    if fields['log_gain'] <= 0.0:
        problems.append(('enhance_log_gain', 'enhance_log_gain must be above 0'))
    if fields['clahe_tile'] < 8:
        problems.append(('enhance_clahe_tile',
                         'enhance_clahe_tile must be at least 8 pixels'))
    if not 0.0 < fields['clahe_clip'] <= 1.0:
        problems.append(('enhance_clahe_clip',
                         'enhance_clahe_clip must be above 0 and at most 1'))
    if fields['sharpen_radius'] <= 0.0:
        problems.append(('enhance_sharpen_radius',
                         'enhance_sharpen_radius must be above 0'))
    if fields['sharpen_amount'] < 0.0:
        problems.append(('enhance_sharpen_amount',
                         'enhance_sharpen_amount cannot be negative'))
    return problems


def prepare_chain(settings, plan=None):
    """The enhancement chain a run applies, read from its ``enhance_*`` settings.

    The chain is :class:`spacr.qt.detect_chain.Chain`, the value Make Masks
    detects with, built from the settings :func:`spacr.qt.detect_chain.chain_settings`
    writes; the optional :class:`PSFPlan` is folded in as the chain's PSF
    stage so the run's order is the chain's own -- background before the
    PSF, then denoise, contrast and sharpen -- rather than the PSF first.

    :param settings: the run's settings; ``enhance_<field>`` per
        :data:`spacr.qt.detect_chain.SETTINGS_FIELDS`, off when absent.
    :param plan: the run's captured :class:`PSFPlan`, or None.
    :returns: the chain, or None when no ``enhance_*`` step is switched on --
        the PSF alone stays on its own path, which is byte for byte the path
        it had before the chain reached Mask.
    :raises ValueError: for a value the chain cannot run with, or a PSF that
        is not two-dimensional.
    """
    from .qt.detect_chain import Chain, pre_active

    problems = chain_problems(settings)
    if problems:
        raise ValueError(problems[0][1])
    fields, _ = _read_chain_settings(settings)
    chain = Chain(**fields)
    if not pre_active(chain):
        return None
    if plan is None:
        return chain
    if len(plan.kernel.shape) != 2:
        raise ValueError('the enhancement chain processes 2D fields; '
                         'give it a two-dimensional PSF')
    return chain._replace(psf_operation=plan.operation, psf=plan.kernel,
                          psf_sampling_um=tuple(plan.sampling_um),
                          psf_iterations=plan.iterations)


def processing_requested(settings):
    """Whether these settings ask for a PSF or an enhancement step at all.

    A requested chain that cannot be built counts as requested: the run
    then refuses with the reason rather than reusing inputs made without it.

    :param settings: the run's ``psf_*``, ``enhance_*``, ``unmix`` and
        ``n2v_denoise`` settings.
    :returns: True when any is switched on or the chain cannot be read.
    """
    if settings.get('psf_operation', 'none') != 'none':
        return True
    if settings.get('unmix', False):
        return True
    if _bool(settings.get('n2v_denoise', False)):
        return True
    try:
        return prepare_chain(settings) is not None
    except ValueError:
        return True


def apply_chain(image, chain, *, cancel=None):
    """Run ``chain`` over every channel of one YXC field, strictly.

    :param image: YXC intensity array, never a merged label stack.
    :param chain: a :class:`spacr.qt.detect_chain.Chain` from :func:`prepare_chain`.
    :param cancel: optional process-safe event shared with a parent worker;
        cancellation raises :class:`spacr.cancellation.PipelineCancelled`
        between stages and inside PSF iterations, as the PSF path does.
    :returns: float32 processed intensities, the source untouched.
    """
    from .qt.detect_chain import prepare

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError('the enhancement chain processes YX fields with a '
                         f'channel axis, not a {array.ndim}-dimensional array')
    out = np.empty(array.shape, dtype=np.float32)
    for index in range(array.shape[-1]):
        out[..., index] = prepare(
            np.asarray(array[..., index], dtype=np.float32), chain,
            cancel=lambda: _check_cancel(cancel), strict=True)
    return out


_UNMIX_RECORD_KEY = '_unmix_record'
_UNMIX_MAX_FIELDS = 24
_UNMIX_CALIBRATED_CONTROL_VALUES = 64 * 1024 ** 2
_UNMIX_MAX_CONDITION = 1e6


@dataclass(frozen=True)
class _UnmixPlan:
    """A bleed-through matrix estimated from controls or explicitly supplied.

    ``matrix[i][j]`` is the signal read in ``channels[i]`` per unit of the
    dye whose own channel is ``channels[j]``, so the diagonal is one.
    Unmixing solves every pixel's readings for the dye amounts, after each
    channel's background percentile is set aside and before it is added
    back, so an unstained channel stays at its own background.

    :param channels: the field's channel indices the matrix spans.
    :param matrix: square nested tuple, one row and column per channel.
    :param background_percentile: per-plane percentile taken as background.
    :param controls: ``{dye channel: (well, ...)}`` as the settings named them.
    :param fields: ``{dye channel: (field stem, ...)}`` the estimate read.
    :param calibration: optional captured optical calibration for ROI-aware preview.
    :param control_selection: immutable JSON plate-control evidence, absent for global plans.
    :param matrix_source: immutable supplied-matrix bundle evidence, when explicitly enabled.
    :param matrix_input_bytes: exact bounded matrix-file bytes retained for the output bundle.
    """

    channels: tuple
    matrix: tuple
    background_percentile: float
    controls: dict
    fields: dict
    calibration: object = None
    control_selection: object = None
    matrix_source: object = None
    matrix_input_bytes: object = None

    def provenance(self):
        """The plan as a JSON-safe record for the run's provenance."""
        record = {
            'method': 'single-stain linear unmixing',
            'channels': [int(channel) for channel in self.channels],
            'matrix': [[float(value) for value in row] for row in self.matrix],
            'background_percentile': float(self.background_percentile),
            'controls': {str(dye): list(wells)
                         for dye, wells in sorted(self.controls.items())},
            'fields': {str(dye): list(stems)
                       for dye, stems in sorted(self.fields.items())},
        }
        if self.calibration is not None:
            record['calibration'] = self.calibration.provenance()
        if self.control_selection is not None:
            record['control_selection'] = json.loads(self.control_selection)
        if self.matrix_source is not None:
            record['method'] = 'supplied_matrix'
            record['matrix_source'] = json.loads(self.matrix_source)
        return record

    def apply(self, image):
        """Unmix the plan's channels of one field, the other planes untouched.

        :param image: YXC or ZYXC intensities holding every plan channel.
        :returns: float32 copy with the plan's channels unmixed.
        """
        if self.calibration is not None:
            from .qt.i18n import tr

            raise ValueError(tr("Calibrated unmixing requires an explicit valid-region result; use the supported 2D live preview."))
        array = np.asarray(image)
        if array.ndim < 3:
            from .qt.i18n import tr

            raise ValueError(tr("Spectral unmixing requires images with an explicit channel axis and at least three dimensions."))
        out = np.array(array, dtype=np.float32, copy=True)
        channels = list(self.channels)
        out[..., channels] = _unmix(array[..., channels], self.matrix,
                                    self.background_percentile)
        return out

    def apply_calibrated(self, image):
        """Return a signed calibrated crop with its original-canvas bounds."""
        if self.calibration is None:
            from .qt.i18n import tr

            raise ValueError(tr("No spectral calibration was supplied."))
        matched, region = self.calibration.apply(image)
        signed = _unmix(matched, self.matrix, self.background_percentile,
                        clip_negative=False)
        record = self.provenance()
        record.update(mode='calibrated_live_preview',
                      valid_roi_yx=[[part.start, part.stop] for part in region],
                      source_origin_px=[part.start for part in region],
                      source_shape=list(np.shape(image)),
                      quantitative_output='signed float32 linear residuals with background restored',
                      negative_fraction_per_channel={str(channel): float(np.mean(signed[..., index] < 0))
                          for index, channel in enumerate(self.channels)})
        return _CalibratedUnmixResult(signed, region,
                                      tuple(part.start for part in region), record)


@dataclass(frozen=True)
class _CalibratedUnmixResult:
    """Signed valid-region intensities and their original image coordinates."""

    image: np.ndarray
    valid_slices: tuple
    source_origin_px: tuple
    record: dict


@dataclass(frozen=True)
class _UnmixCalibration:
    """Explicit, captured same-grid optical calibration for a 2D preview."""

    channels: tuple
    shifts: tuple
    matching_kernels: tuple
    record: dict

    def provenance(self):
        """Return an independent JSON-safe copy of the captured calibration."""
        return json.loads(json.dumps(self.record))

    def valid_region(self, shape):
        """Validate target dimensions and return acquired support without decoding pixels."""
        from .qt.i18n import tr

        if (len(shape) != 3 or shape[-1] != len(self.channels)
                or any(size <= 0 for size in shape)
                or int(shape[0]) * int(shape[1]) * int(shape[2]) > 16 * 1024 ** 2):
            raise ValueError(tr("Calibrated unmixing requires a nonempty 2D channel-last image with at most 16,777,216 channel pixels."))
        height, width = shape[:2]
        top, left, bottom, right = 0, 0, height, width
        for shift, kernel in zip(self.shifts, self.matching_kernels):
            dy, dx = shift
            ry, rx = (size // 2 for size in kernel.shape)
            top, left = max(top, dy + ry), max(left, dx + rx)
            bottom, right = min(bottom, height + dy - ry), min(right, width + dx - rx)
        if bottom <= top or right <= left:
            raise ValueError(tr("The calibrated shifts and PSF support leave no common valid image region."))
        return slice(top, bottom), slice(left, right)

    def apply(self, image):
        """Match only real acquired patches within a common valid rectangle."""
        from scipy.signal import fftconvolve
        from .qt.i18n import tr

        array = np.asarray(image)
        region = self.valid_region(array.shape)
        if np.iscomplexobj(array):
            raise ValueError(tr("Calibrated unmixing requires a nonempty 2D channel-last image with at most 16,777,216 channel pixels."))
        if not np.isfinite(array).all():
            raise ValueError(tr("Calibrated unmixing requires finite source pixels."))
        top, bottom = region[0].start, region[0].stop
        left, right = region[1].start, region[1].stop
        matched = np.empty((bottom - top, right - left, len(self.channels)), dtype=np.float64)
        for index, (shift, kernel) in enumerate(zip(self.shifts, self.matching_kernels)):
            checkpoint()
            dy, dx = shift
            ry, rx = (size // 2 for size in kernel.shape)
            patch = array[top - dy - ry:bottom - dy + ry,
                          left - dx - rx:right - dx + rx, index]
            matched[..., index] = fftconvolve(patch, kernel, mode='valid')
        return matched, region


def _load_unmix_calibration(settings, channels):
    """Capture and validate explicitly supplied 2D registration and PSF samples."""
    from scipy.signal import convolve2d
    from .qt.i18n import tr

    path = Path(settings['unmix_calibration']).expanduser()
    with path.open('rb') as stream:
        raw = stream.read(1024 ** 2 + 1)
    if len(raw) > 1024 ** 2:
        raise ValueError(tr("The spectral calibration file must not exceed 1 MiB."))
    payload = json.loads(raw)
    def spacing(value):
        """Read an explicitly positive finite YX pixel spacing."""
        result = np.asarray(value, dtype=float)
        if result.shape != (2,) or not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError(tr("Spectral calibration requires explicit positive finite YX pixel sizes."))
        return result
    actual_spacing = spacing(settings.get('unmix_pixel_size_um'))
    if (not isinstance(payload, dict) or type(payload.get('version')) is not int
            or payload.get('version') != 1 or payload.get('axes') != 'YX'
            or not np.allclose(spacing(payload.get('pixel_size_um')), actual_spacing, rtol=1e-6, atol=0)):
        raise ValueError(tr("Spectral calibration axes, version and pixel sampling must match the 2D source; no resampling is performed."))
    entries = payload.get('channels')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 16:
        raise ValueError(tr("Spectral calibration must name between 1 and 16 physical channels exactly once."))
    by_channel = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(tr("Spectral calibration must name between 1 and 16 physical channels exactly once."))
        channel = entry.get('channel')
        if isinstance(channel, bool) or not isinstance(channel, int) or channel in by_channel:
            raise ValueError(tr("Spectral calibration must name between 1 and 16 physical channels exactly once."))
        by_channel[channel] = entry
    if set(by_channel) != set(channels):
        raise ValueError(tr("The spectral calibration channels must exactly match the selected physical channels."))
    reference = payload.get('reference_channel')
    if type(reference) is not int or reference not in by_channel:
        raise ValueError(tr("The spectral calibration must name a selected reference channel with zero shift."))
    kernels, shifts, records = [], [], []
    for channel in channels:
        entry = by_channel[channel]
        shift = entry.get('shift_to_reference_px')
        if (not isinstance(shift, list) or len(shift) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or abs(value) > 4096 for value in shift)):
            raise ValueError(tr("Spectral registration requires integer YX shifts between -4096 and 4096 pixels."))
        if channel == reference and shift != [0, 0]:
            raise ValueError(tr("The spectral calibration must name a selected reference channel with zero shift."))
        kernel = np.asarray(entry.get('psf'), dtype=np.float64)
        if (kernel.ndim != 2 or not all(0 < size <= 31 and size % 2 for size in kernel.shape)
                or not np.isfinite(kernel).all() or np.any(kernel < 0)
                or not 0 < kernel.sum() < np.inf):
            raise ValueError(tr("Each calibrated PSF must be a finite nonnegative odd 2D kernel, at most 31 pixels per side, with positive mass."))
        if (entry.get('psf_origin_px') != [size // 2 for size in kernel.shape]
                or not np.allclose(spacing(entry.get('psf_pixel_size_um')), actual_spacing, rtol=1e-6, atol=0)):
            raise ValueError(tr("Calibrated PSF origins must be their declared centers and their sampling must match the source."))
        kernel = kernel / kernel.sum()
        kernels.append(kernel)
        shifts.append(tuple(shift))
        records.append({'channel': channel, 'shift_to_reference_px': shift,
                        'psf': kernel.tolist(), 'psf_origin_px': entry['psf_origin_px'],
                        'psf_pixel_size_um': actual_spacing.tolist(),
                        'normalized_psf_sha256': hashlib.sha256(kernel.tobytes()).hexdigest()})
    common_shape = tuple(1 + sum(kernel.shape[axis] - 1 for kernel in kernels) for axis in (0, 1))
    if max(common_shape) > 127:
        raise ValueError(tr("The combined calibrated PSF support must not exceed 127 pixels per side."))
    common = np.ones((1, 1))
    for kernel in kernels:
        common = convolve2d(common, kernel, mode='full')
    matching = []
    for own in range(len(kernels)):
        combined = np.ones((1, 1))
        for other, kernel in enumerate(kernels):
            if other != own:
                combined = convolve2d(combined, kernel, mode='full')
        matching.append(combined)
    record = {'version': 1, 'axes': 'YX', 'pixel_size_um': actual_spacing.tolist(),
              'reference_channel': reference, 'channels': records,
              'source': str(path.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'shift_convention': 'aligned[y,x] = acquired[y-dy,x-dx]',
              'matching': 'cross-convolution to the product of all supplied channel PSFs',
              'common_psf': common.tolist(), 'common_psf_support_yx': list(common.shape),
              'resolution': 'broadened to the common product PSF; no deconvolution',
              'boundary': 'intersection of acquired support; valid convolution only'}
    return _UnmixCalibration(tuple(channels), tuple(shifts), tuple(matching), record)


def _parse_unmix_controls(value):
    """Read the single-stain controls as ``{dye channel: (well, ...)}``.

    :param value: ``'0:A01,A02; 1:B01'`` text, or a mapping of channel to a
        well or a list of wells.
    :returns: dict of int channel to a tuple of well names, empty for blank.
    :raises ValueError: for an entry that is not ``channel:well[,well]``.
    """
    from .schema import parse_well

    if value is None:
        return {}
    if isinstance(value, dict):
        items = [(key, [wells] if isinstance(wells, str) else list(wells))
                 for key, wells in value.items()]
    else:
        items = []
        for entry in str(value).replace('\n', ';').split(';'):
            if not entry.strip():
                continue
            if ':' not in entry:
                raise ValueError(
                    f'unmix_controls entry {entry.strip()!r} is not '
                    'channel:well[,well], for example 0:A01,A02; 1:B01')
            key, wells = entry.split(':', 1)
            items.append((key, wells.split(',')))
    controls = {}
    for key, wells in items:
        try:
            dye = int(str(key).strip())
        except ValueError as exc:
            raise ValueError(
                f'unmix_controls channel {key!r} is not a whole number') from exc
        names = tuple(str(well).strip() for well in wells if str(well).strip())
        if not names:
            raise ValueError(f'unmix_controls names no well for channel {dye}')
        for name in names:
            parse_well(name, strict=True)
        controls[dye] = controls.get(dye, ()) + names
    return controls


def _unmix_background(pixels, background_percentile):
    """Per-channel background of flattened ``(pixels, channels)`` values."""
    return np.percentile(pixels, float(background_percentile), axis=0)


def _estimate_bleed_through(fields_by_dye, channels):
    """Estimate the bleed-through matrix from single-stain control fields.

    For each dye, every other channel is regressed on the dye's own channel
    over all pixels of its control fields, each field centred on its own
    means so a background offset never reads as bleed-through, and the
    slopes pooled; a negative spill is taken as none.

    :param fields_by_dye: ``{dye channel: [field array over channels]}``.
    :param channels: the channel indices the fields' last axis holds.
    :returns: float64 square matrix with a unit diagonal.
    :raises ValueError: for a control with no signal above background, or a
        matrix too close to singular to invert.
    """
    channels = list(channels)
    count = len(channels)
    matrix = np.eye(count)
    for dye, fields in sorted(fields_by_dye.items()):
        own = channels.index(dye)
        cross = np.zeros(count)
        power = 0.0
        for field in fields:
            pixels = np.asarray(field, dtype=np.float64).reshape(-1, count)
            pixels = pixels - pixels.mean(axis=0)
            signal = pixels[:, own]
            cross += pixels.T @ signal
            power += float(signal @ signal)
        if not power > 0.0:
            raise ValueError(
                f'the single-stain control for channel {dye} is flat, so '
                'its bleed-through cannot be estimated')
        column = np.clip(cross / power, 0.0, None)
        column[own] = 1.0
        matrix[:, own] = column
    if not np.linalg.cond(matrix) < _UNMIX_MAX_CONDITION:
        raise ValueError(
            'the bleed-through matrix is too close to singular to unmix; '
            'check that each control well holds only its own dye')
    return matrix


def _unmix(image, matrix, background_percentile, *, clip_negative=True):
    """Solve every pixel of ``image`` for its dye amounts with ``matrix``.

    :param image: array whose last axis holds the matrix's channels in order.
    :param matrix: square bleed-through matrix.
    :param background_percentile: per-plane percentile taken as background;
        it is removed before solving and added back after, and the result is
        clipped at zero.
    :param clip_negative: preserve the legacy zero floor by default; False
        retains signed calibrated-preview residuals.
    :returns: float32 unmixed intensities of the same shape.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    array = np.asarray(image, dtype=np.float64)
    count = matrix.shape[0]
    if array.shape[-1] != count:
        raise ValueError(f'unmixing expects {count} channels on the last '
                         f'axis, got {array.shape[-1]}')
    pixels = array.reshape(-1, count)
    background = _unmix_background(pixels, background_percentile)
    solved = (pixels - background) @ np.linalg.inv(matrix).T + background
    if clip_negative:
        solved = np.clip(solved, 0.0, None)
    return solved.reshape(array.shape).astype(np.float32)


def _validate_unmix_plate_scope(settings, *, allow_per_plate=False):
    """Admit plate-qualified controls only through their explicit supported route.

    :param settings: spectral settings including the optional per-plate switch.
    :param allow_per_plate: whether the caller supplies calibrated V2 identities.
    :returns: whether the explicitly enabled plate-qualified mode is active.
    """
    from .qt.i18n import tr

    enabled = _bool(settings.get('unmix_per_plate', False))
    if enabled and (not allow_per_plate or not _bool(settings.get('unmix', False))
                    or not settings.get('unmix_calibration')):
        raise ValueError(tr("Per-plate spectral controls require calibrated organized 2D cell segmentation through V2; this processing path is not supported."))
    return enabled


def _plate_control_pairs(pairs):
    """Reject repeated JSON keys before their original spelling can be lost.

    :param pairs: JSON object entries in their supplied order.
    :returns: a mapping without duplicate keys.
    """
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError
    return dict(pairs)


def _plate_unmix_controls(settings, channels, identities):
    """Capture one exact plate's complete controls from a bounded master map.

    :param settings: enabled plate-qualified spectral settings.
    :param channels: physical channel IDs in their actual packed order.
    :param identities: captured physical identities of all source fields.
    :returns: selected controls and independent JSON-safe selection metadata.
    """
    from .qt.i18n import tr
    from .schema import parse_well, well_id

    message = tr("Per-plate spectral controls require a bounded JSON plate-to-channel-to-well-list map with complete controls for the exact source plate and physical channels.")
    try:
        raw = settings.get('unmix_controls')
        text = raw if isinstance(raw, str) else json.dumps(raw, allow_nan=False)
        if len(text.encode('utf-8')) > 1024 ** 2:
            raise ValueError
        master = json.loads(text, object_pairs_hook=_plate_control_pairs)
        if not isinstance(master, dict) or not master or len(master) > 8192:
            raise ValueError
        canonical = {}
        for plate, entries in master.items():
            if not isinstance(plate, str) or not plate or plate != plate.strip() or not isinstance(entries, dict) or not 1 <= len(entries) <= 16:
                raise ValueError
            mapped = {}
            used = set()
            for channel, wells in entries.items():
                number = int(channel)
                if str(number) != channel or number < 0 or not isinstance(wells, list) or not wells:
                    raise ValueError
                normalized = []
                for well in wells:
                    if not isinstance(well, str):
                        raise ValueError
                    row, column = parse_well(well, strict=True)
                    normalized.append(well_id(row, column))
                if len(set(normalized)) != len(normalized) or used.intersection(normalized):
                    raise ValueError
                used.update(normalized)
                mapped[str(number)] = sorted(normalized)
            canonical[plate] = mapped
        if not identities or any(not _cell_record_identity_valid(identity) for identity in identities.values()):
            raise ValueError
        contexts = {(identity['plate'], identity['time'], identity['z'], identity['source_dataset']) for identity in identities.values()}
        if len(contexts) != 1 or not channels:
            raise ValueError
        plate = next(iter(contexts))[0]
        selected = canonical[plate]
        if (not 1 <= len(channels) <= 16
                or any(isinstance(channel, (bool, np.bool_))
                       or not isinstance(channel, (int, np.integer)) or channel < 0
                       for channel in channels)
                or len(set(channels)) != len(channels)):
            raise ValueError
        physical = tuple(int(channel) for channel in channels)
        if set(selected) != {str(channel) for channel in physical}:
            raise ValueError
        available = {parse_well(identity['well'], strict=True) for identity in identities.values()}
        if any(parse_well(well, strict=True) not in available for wells in selected.values() for well in wells):
            raise ValueError
        selection = dict(scope='plate', plate=plate, physical_channel_order=list(physical),
                         canonical_controls_by_plate=canonical,
                         controls_map_sha256=_cell_record_hash(canonical),
                         selected_controls=selected, sampled_fields={})
        return _parse_unmix_controls(selected), selection
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError(message) from None


def _plate_control_hash(stream, expected_size):
    """Hash at most the captured input length plus one byte to detect growth.

    :param stream: the same open descriptor used for the control pixels.
    :param expected_size: captured byte count already admitted by the budget.
    :returns: exact SHA256 and unchanged native file identity.
    """
    before = os.fstat(stream.fileno())
    if before.st_size != expected_size:
        raise ValueError
    stream.seek(0)
    digest = hashlib.sha256()
    consumed = 0
    while True:
        checkpoint()
        block = stream.read(min(1024 ** 2, expected_size - consumed + 1))
        if not block:
            break
        consumed += len(block)
        if consumed > expected_size:
            raise ValueError
        digest.update(block)
    after = os.fstat(stream.fileno())
    original = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    current = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if consumed != expected_size or current != original:
        raise ValueError
    return digest.hexdigest(), current


@contextmanager
def _plate_control_input(path, remaining_bytes):
    """Map and hash the same immutable bounded NPY descriptor used by estimation.

    :param path: assembled raw control NPY, before segmentation replaces it.
    :param remaining_bytes: remaining unique-control input-byte hashing budget.
    :yields: a read-only numeric map and its captured input facts.
    """
    from .qt.i18n import tr

    message = tr("Per-plate control inputs must be stable bounded numeric 2D NPY fields throughout calibration.")
    mapped = None
    delivered = checking = False
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(descriptor, 'rb') as stream:
            size = os.fstat(stream.fileno()).st_size
            if size > min(remaining_bytes, 128 * 1024 ** 2 + 16384):
                raise ValueError(message)
            version = np.lib.format.read_magic(stream)
            readers = {(1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0}
            if version not in readers:
                raise ValueError(message)
            shape, fortran, dtype = readers[version](stream, max_header_size=16384)
            offset = stream.tell()
            if (fortran or len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape)
                    or shape[-1] > 16 or np.prod(shape, dtype=object) > 16 * 1024 ** 2
                    or dtype.hasobject or dtype.kind not in 'uif' or dtype.itemsize > 8
                    or size != offset + int(np.prod(shape, dtype=object)) * dtype.itemsize):
                raise ValueError(message)
            digest, identity = _plate_control_hash(stream, size)
            mapped = np.memmap(stream, mode='r', dtype=dtype, offset=offset, shape=shape, order='C')
            facts = dict(input_stage='assembled_pre_unmixing', input_sha256=digest,
                         input_size_bytes=size, input_shape=list(shape), input_dtype=dtype.str)
            delivered = True
            yield mapped, facts
            checking = True
            after_digest, after_identity = _plate_control_hash(stream, size)
            current = os.stat(path, follow_symlinks=False)
            if (after_digest != digest or after_identity != identity
                    or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != identity):
                raise ValueError(message)
    except (OSError, TypeError, ValueError) as error:
        if delivered and not checking:
            raise
        raise ValueError(message) from error
    finally:
        if mapped is not None:
            mapped._mmap.close()


def _validate_unmix_matrix_scope(settings, *, allow_matrix=False):
    """Refuse supplied matrices outside the explicitly supported calibrated V2 route.

    :param settings: spectral settings including the optional matrix-file path.
    :param allow_matrix: this caller transports calibrated physical identities.
    :returns: whether a supplied matrix was requested on an admitted route.
    """
    path = settings.get('unmix_matrix_file')
    if path is None or isinstance(path, str) and not path.strip():
        return False
    from .qt.i18n import tr

    message = tr("A supplied spectral matrix requires calibrated organized 2D cell V2 segmentation, without control wells or per-plate estimation; this processing path is not supported.")
    controls = settings.get('unmix_controls')
    if isinstance(controls, str):
        controls = controls.strip()
    try:
        if (not isinstance(path, (str, os.PathLike)) or not allow_matrix
                or not _bool(settings.get('unmix', False))
                or not settings.get('unmix_calibration') or controls
                or _bool(settings.get('unmix_per_plate', False))
                or not hasattr(os, 'O_DIRECTORY') or not hasattr(os, 'O_NOFOLLOW')
                or not hasattr(os, 'O_NONBLOCK')
                or os.open not in os.supports_dir_fd or os.mkdir not in os.supports_dir_fd):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(message) from None
    return True


def _matrix_input_error():
    """Return the localized refusal for inconsistent supplied-matrix evidence."""
    from .qt.i18n import tr

    return ValueError(tr("The supplied spectral matrix must be a stable version-1 JSON file of at most 1 MiB, with matching physical channel order, acquisition context, calibration and background policy; acquisition compatibility remains an unverified user assertion."))


def _matrix_json_constant(value):
    """Reject nonfinite JSON extensions before numeric conversion.

    :param value: the unsupported JSON numeric constant.
    """
    raise ValueError(value)


def _matrix_input_snapshot(stream):
    """Capture bounded exact bytes from a stable regular-file descriptor.

    :param stream: binary stream for the supplied matrix or its copied bundle.
    :returns: raw bytes, parsed artifact, SHA256 and native file identity.
    """
    import stat

    before = os.fstat(stream.fileno())
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024 ** 2:
        raise _matrix_input_error()
    raw = stream.read(1024 ** 2 + 1)
    after = os.fstat(stream.fileno())
    if (len(raw) != before.st_size or identity !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise _matrix_input_error()
    payload = json.loads(raw, object_pairs_hook=_plate_control_pairs,
                         parse_constant=_matrix_json_constant)
    return raw, payload, hashlib.sha256(raw).hexdigest(), identity


def _validate_matrix_artifact(payload, channels, calibration, percentile, contexts):
    """Bind a supplied matrix to the actual selected operator and acquisition IDs.

    :param payload: parsed version-1 matrix artifact.
    :param channels: physical channel IDs in persisted order.
    :param calibration: existing captured optical-calibration provenance.
    :param percentile: configured per-field background percentile.
    :param contexts: exact actual plate, time and Z identity tuples.
    :returns: the finite, unit-diagonal float64 mixing matrix.
    """
    keys = {'artifact', 'version', 'source_context', 'channels', 'orientation',
            'matrix', 'background_percentile', 'background_operator',
            'calibration_source_sha256', 'acquisition_compatibility'}
    if (not isinstance(payload, dict) or set(payload) != keys
            or payload['artifact'] != 'spacr_spectral_matrix'
            or type(payload['version']) is not int or payload['version'] != 1
            or payload['orientation'] != 'rows_readings_columns_dyes'
            or payload['background_operator'] != 'per_field_percentile_subtract_solve_restore'
            or payload['acquisition_compatibility'] != 'user_asserts_same_channel_gains_exposures_and_linear_response'):
        raise _matrix_input_error()
    physical = payload['channels']
    context = payload['source_context']
    if (not isinstance(physical, list) or not 1 <= len(physical) <= 16
            or any(type(channel) is not int or channel < 0 for channel in physical)
            or len(set(physical)) != len(physical) or physical != list(channels)
            or not isinstance(context, dict) or set(context) != {'plate', 'time', 'z'}
            or not isinstance(context['plate'], str) or not context['plate']
            or any(not (type(context[key]) is int and context[key] >= 0
                        or isinstance(context[key], str)) for key in ('time', 'z'))
            or contexts != {(context['plate'], context['time'], context['z'])}):
        raise _matrix_input_error()
    background = payload['background_percentile']
    digest = payload['calibration_source_sha256']
    if (type(background) not in (int, float) or not np.isfinite(background)
            or not 0 <= background < 100 or background != percentile
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in '0123456789abcdef' for character in digest)
            or digest != calibration['source_sha256']):
        raise _matrix_input_error()
    rows = payload['matrix']
    count = len(physical)
    if (not isinstance(rows, list) or len(rows) != count
            or any(not isinstance(row, list) or len(row) != count
                   or any(type(value) not in (int, float) for value in row) for row in rows)):
        raise _matrix_input_error()
    matrix = np.asarray(rows, dtype=np.float64)
    if (not np.isfinite(matrix).all() or np.any(matrix < 0)
            or not np.all(np.diag(matrix) == 1)
            or not np.linalg.cond(matrix) < _UNMIX_MAX_CONDITION):
        raise _matrix_input_error()
    return matrix


def _prepare_supplied_matrix(settings, channels, field_identities):
    """Capture one supplied matrix before creating outputs or opening source pixels.

    :param settings: admitted matrix-file, calibration and background settings.
    :param channels: physical IDs in packed order.
    :param field_identities: actual mapper identities from one source context.
    :returns: immutable numeric plan carrying the exact input bundle bytes.
    """
    try:
        if (not channels or len(channels) > 16 or len(set(channels)) != len(channels)
                or any(isinstance(channel, (bool, np.bool_))
                       or not isinstance(channel, (int, np.integer)) or channel < 0 for channel in channels)):
            raise _matrix_input_error()
        channels = tuple(int(channel) for channel in channels)
        if (not field_identities or any(not _cell_record_identity_valid(identity)
                                       for identity in field_identities.values())):
            raise _matrix_input_error()
        roots = {identity['source_dataset'] for identity in field_identities.values()}
        if len(roots) != 1:
            raise _matrix_input_error()
        contexts = {(identity['plate'], identity['time'], identity['z'])
                    for identity in field_identities.values()}
        path = Path(settings['unmix_matrix_file']).expanduser()
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream:
            raw, payload, digest, identity = _matrix_input_snapshot(stream)
            current = path.lstat()
            if identity != (current.st_dev, current.st_ino, current.st_size,
                            current.st_mtime_ns, current.st_ctime_ns):
                raise _matrix_input_error()
        calibration = _load_unmix_calibration(settings, channels)
        background = settings.get('unmix_background_percentile', 5.0)
        if isinstance(background, (bool, np.bool_)):
            raise _matrix_input_error()
        percentile = float(background)
        matrix = _validate_matrix_artifact(payload, channels, calibration.provenance(),
                                           percentile, contexts)
        source = dict(kind='supplied_matrix', version=1, path='psf/spectral_matrix_input.json',
                      sha256=digest, size_bytes=len(raw), artifact=payload,
                      acquisition_compatibility_verified=False)
        return _UnmixPlan(tuple(channels), tuple(tuple(float(value) for value in row)
                                                for row in matrix), percentile, {}, {},
                          calibration=calibration, matrix_source=json.dumps(source, allow_nan=False),
                          matrix_input_bytes=raw)
    except (OSError, KeyError, TypeError, ValueError, OverflowError, RecursionError, np.linalg.LinAlgError):
        raise _matrix_input_error() from None


def _persist_supplied_matrix(root, plan):
    """Copy captured bytes once without consulting the original matrix path again.

    :param root: fresh calibrated output directory.
    :param plan: captured supplied-matrix plan.
    """
    source = json.loads(plan.matrix_source)
    raw = plan.matrix_input_bytes
    if (not isinstance(raw, bytes) or len(raw) != source['size_bytes']
            or hashlib.sha256(raw).hexdigest() != source['sha256']):
        raise _matrix_input_error()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.mkdir('psf', dir_fd=root_fd)
        except FileExistsError:
            pass
        directory_fd = os.open('psf', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            descriptor = os.open('spectral_matrix_input.json',
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(directory_fd)
    finally:
        os.close(root_fd)


def _prepare_unmixing(settings, source_dir, channels=None, load=None, *, allow_calibrated=False,
                     channel_positions=None, field_identities=None):
    """Capture the run's supplied matrix or estimate it from single-stain controls.

    :param settings: ``unmix``, ``unmix_controls`` and
        ``unmix_background_percentile``. Optional ``unmix_per_plate`` requires
        a complete JSON plate-to-physical-channel-to-well-list master map;
        only the exact source plate is selected, without cross-plate fallback.
    :param source_dir: folder of ``plate_well_field[_time].npy`` fields.
    :param channels: channel indices to unmix; None means every channel of
        the fields, as Make Masks unmixes whole stacks.
    :param load: reads one field into a channel-last array; ``np.load``.
        Plate-qualified controls instead use a bounded read-only NPY descriptor
        shared with their captured input hashes. A supplied ``unmix_matrix_file``
        bypasses control estimation on the explicitly supported calibrated route.
    :param allow_calibrated: allow an explicit ROI-aware 2D processing route.
    :param channel_positions: physical channel IDs mapped to packed positions.
    :param field_identities: captured source identities keyed by packed filename.
    :returns: a :class:`_UnmixPlan`, or None when ``unmix`` is off.
    :raises ValueError: for missing or unreadable controls or settings.
    """
    from .schema import parse_field_stem, parse_well

    per_plate = _validate_unmix_plate_scope(
        settings, allow_per_plate=bool(allow_calibrated and channel_positions is not None
                                      and field_identities is not None and channels is not None))
    supplied_matrix = _validate_unmix_matrix_scope(
        settings, allow_matrix=bool(allow_calibrated and channel_positions is not None
                                   and field_identities is not None and channels is not None))
    requested_calibration = bool(settings.get('unmix_calibration'))
    if requested_calibration and (not allow_calibrated or not settings.get('unmix', False)):
        from .qt.i18n import tr

        raise ValueError(tr("Calibrated spectral unmixing requires an explicit 2D live preview or calibrated V2 batch route; this processing path, including Measure, is not supported."))
    if not settings.get('unmix', False):
        return None
    if channels is not None:
        try:
            if isinstance(channels, (str, bytes)):
                raise ValueError
            selected = []
            for channel in channels:
                if np.ndim(channel) != 0:
                    raise ValueError
                if isinstance(channel, np.ndarray):
                    channel = channel.item()
                if isinstance(channel, (bool, np.bool_, complex, np.complexfloating, np.ndarray)):
                    raise ValueError
                index = int(channel)
                if not isinstance(channel, (str, bytes)) and channel != index:
                    raise ValueError
                if index < 0 or index in selected:
                    raise ValueError
                selected.append(index)
            if not selected:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            from .qt.i18n import tr

            raise ValueError(tr("Spectral unmixing requires distinct nonnegative integer channel indices.")) from None
        channels = tuple(selected)
    if supplied_matrix:
        return _prepare_supplied_matrix(settings, channels, field_identities)
    control_selection = None
    if per_plate:
        controls, control_selection = _plate_unmix_controls(settings, channels, field_identities)
    else:
        controls = _parse_unmix_controls(settings.get('unmix_controls'))
    if not controls:
        raise ValueError(
            'unmix is on but unmix_controls names no single-stain control '
            'wells; give them as channel:well[,well], for example 0:A01; 1:B01')
    percentile = float(settings.get('unmix_background_percentile', 5.0))
    if not 0.0 <= percentile < 100.0:
        raise ValueError('unmix_background_percentile must be at least 0 '
                         'and below 100')
    if requested_calibration and load is None:
        from functools import partial

        load = partial(np.load, mmap_mode='r', allow_pickle=False)
    else:
        load = load or np.load
    wanted = {dye: {parse_well(well, strict=True) for well in wells}
              for dye, wells in controls.items()}
    source = Path(source_dir)
    names = sorted(path.name for path in source.glob('*.npy')
                   if not path.name.startswith('.')) if source.is_dir() else []
    stems = {dye: [] for dye in controls}
    if field_identities is not None:
        from .qt.i18n import tr

        contexts = set()
        for name in names:
            identity = field_identities.get(name)
            if not isinstance(identity, dict) or not identity.get('plate') or not identity.get('well'):
                raise ValueError(tr("Calibrated batch inputs require explicit plate and well identities for every field."))
            contexts.add((str(identity['plate']), identity.get('time', 0), identity.get('z', 0)))
        if len(contexts) != 1:
            raise ValueError(tr("Calibrated batch controls and targets must belong to one plate and one acquisition time and plane."))
    for name in names:
        try:
            if field_identities is None:
                identity = parse_field_stem(name)
                well = (identity.rowID, identity.columnID)
            else:
                well = parse_well(field_identities[name]['well'], strict=True)
        except ValueError:
            continue
        for dye, wells in wanted.items():
            if (well in wells
                    and len(stems[dye]) < _UNMIX_MAX_FIELDS):
                stems[dye].append(name)
    missing = sorted(dye for dye, found in stems.items() if not found)
    if missing:
        raise ValueError(
            f'no field in {source} comes from the single-stain control wells '
            f'of channel {missing[0]}: {", ".join(controls[missing[0]])}')
    fields_by_dye = {}
    calibration = None
    calibrated_values = 0
    control_bytes = 0
    for dye, found in stems.items():
        fields_by_dye[dye] = []
        for name in found:
            checkpoint()
            context = (_plate_control_input(source / name, 512 * 1024 ** 2 - control_bytes)
                       if per_plate else nullcontext((np.asarray(load(source / name)), None)))
            with context as (field, facts):
                if requested_calibration and (field.ndim != 3 or not all(field.shape)
                        or field.shape[-1] > 16):
                    from .qt.i18n import tr

                    raise ValueError(tr("Calibrated unmixing requires a nonempty 2D channel-last image with at most 16,777,216 channel pixels."))
                if field.ndim < 3:
                    from .qt.i18n import tr

                    raise ValueError(tr("Spectral unmixing requires images with an explicit channel axis and at least three dimensions."))
                if channels is None:
                    channels = tuple(range(field.shape[-1]))
                positions = [channel_positions[c] if channel_positions is not None else c
                             for c in channels]
                outside = [c for c, position in zip(channels, positions)
                           if not 0 <= position < field.shape[-1]]
                if channel_positions is None and not 0 <= dye < field.shape[-1]:
                    outside.append(dye)
                if outside:
                    raise ValueError(
                        f'unmixing channel {outside[0]} is outside the '
                        f'{field.shape[-1]} channels of {name}')
                if dye not in channels:
                    raise ValueError(
                        f'unmix_controls names channel {dye}, which is not among '
                        f'the unmixed channels {list(channels)}')
                if requested_calibration and field.shape[0] * field.shape[1] * len(channels) > 16 * 1024 ** 2:
                    from .qt.i18n import tr

                    raise ValueError(tr("Calibrated unmixing requires a nonempty 2D channel-last image with at most 16,777,216 channel pixels."))
                if requested_calibration:
                    calibrated_values += field.shape[0] * field.shape[1] * len(channels)
                    if calibrated_values > _UNMIX_CALIBRATED_CONTROL_VALUES:
                        from .qt.i18n import tr

                        raise ValueError(tr("The selected spectral controls exceed the calibrated control-pixel budget; choose fewer or smaller control fields."))
                selected_field = field[..., positions]
                if requested_calibration:
                    if calibration is None:
                        calibration = _load_unmix_calibration(settings, channels)
                    selected_field, _ = calibration.apply(selected_field)
                fields_by_dye[dye].append(selected_field)
                if per_plate:
                    stem = Path(name).stem
                    if stem not in control_selection['sampled_fields']:
                        control_bytes += facts['input_size_bytes']
                        control_selection['sampled_fields'][stem] = dict(
                            facts, source_identity=json.loads(json.dumps(field_identities[name])))
    matrix = _estimate_bleed_through(fields_by_dye, channels)
    return _UnmixPlan(
        channels=tuple(int(channel) for channel in channels),
        matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        background_percentile=percentile, controls=controls,
        fields={dye: tuple(Path(name).stem for name in found)
                for dye, found in stems.items()}, calibration=calibration,
        control_selection=(json.dumps(control_selection, sort_keys=True, allow_nan=False)
                           if control_selection is not None else None))


def _describe_unmixing(plan):
    """Print the estimated bleed-through matrix, one row per channel."""
    print('Spectral unmixing: bleed-through matrix (row reads, column dye):')
    header = ''.join(f'{channel:>9}' for channel in plan.channels)
    print(f'{"":>6}{header}')
    for channel, row in zip(plan.channels, plan.matrix):
        print(f'{channel:>6}' + ''.join(f'{value:9.4f}' for value in row))


def _prepare_measure_unmixing(settings):
    """Estimate Measure's unmixing plan once and record it in ``settings``.

    The plan spans the measured ``channels`` of the merged fields in
    ``settings['src']``. Its record is stored as JSON under a private key,
    so the saved run settings carry the matrix and every worker unmixes with
    the same one, and is written to ``measurements/bleed_through.json``.

    :param settings: Measure settings, updated in place.
    :returns: the plan, or None when ``unmix`` is off.
    """
    if settings.get('unmix_calibration'):
        from .qt.i18n import tr

        raise ValueError(tr("Calibrated spectral unmixing requires an explicit 2D live preview or calibrated V2 batch route; this processing path, including Measure, is not supported."))
    settings.pop(_UNMIX_RECORD_KEY, None)
    if not settings.get('unmix', False):
        return None
    plan = _prepare_unmixing(settings, settings['src'],
                             channels=settings['channels'])
    if plan is None:
        return None
    record = plan.provenance()
    settings[_UNMIX_RECORD_KEY] = json.dumps(record)
    folder = Path(settings['src']).parent / 'measurements'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'bleed_through.json').write_text(json.dumps(record, indent=2), encoding="utf-8")
    _describe_unmixing(plan)
    return plan


def _apply_recorded_unmixing(channel_arrays, settings):
    """Unmix Measure's channel arrays with the matrix recorded in ``settings``.

    :param channel_arrays: the field's measured channels, channel last.
    :param settings: Measure settings holding the record, or not.
    :returns: the arrays unchanged without a record; otherwise unmixed, in
        the input's integer type, rounded and clipped, when it has one.
    """
    text = settings.get(_UNMIX_RECORD_KEY)
    if not text:
        return channel_arrays
    record = json.loads(text)
    out = _unmix(channel_arrays, record['matrix'],
                 record['background_percentile'])
    dtype = np.asarray(channel_arrays).dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(dtype)
    return out


_N2V_TRAIN_FIELDS = 8
_N2V_FOLDER = 'n2v'


@dataclass(frozen=True)
class _N2VPlan:
    """Trained Noise2Void checkpoints, one per segmentation channel.

    :param channels: the field's channel indices, in the order the session
        is handed them.
    :param checkpoints: ``{channel: checkpoint path}``.
    :param records: ``{channel: training record}``: the checkpoint's sha256,
        its losses, epochs, patch size, fields and device.
    """

    channels: tuple
    checkpoints: dict
    records: dict

    def provenance(self):
        """The plan as a JSON-safe record; paths are left out so a moved
        experiment still matches, the hashes identify the models."""
        return {
            'method': 'Noise2Void (N2V2, CAREamics), self-supervised',
            'stage': 'after illumination, before PSF and enhancement chain',
            'channels': [int(channel) for channel in self.channels],
            'models': {str(channel): {key: value for key, value
                                      in self.records[channel].items()
                                      if key not in ('checkpoint', 'seconds',
                                                     'device')}
                       for channel in self.channels},
        }

    def apply(self, image, denoise=None):
        """Denoise every channel of one field with its own model.

        :param image: YXC intensities, or planes of them stacked in front
            (ZYXC, TYXC), channel ``i`` being ``channels[i]``.
        :param denoise: :func:`spacr._segmentation_backends._n2v_denoise`,
            or a stand-in for tests.
        :returns: a float32 copy.
        """
        if denoise is None:
            from ._segmentation_backends import _n2v_denoise as denoise
        array = np.asarray(image)
        out = np.array(array, dtype=np.float32, copy=True)
        planes = out.reshape(-1, *out.shape[-3:])
        for position, channel in enumerate(self.channels):
            for plane in planes:
                checkpoint()
                plane[..., position] = denoise(plane[..., position],
                                               self.checkpoints[channel])
        return out


def _n2v_training_planes(source_dir, channel, load=None,
                         limit=_N2V_TRAIN_FIELDS):
    """One channel's planes from up to ``limit`` raw fields, evenly spaced
    through the sorted field names so every well and plate is sampled."""
    load = load or np.load
    names = sorted(path.name for path in Path(source_dir).glob('*.npy')
                   if not path.name.startswith('.'))
    if not names:
        raise ValueError(f'n2v_denoise found no field to train on in {source_dir}')
    picks = sorted({int(round(i)) for i in np.linspace(
        0, len(names) - 1, min(limit, len(names)))})
    planes, stems = [], []
    for index in picks:
        checkpoint()
        field = np.asarray(load(Path(source_dir) / names[index]))
        if not 0 <= channel < field.shape[-1]:
            raise ValueError(f'n2v_denoise channel {channel} is outside the '
                             f'{field.shape[-1]} channels of {names[index]}')
        stack = field[..., channel]
        planes.extend(np.asarray(stack, dtype=np.float32).reshape(
            -1, *stack.shape[-2:]))
        stems.append(Path(names[index]).stem)
    return planes, stems


def _prepare_n2v(settings, root, channels, *, stack_dir=None, load=None,
                 train=None):
    """Train, reuse or read the run's Noise2Void models.

    With ``n2v_model`` blank, each channel's model is trained on the run's
    own fields into ``<root>/n2v/channel_<c>.ckpt`` with its record beside
    it, and a checkpoint already there from the same epochs is reused, so a
    resumed run does not train again. With ``n2v_model`` set, that folder's
    ``channel_<c>.ckpt`` files are used as they are.

    :param settings: ``n2v_denoise``, ``n2v_model`` and ``n2v_epochs``.
    :param root: experiment root.
    :param channels: selected channel indices, in the session's order.
    :param stack_dir: raw fields to train on; ``root/stack`` when None.
    :param load: reads one field channel-last; ``np.load`` when None.
    :param train: :func:`spacr._segmentation_backends._n2v_train`, or a
        stand-in for tests.
    :returns: a :class:`_N2VPlan`, or None when ``n2v_denoise`` is off.
    :raises ValueError: for a missing model or bad settings.
    :raises ImportError: when training is needed and CAREamics is not
        installed.
    """
    if not _bool(settings.get('n2v_denoise', False)):
        return None
    epochs = int(settings.get('n2v_epochs', 20))
    if epochs < 1:
        raise ValueError('n2v_epochs must be at least 1')
    given = str(settings.get('n2v_model') or '').strip()
    folder = Path(given).expanduser() if given else Path(root) / _N2V_FOLDER
    checkpoints, records = {}, {}
    for channel in (int(c) for c in channels):
        path = folder / f'channel_{channel}.ckpt'
        sidecar = path.with_suffix('.json')
        record = None
        if path.is_file():
            try:
                record = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {}
            if not given and record.get('epochs') != epochs:
                record = None
        if record is None and given:
            raise ValueError(f'n2v_model has no model for channel {channel}: '
                             f'{path} does not exist')
        if record is None:
            if train is None:
                from ._segmentation_backends import _n2v_train as train
            source = stack_dir if stack_dir is not None else Path(root) / 'stack'
            planes, stems = _n2v_training_planes(source, channel, load)
            print(f'Noise2Void: training channel {channel} on {len(planes)} '
                  f'planes of {len(stems)} fields for {epochs} epochs')
            path.parent.mkdir(parents=True, exist_ok=True)
            record = dict(train(planes, str(path), epochs=epochs))
            record['fields'] = stems
            record['channel'] = channel
            _write_json(sidecar, record)
        record['sha256'] = _digest(path)
        checkpoints[channel] = str(path)
        records[channel] = record
    return _N2VPlan(channels=tuple(int(c) for c in channels),
                    checkpoints=checkpoints, records=records)


def _write_json(path, record):
    """Atomically replace one JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.n2v_')
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record_path(root):
    """Return the application record below an experiment's PSF directory."""
    return Path(root) / 'psf' / 'segmentation_application.json'


def _digest(path):
    """Hash durable file bytes with cancellation checks between read blocks."""
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checkpoint()
            result.update(chunk)
    return result.hexdigest()


def _configuration(plan, channels, pipeline_style, chain=None, unmix=None,
                   n2v=None):
    """Describe the processing stage and source-channel order for later reuse.

    The chain's record is written ONLY when a chain step is on, and the
    unmixing matrix only when unmixing is on and the Noise2Void models only
    when denoising is on, so a record made before any of them existed still matches a run that asks for neither.
    """
    record = {
        'version': 1, 'pipeline_style': pipeline_style,
        'channels': [int(channel) for channel in channels],
        'processing': plan.provenance() if plan else {'operation': 'none'},
        'stage': 'after illumination, before background/percentile normalization',
        'measurement_intensity_source': 'original persisted intensities',
    }
    if chain is not None:
        from .qt.detect_chain import provenance

        record['enhancement'] = provenance(chain)['enhancement']
    if unmix is not None:
        record['unmixing'] = unmix.provenance()
        if unmix.calibration is not None:
            record['coordinate_system'] = 'calibrated_reference_canvas'
            record['measurement_intensity_source'] = 'nonquantitative calibrated display; measurements unsupported'
            record['physical_channel_order'] = list(unmix.channels)
            record['stage'] = 'signed calibrated valid crop before a nonnegative model view; no illumination correction'
    if n2v is not None:
        record['n2v'] = n2v.provenance()
    return record


def _initialize_calibrated_output(root, settings):
    """Mark a fresh calibrated dataset as nonquantitative before any image write."""
    from .qt.i18n import tr

    root = Path(root)
    if not settings.get('unmix') or not settings.get('unmix_calibration'):
        raise ValueError(tr("Calibrated batch output requires spectral unmixing and an explicit calibration."))
    if root.exists() and any(root.iterdir()):
        raise ValueError(tr("Calibrated batch output must use a fresh empty destination; existing results are not resumed or replaced."))
    _write_json(_record_path(root), {
        'configuration': {
            'version': 1, 'pipeline_style': 'v2',
            'coordinate_system': 'calibrated_reference_canvas',
            'measurement_intensity_source': 'nonquantitative calibrated display; measurements unsupported',
        },
        'complete': False, 'fields': [], 'archives': {},
    })


def _validate_calibrated_batch_settings(settings):
    """Refuse processing that has no declared calibrated coordinate contract."""
    from .qt.i18n import tr
    from .utils import _validated_intensity_bounds

    if (settings.get('illumination_correction') or _bool(settings.get('n2v_denoise', False))
            or settings.get('timelapse')
            or prepare_psf(settings) is not None or prepare_chain(settings, None) is not None):
        raise ValueError(tr("Calibrated batch segmentation supports independent 2D fields without illumination, additional PSF, enhancement, denoising or timelapse processing."))
    if settings.get('image_qc_mode') == 'exclude':
        raise ValueError(tr("Calibrated batch segmentation does not support excluding image-QC fields; use report mode so every emitted field retains a declared coordinate frame."))
    raw = settings.get('object_filters')
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError:
            raise ValueError(tr("Calibrated batch segmentation supports size and border filters only; custom object filters are not supported.")) from None
    if raw is not None and (not isinstance(raw, dict) or any(raw.values())):
        raise ValueError(tr("Calibrated batch segmentation supports size and border filters only; custom object filters are not supported."))
    bounds = _validated_intensity_bounds(settings.get('cell_min_intensity'),
                                          settings.get('cell_max_intensity'))
    perimeter = settings.get('cell_perimeter_fraction', settings.get('cell_perimiter_fraction', 0))
    if any(bounds) or float(perimeter or 0) != 0:
        raise ValueError(tr("Calibrated batch segmentation does not support intensity or perimeter-merge filters; use size and border filters only."))


def validate_psf_resume(settings, root, channels, *, expected_fields):
    """Refuse V1 archive reuse after a PSF change or incomplete processing.

    :param settings: requested PSF, ``enhance_*`` and ``unmix`` settings;
        off also checks a prior PSF, chain or unmixing run.
    :param root: experiment root containing masks/ and psf/.
    :param channels: source intensity indices in archive order.
    :param expected_fields: exact field stems carried by the input archives.
    :returns: None for a compatible completed set or untouched legacy inputs.
    :raises ValueError: if archives cannot be proven to match this request.
    """
    fill_psf_settings(settings, root)
    plan = prepare_psf(settings)
    chain = prepare_chain(settings, plan)
    unmix = _prepare_unmixing(settings, Path(root) / 'stack')
    n2v = _prepare_n2v(settings, root, channels)
    path = _record_path(root)
    if (plan is None and chain is None and unmix is None and n2v is None
            and not path.exists()):
        return
    message = ('PSF or enhancement preprocessing does not match the saved '
               'Mask inputs. Enable preprocessing to rebuild the complete '
               'archive set from stack/ before generating masks.')
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record['configuration'] != _configuration(
                plan, channels, 'v1', chain, unmix, n2v)
                or not record['complete']
                or set(record['fields']) != set(expected_fields)):
            raise ValueError(message)
        archives = sorted((Path(root) / 'masks').glob('*.npz'))
        if {p.name: _digest(p) for p in archives} != record['archives']:
            raise ValueError(message)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(message) from exc


class _SegmentationPSFSession:
    """Track a complete application, including switching processing off.

    Holds the captured :class:`PSFPlan`, the enhancement chain of
    :func:`prepare_chain` and the unmixing plan. With a chain, every selected
    channel goes through :func:`apply_chain`, whose PSF stage is the same
    kernel in the chain's order; without one, the PSF path is exactly what
    it was. Unmixing runs first, on the whole raw field, through
    :meth:`unmix`; Noise2Void denoising, when on, opens :meth:`correct`.
    """

    def __init__(self, plan, root, channels, pipeline_style, chain=None,
                 unmix=None, n2v=None):
        """Start a captured application with an explicit incomplete record."""
        self.plan = plan
        self.chain = chain
        self.unmixing = unmix
        self.n2v = n2v
        self.root = Path(root)
        self.pipeline_style = pipeline_style
        if unmix is not None and unmix.matrix_source is not None:
            _persist_supplied_matrix(self.root, unmix)
        self.configuration = _configuration(plan, channels, pipeline_style,
                                            chain, unmix, n2v)
        self.fields = set()
        self.calibrated_fields = {}
        self.field_identities = {}
        self._write(False)

    @property
    def calibrated(self):
        """Whether fields require explicit calibrated crop and canvas transport."""
        return self.unmixing is not None and self.unmixing.calibration is not None

    def validate_calibrated_shape(self, shape):
        """Preflight source support before allocating calibrated pixels or a model."""
        return self.unmixing.calibration.valid_region(shape)

    def prepare_calibrated(self, field, field_id):
        """Capture one signed valid crop and its full reference-canvas identity."""
        from .qt.i18n import tr

        if not self.calibrated:
            raise ValueError(tr("No spectral calibration was supplied."))
        identity = self.field_identities.get(str(field_id))
        if not isinstance(identity, dict):
            raise ValueError(tr("Calibrated batch inputs require explicit plate and well identities for every field."))
        result = self.unmixing.apply_calibrated(field)
        result.record.update(
            mode='calibrated_batch_2d', coordinate_system='calibrated_reference_canvas',
            field_id=str(field_id), physical_channel_order=list(self.unmixing.channels),
            segmentation_view='nonnegative clipped view; not quantitative correction',
            source_identity=dict(identity),
        )
        field_keys = (
            'mode', 'coordinate_system', 'field_id', 'source_identity',
            'valid_roi_yx', 'source_origin_px', 'source_shape',
            'quantitative_output', 'negative_fraction_per_channel',
            'segmentation_view',
        )
        row = {key: result.record[key] for key in field_keys}
        row['configuration_sha256'] = hashlib.sha256(json.dumps(
            self.configuration, sort_keys=True, allow_nan=False,
        ).encode()).hexdigest()
        self.calibrated_fields[str(field_id)] = row
        self._write(False)
        return result

    def calibrated_display(self, field, result):
        """Embed only the nonnegative display crop, leaving invalid support blank."""
        canvas = np.zeros(np.shape(field), dtype=np.float32)
        canvas[result.valid_slices] = np.maximum(result.image, 0)
        return canvas

    def restore_calibrated(self, mask, result, field_id):
        """Remove whole boundary labels and restore exact IDs on the reference canvas."""
        from .qt.i18n import tr

        labels = np.asarray(mask)
        if (labels.ndim != 2 or labels.shape != result.image.shape[:2]
                or np.iscomplexobj(labels) or not np.isfinite(labels).all()
                or np.any(labels < 0) or np.any(labels > np.iinfo(np.uint16).max)
                or not np.equal(labels, np.floor(labels)).all()):
            raise ValueError(tr("Calibrated batch labels must match the valid crop and contain exact integer IDs from 0 to 65535."))
        boundary = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        boundary = boundary[boundary != 0]
        kept = np.where(np.isin(labels, boundary), 0, labels).astype(np.uint16)
        canvas = np.zeros(tuple(result.record['source_shape'][:2]), dtype=np.uint16)
        canvas[result.valid_slices] = kept
        result.record['excluded_boundary_label_ids'] = [int(value) for value in boundary]
        self.calibrated_fields[str(field_id)]['excluded_boundary_label_ids'] = result.record['excluded_boundary_label_ids']
        return canvas

    @property
    def processes(self):
        """Whether :meth:`correct` changes intensities, so callers make float copies."""
        return (self.plan is not None or self.chain is not None
                or self.unmixing is not None or self.n2v is not None)

    def unmix(self, field):
        """Unmix one whole raw field, or return it unchanged without a plan."""
        if self.unmixing is None:
            return field
        checkpoint()
        return self.unmixing.apply(field)

    def correct(self, image):
        """Process one private field, or return it unchanged when switching off.

        Noise2Void denoising, when on, comes first; then the chain, or the
        PSF alone.
        """
        checkpoint()
        if self.n2v is not None:
            image = self.n2v.apply(image)
        if self.chain is not None:
            return apply_chain(image, self.chain)
        return self.plan.apply(image) if self.plan else image

    def mark_completed(self, field_id, *, output_path=None, signed_path=None, mask_role=None):
        """Record a field only after its archive or combined stack is durable."""
        if self.calibrated:
            from .qt.i18n import tr

            if output_path is None or signed_path is None or str(field_id) not in self.calibrated_fields:
                raise ValueError(tr("Calibrated batch completion requires durable display, mask and signed-crop outputs."))
            row = self.calibrated_fields[str(field_id)]
            for key, path in (('display_mask', output_path), ('signed_crop', signed_path)):
                path = Path(path)
                relative = path.resolve().relative_to(self.root.resolve())
                row[key] = {'path': str(relative), 'sha256': _digest(path)}
            if mask_role is not None:
                if mask_role != 'cell':
                    raise _cell_record_error()
                identity = dict(row['source_identity'])
                if identity.get('source_dataset', self.configuration['source_dataset']) != self.configuration['source_dataset']:
                    raise _cell_record_error()
                identity.setdefault('source_dataset', self.configuration['source_dataset'])
                row['source_identity'] = identity
                row.update(measurement_schema_version=1, mask_role='cell',
                           mask_channel_position=len(self.unmixing.channels))
                row['binding_sha256'] = _cell_record_hash({
                    key: value for key, value in row.items() if key != 'binding_sha256'})
        self.fields.add(str(field_id))
        if self.calibrated:
            self._write(False)

    def finish(self, expected_fields):
        """Publish completion only for an exact set of committed field stems."""
        if self.fields != set(expected_fields):
            raise RuntimeError('PSF processing did not complete every field')
        self._write(True)

    def _write(self, complete):
        """Atomically replace the JSON record; hash V1 archives on completion."""
        path = _record_path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        archives = {}
        if complete and self.pipeline_style == 'v1':
            archives = {p.name: _digest(p) for p in
                        sorted((self.root / 'masks').glob('*.npz'))}
        record = {'configuration': self.configuration, 'complete': complete,
                  'fields': sorted(self.fields), 'archives': archives}
        if self.calibrated:
            record['calibrated_fields'] = self.calibrated_fields
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.psf_')
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _prepare_segmentation_psf(settings, root, channels, *, pipeline_style='v1',
                              stack_dir=None, load=None, source_channels=None,
                              field_identities=None, prepared_unmix=None):
    """Capture a kernel and start provenance, including a prior run's off switch.

    :param settings: PSF configuration accepted by :func:`prepare_psf` and
        the ``enhance_*`` chain settings accepted by :func:`prepare_chain`.
    :param root: experiment root; original image files remain untouched.
    :param channels: selected indices in persisted intensity-channel order.
    :param pipeline_style: V1 normalized archives or V2 combined output stacks.
    :param stack_dir: folder of the raw field stacks the unmixing controls
        are read from; ``root/stack`` when None.
    :param load: reads one raw field channel-last; ``np.load`` when None.
    :param source_channels: physical IDs in packed order for calibrated V2 fields.
    :param field_identities: captured physical identities for packed V2 filenames.
    :param prepared_unmix: supplied-matrix plan captured before output creation.
    :returns: a new session, or None for an untracked run with the PSF, the
        chain, unmixing and Noise2Void all off.
    """
    _validate_unmix_plate_scope(
        settings, allow_per_plate=bool(pipeline_style == 'v2' and source_channels is not None
                                      and field_identities is not None))
    supplied_matrix = _validate_unmix_matrix_scope(
        settings, allow_matrix=bool(pipeline_style == 'v2' and source_channels is not None
                                   and field_identities is not None))
    if settings.get('unmix_calibration'):
        from .qt.i18n import tr

        if (pipeline_style != 'v2' or source_channels is None or field_identities is None
                or not settings.get('unmix')):
            raise ValueError(tr("Calibrated batch segmentation requires organized 2D V2 fields with explicit physical channels and source identities."))
        _validate_calibrated_batch_settings(settings)
        raw_channels = tuple(source_channels)
        try:
            physical = tuple(int(value) for value in raw_channels)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(tr("Calibrated batch inputs require distinct nonnegative integer physical channel IDs.")) from None
        if (not physical or len(physical) > 16 or len(set(physical)) != len(physical)
                or any(isinstance(value, (bool, np.bool_)) or value != channel or channel < 0
                       for value, channel in zip(raw_channels, physical))):
            raise ValueError(tr("Calibrated batch inputs require distinct nonnegative integer physical channel IDs."))
        if prepared_unmix is not None:
            if (not supplied_matrix or prepared_unmix.matrix_source is None
                    or prepared_unmix.channels != physical):
                raise _matrix_input_error()
            source = json.loads(prepared_unmix.matrix_source)
            contexts = {(identity['plate'], identity['time'], identity['z'])
                        for identity in field_identities.values()}
            _validate_matrix_artifact(source['artifact'], physical,
                prepared_unmix.calibration.provenance(),
                float(settings.get('unmix_background_percentile', 5.0)), contexts)
            unmix = prepared_unmix
        else:
            unmix = _prepare_unmixing(
                settings, stack_dir if stack_dir is not None else Path(root) / 'merged',
                channels=physical, load=load, allow_calibrated=True,
                channel_positions={channel: index for index, channel in enumerate(physical)},
                field_identities=field_identities,
            )
        session = _SegmentationPSFSession(None, root, channels, pipeline_style, unmix=unmix)
        session.field_identities = dict(field_identities)
        source_roots = {str(Path(identity['source_dataset']).resolve())
                        for identity in field_identities.values()
                        if identity.get('source_dataset')}
        if len(source_roots) > 1:
            raise ValueError(tr("Calibrated batch fields must identify one original source dataset."))
        session.configuration['source_dataset'] = (next(iter(source_roots)) if source_roots else
            str(Path(stack_dir if stack_dir is not None else Path(root) / 'merged').resolve().parent))
        session.configuration['control_identities'] = {
            stem: dict(field_identities[stem + '.npy'])
            for stems in unmix.fields.values() for stem in stems
        }
        mapper = Path(root) / 'filename_map.csv'
        if mapper.is_file():
            session.configuration['filename_map_sha256'] = _digest(mapper)
        session._write(False)
        return session
    inferred = fill_psf_settings(settings, root)
    if inferred is not None:
        print('PSF calibration was not set; inferred values (source in brackets):')
        for line in describe_optics(inferred):
            print('  ' + line)
    plan = prepare_psf(settings)
    chain = prepare_chain(settings, plan)
    unmix = _prepare_unmixing(
        settings, stack_dir if stack_dir is not None else Path(root) / 'stack',
        load=load)
    if unmix is not None:
        _describe_unmixing(unmix)
    n2v = _prepare_n2v(settings, root, channels, stack_dir=stack_dir,
                       load=load)
    if (plan is None and chain is None and unmix is None and n2v is None
            and not _record_path(root).exists()):
        return None
    return _SegmentationPSFSession(plan, root, channels, pipeline_style, chain,
                                   unmix, n2v)



def _cell_record_error():
    """Return one headless localized refusal for unsupported or inconsistent records."""
    from .qt.i18n import tr

    return ValueError(tr(
        "Signed cell measurements require a completed cell-role dataset with matching signed float32 pixels, masks and calibration provenance. Regenerate older or inconsistent outputs.",
        language=os.environ.get('SPACR_LANGUAGE', 'en')))


def _cell_record_checkpoint(cancelled):
    """Honor process cancellation and the optional headless caller predicate."""
    checkpoint()
    if cancelled is not None and cancelled():
        from .cancellation import PipelineCancelled
        raise PipelineCancelled()


def _cell_record_identity_valid(identity):
    """Preserve typed acquisition IDs and the existing absent time/Z sentinels."""
    if (not isinstance(identity, dict)
            or any(not isinstance(identity.get(key), str) or not identity[key]
                   for key in ('source_dataset', 'plate', 'well'))):
        return False
    for key in ('field', 'time', 'z'):
        value = identity.get(key)
        if type(value) is int and value >= 0:
            continue
        if isinstance(value, str) and (value or key in ('time', 'z')):
            continue
        return False
    return True


def _cell_record_hash(value):
    """Bind JSON metadata without including mutable cache or pixel objects."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _cell_record_open(root_fd, relative):
    """Open a regular descendant through descriptor-relative nonsymlink components."""
    import stat

    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise _cell_record_error()
    directory = os.dup(root_fd)
    try:
        parts = relative.split('/')
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise _cell_record_error()
        return os.fdopen(descriptor, 'rb')
    finally:
        os.close(directory)


def _cell_record_stream_hash(stream, cancelled):
    """Stream exact bytes with bounded buffers and detect concurrent replacement writes."""
    before = os.fstat(stream.fileno())
    stream.seek(0)
    result = hashlib.sha256()
    while True:
        _cell_record_checkpoint(cancelled)
        block = stream.read(1024 ** 2)
        if not block:
            break
        result.update(block)
    after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after):
        raise _cell_record_error()
    return result.hexdigest(), identity(after)


def _cell_record_snapshot(stream, cancelled):
    """Capture at most 16 MiB of exact stable metadata before decoding JSON."""
    _cell_record_checkpoint(cancelled)
    before = os.fstat(stream.fileno())
    stream.seek(0)
    raw = stream.read(16 * 1024 ** 2 + 1)
    after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if len(raw) > 16 * 1024 ** 2 or identity(before) != identity(after):
        raise _cell_record_error()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), identity(after)


def _cell_record_map(stream, expected_shape):
    """Validate a bounded native float32 NPY header before mapping its payload read-only."""
    stream.seek(0)
    version = np.lib.format.read_magic(stream)
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0}
    if version not in readers:
        raise _cell_record_error()
    shape, fortran, dtype = readers[version](stream, max_header_size=16384)
    if (shape != tuple(expected_shape) or fortran or dtype != np.dtype(np.float32)
            or dtype.hasobject or any(type(size) is not int or size <= 0 for size in shape)
            or np.prod(shape, dtype=object) * dtype.itemsize > 128 * 1024 ** 2
            or os.fstat(stream.fileno()).st_size != stream.tell() + int(np.prod(shape, dtype=object)) * dtype.itemsize):
        raise _cell_record_error()
    return np.memmap(stream, dtype=dtype, mode='r', offset=stream.tell(), shape=shape, order='C')


def _validate_plate_control_record(config, rows):
    """Check optional plate evidence without relaxing older signed-cell records.

    :param config: captured calibrated dataset configuration.
    :param rows: validated per-field signed artifact records.
    :returns: None when absent or when all recorded plate-control facts agree.
    """
    unmixing = config['unmixing']
    if 'control_selection' not in unmixing:
        return
    try:
        selection = unmixing['control_selection']
        identities = {key: row['source_identity'] for key, row in rows.items()}
        identities.update(config['control_identities'])
        _, expected = _plate_unmix_controls(
            {'unmix_controls': selection['canonical_controls_by_plate']},
            config['physical_channel_order'], identities)
        if any(selection[key] != expected[key] for key in (
                'scope', 'plate', 'physical_channel_order', 'canonical_controls_by_plate',
                'controls_map_sha256', 'selected_controls')):
            raise _cell_record_error()
        if selection['selected_controls'] != unmixing['controls']:
            raise _cell_record_error()
        samples = selection['sampled_fields']
        chosen = unmixing['fields']
        if (set(chosen) != set(selection['selected_controls'])
                or any(not isinstance(values, list) or not 1 <= len(values) <= _UNMIX_MAX_FIELDS
                       or len(set(values)) != len(values) for values in chosen.values())):
            raise _cell_record_error()
        stems = [stem for values in chosen.values() for stem in values]
        if not isinstance(samples, dict) or set(samples) != set(stems) or len(stems) != len(set(stems)):
            raise _cell_record_error()
        from .schema import parse_well

        for dye, values in chosen.items():
            wanted = {parse_well(well, strict=True) for well in selection['selected_controls'][dye]}
            for stem in values:
                if parse_well(samples[stem]['source_identity']['well'], strict=True) not in wanted:
                    raise _cell_record_error()
        by_identity = {json.dumps(row['source_identity'], sort_keys=True, allow_nan=False): row for row in rows.values()}
        total_bytes = total_values = 0
        for stem, facts in samples.items():
            identity = facts['source_identity']
            if (identity != config['control_identities'][stem] or identity['plate'] != selection['plate']
                    or identity['source_dataset'] != config['source_dataset']):
                raise _cell_record_error()
            target = by_identity.get(json.dumps(identity, sort_keys=True, allow_nan=False))
            shape = facts['input_shape']
            dtype = np.dtype(facts['input_dtype'])
            size = facts['input_size_bytes']
            digest = facts['input_sha256']
            if (facts['input_stage'] != 'assembled_pre_unmixing'
                    or (target is not None and shape != target['source_shape'])
                    or len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape)
                    or shape[-1] != len(config['physical_channel_order'])
                    or np.prod(shape, dtype=object) > 16 * 1024 ** 2
                    or dtype.hasobject or dtype.kind not in 'uif' or dtype.itemsize > 8
                    or dtype.str != facts['input_dtype'] or type(size) is not int
                    or not 0 < size - int(np.prod(shape, dtype=object)) * dtype.itemsize <= 16384
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(character not in '0123456789abcdef' for character in digest)):
                raise _cell_record_error()
            total_bytes += size
            total_values += int(np.prod(shape, dtype=object))
        if total_bytes > 512 * 1024 ** 2 or total_values > _UNMIX_CALIBRATED_CONTROL_VALUES:
            raise _cell_record_error()
    except (KeyError, TypeError, ValueError, OverflowError):
        raise _cell_record_error() from None


def _validate_supplied_matrix_record(root_fd, config, rows):
    """Validate the copied artifact instead of reopening its historical external path.

    :param root_fd: owned descriptor for the calibrated output directory.
    :param config: existing version-1 calibrated application configuration.
    :param rows: bound target field records.
    :returns: copied bundle identity, or None for unchanged historical records.
    """
    unmixing = config['unmixing']
    if 'matrix_source' not in unmixing:
        if unmixing.get('method') == 'supplied_matrix':
            raise _cell_record_error()
        return None
    try:
        source = unmixing['matrix_source']
        if (not isinstance(source, dict) or set(source) != {
                'kind', 'version', 'path', 'sha256', 'size_bytes', 'artifact',
                'acquisition_compatibility_verified'}
                or source['kind'] != 'supplied_matrix' or type(source['version']) is not int
                or source['version'] != 1 or source['path'] != 'psf/spectral_matrix_input.json'
                or source['acquisition_compatibility_verified'] is not False
                or unmixing['method'] != 'supplied_matrix' or unmixing['controls'] != {}
                or unmixing['fields'] != {} or 'control_selection' in unmixing
                or type(source['size_bytes']) is not int):
            raise _cell_record_error()
        contexts = {(row['source_identity']['plate'], row['source_identity']['time'],
                     row['source_identity']['z']) for row in rows.values()}
        with _cell_record_open(root_fd, source['path']) as stream:
            raw, payload, digest, identity = _matrix_input_snapshot(stream)
        if (len(raw) != source['size_bytes'] or digest != source['sha256']
                or payload != source['artifact']):
            raise _cell_record_error()
        matrix = _validate_matrix_artifact(payload, config['physical_channel_order'],
            unmixing['calibration'], unmixing['background_percentile'], contexts)
        if not np.array_equal(matrix, np.asarray(unmixing['matrix'], dtype=np.float64)):
            raise _cell_record_error()
        return identity
    except (OSError, KeyError, TypeError, ValueError, OverflowError, RecursionError, np.linalg.LinAlgError):
        raise _cell_record_error() from None


class _CalibratedCellDataset:
    """Own bounded one-field maps and exact completed-record provenance for a reader."""

    def __init__(self, source, channels, cancelled):
        """Capture metadata and validate every field before exposing any measurements."""
        if (not hasattr(os, 'O_DIRECTORY') or not hasattr(os, 'O_NOFOLLOW')
                or os.open not in os.supports_dir_fd):
            from .qt.i18n import tr

            raise ValueError(tr(
                "Signed cell measurement loading requires POSIX descriptor-relative file access without symbolic links; this platform is not supported.",
                language=os.environ.get('SPACR_LANGUAGE', 'en')))
        self._root = Path(source).expanduser().resolve(strict=True)
        self._root_fd = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._cancelled = cancelled
        self._active = None
        self._iterator = None
        self._closed = False
        try:
            with _cell_record_open(self._root_fd, 'psf/segmentation_application.json') as stream:
                self._record, self.record_sha256, self._record_identity = _cell_record_snapshot(stream, cancelled)
            record = self._record
            if (not isinstance(record, dict) or record.get('complete') is not True
                    or not isinstance(record.get('fields'), list) or not 1 <= len(record['fields']) <= 8192
                    or any(not isinstance(key, str) or not key for key in record['fields'])
                    or len(set(record['fields'])) != len(record['fields'])
                    or not isinstance(record.get('calibrated_fields'), dict)
                    or set(record['fields']) != record['calibrated_fields'].keys()):
                raise _cell_record_error()
            self._configuration = record['configuration']
            config = self._configuration
            unmixing = config['unmixing']
            calibration = unmixing['calibration']
            physical = config['physical_channel_order']
            if (type(config.get('version')) is not int or config.get('version') != 1 or config.get('pipeline_style') != 'v2'
                    or config.get('coordinate_system') != 'calibrated_reference_canvas'
                    or not isinstance(physical, list) or not 1 <= len(physical) <= 16
                    or any(type(channel) is not int or channel < 0 for channel in physical)
                    or len(set(physical)) != len(physical) or unmixing.get('channels') != physical
                    or type(calibration.get('version')) is not int or calibration.get('version') != 1
                    or calibration.get('axes') != 'YX'):
                raise _cell_record_error()
            self.physical_channel_order = tuple(physical)
            self.channels = tuple(physical if channels is None else channels)
            if (not self.channels or len(set(self.channels)) != len(self.channels)
                    or any(type(channel) is not int or channel not in physical for channel in self.channels)):
                raise _cell_record_error()
            spacing = calibration['pixel_size_um']
            if (not isinstance(spacing, list) or len(spacing) != 2
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not np.isfinite(value) or value <= 0 for value in spacing)):
                raise _cell_record_error()
            self.pixel_size_um = tuple(float(value) for value in spacing)
            entries = calibration['channels']
            if (not isinstance(entries, list) or len(entries) != len(physical)
                    or [entry['channel'] for entry in entries] != physical
                    or type(calibration['reference_channel']) is not int
                    or calibration['reference_channel'] not in physical):
                raise _cell_record_error()
            self._matching_shapes = []
            shapes = []
            for entry in entries:
                shape = np.shape(entry['psf'])
                if (len(shape) != 2 or any(size <= 0 or size > 31 or size % 2 != 1 for size in shape)
                        or entry['psf_origin_px'] != [size // 2 for size in shape]
                        or entry['psf_pixel_size_um'] != spacing):
                    raise _cell_record_error()
                kernel = np.asarray(entry['psf'], dtype=np.float64)
                shift = entry['shift_to_reference_px']
                if (not np.isfinite(kernel).all() or np.any(kernel < 0)
                        or not np.isclose(kernel.sum(), 1, rtol=1e-6, atol=1e-12)
                        or hashlib.sha256(kernel.tobytes()).hexdigest() != entry['normalized_psf_sha256']
                        or not isinstance(shift, list) or len(shift) != 2
                        or any(type(value) is not int or abs(value) > 4096 for value in shift)
                        or (entry['channel'] == calibration['reference_channel'] and shift != [0, 0])):
                    raise _cell_record_error()
                shapes.append(shape)
            from scipy.signal import convolve2d

            common_shape = tuple(1 + sum(shape[axis] - 1 for shape in shapes) for axis in (0, 1))
            if (max(common_shape) > 127 or calibration.get('common_psf_support_yx') != list(common_shape)
                    or np.shape(calibration.get('common_psf')) != common_shape):
                raise _cell_record_error()
            common = np.ones((1, 1), dtype=np.float64)
            for entry in entries:
                common = convolve2d(common, np.asarray(entry['psf'], dtype=np.float64), mode='full')
            if not np.array_equal(common, np.asarray(calibration['common_psf'], dtype=np.float64)):
                raise _cell_record_error()
            self._matching_shapes = [tuple(1 + sum(shape[axis] - 1 for j, shape in enumerate(shapes) if j != i)
                                                 for axis in (0, 1)) for i in range(len(shapes))]
            matrix = np.asarray(unmixing['matrix'], dtype=np.float64)
            if (matrix.shape != (len(physical), len(physical)) or not np.isfinite(matrix).all()
                    or np.any(matrix < 0) or not np.all(np.diag(matrix) == 1)
                    or not np.isfinite(np.linalg.cond(matrix))):
                raise _cell_record_error()
            self._configuration_sha256 = _cell_record_hash(config)
            identities, physical_fields, paths = set(), set(), set()
            self._artifact_owners = {}
            if not isinstance(config.get('source_dataset'), str) or not config['source_dataset']:
                raise _cell_record_error()
            for field_id in record['fields']:
                row = record['calibrated_fields'][field_id]
                identity = row['source_identity']
                if (not _cell_record_identity_valid(identity)
                        or identity['source_dataset'] != config['source_dataset']):
                    raise _cell_record_error()
                physical_identity = tuple(identity[key] for key in ('source_dataset', 'plate', 'well', 'field', 'time', 'z'))
                if physical_identity in physical_fields:
                    raise _cell_record_error()
                physical_fields.add(physical_identity)
                for key in ('signed_crop', 'display_mask'):
                    path = row[key]['path']
                    if not isinstance(path, str) or path in paths:
                        raise _cell_record_error()
                    paths.add(path)
                with self._field(field_id):
                    identities.add((identity['plate'], identity['time'], identity['z']))
            if len(identities) != 1:
                raise _cell_record_error()
            _validate_plate_control_record(config, record['calibrated_fields'])
            self._matrix_identity = _validate_supplied_matrix_record(
                self._root_fd, config, record['calibrated_fields'])
            self._metadata_current(full=True)
        except BaseException:
            self.close()
            raise

    def provenance(self):
        """Return an independent JSON-safe snapshot of the validated complete record."""
        return json.loads(json.dumps(self._record))

    def _metadata_current(self, *, full=False):
        """Check descriptor identity per field and exact metadata bytes at final publication."""
        if self._matrix_identity is not None:
            source = self._configuration['unmixing']['matrix_source']
            with _cell_record_open(self._root_fd, source['path']) as stream:
                stat = os.fstat(stream.fileno())
                if self._matrix_identity != (stat.st_dev, stat.st_ino, stat.st_size,
                                             stat.st_mtime_ns, stat.st_ctime_ns):
                    raise _cell_record_error()
            if full and self._matrix_identity != _validate_supplied_matrix_record(
                    self._root_fd, self._configuration, self._record['calibrated_fields']):
                raise _cell_record_error()
        with _cell_record_open(self._root_fd, 'psf/segmentation_application.json') as stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if identity != self._record_identity:
                raise _cell_record_error()
            if full:
                digest, identity = _cell_record_stream_hash(stream, self._cancelled)
                if digest != self.record_sha256 or identity != self._record_identity:
                    raise _cell_record_error()

    @contextmanager
    def _field(self, field_id):
        """Validate a single matching signed crop and cell canvas, closing every mapping."""
        _cell_record_checkpoint(self._cancelled)
        row = self._record['calibrated_fields'][field_id]
        source_shape = row['source_shape']
        roi = row['valid_roi_yx']
        identity = row['source_identity']
        count = len(self.physical_channel_order)
        if (row.get('measurement_schema_version') != 1 or type(row.get('measurement_schema_version')) is not int
                or row.get('mask_role') != 'cell' or type(row.get('mask_channel_position')) is not int
                or row.get('mask_channel_position') != count
                or row.get('field_id') != field_id or row.get('mode') != 'calibrated_batch_2d'
                or row.get('coordinate_system') != 'calibrated_reference_canvas'
                or row.get('configuration_sha256') != self._configuration_sha256
                or row.get('binding_sha256') != _cell_record_hash({key: value for key, value in row.items() if key != 'binding_sha256'})
                or not isinstance(source_shape, list) or len(source_shape) != 3
                or any(type(size) is not int or size <= 0 for size in source_shape)
                or source_shape[2] != count or np.prod(source_shape, dtype=object) > 16 * 1024 ** 2
                or not isinstance(roi, list) or len(roi) != 2
                or any(not isinstance(part, list) or len(part) != 2 or any(type(v) is not int for v in part)
                       or not 0 <= part[0] < part[1] <= source_shape[axis] for axis, part in enumerate(roi))
                or not isinstance(row.get('source_origin_px'), list)
                or any(type(value) is not int for value in row['source_origin_px'])
                or row['source_origin_px'] != [roi[0][0], roi[1][0]]
                or not _cell_record_identity_valid(identity)):
            raise _cell_record_error()
        height, width = source_shape[:2]
        top, left, bottom, right = 0, 0, height, width
        calibration = self._configuration['unmixing']['calibration']
        for entry, shape in zip(calibration['channels'], self._matching_shapes):
            dy, dx = entry['shift_to_reference_px']
            ry, rx = (size // 2 for size in shape)
            top, left = max(top, dy + ry), max(left, dx + rx)
            bottom, right = min(bottom, height + dy - ry), min(right, width + dx - rx)
        if roi != [[top, bottom], [left, right]]:
            raise _cell_record_error()
        excluded = row['excluded_boundary_label_ids']
        if (not isinstance(excluded, list) or any(type(value) is not int or not 0 < value <= 65535 for value in excluded)
                or excluded != sorted(set(excluded))):
            raise _cell_record_error()
        streams, maps = [], []
        self._active = (streams, maps)
        try:
            crop_shape = (bottom - top, right - left, count)
            for key, shape in (('signed_crop', crop_shape), ('display_mask', (height, width, count + 1))):
                artifact = row[key]
                stream = _cell_record_open(self._root_fd, artifact['path'])
                streams.append((stream, key, None))
                stat = os.fstat(stream.fileno())
                artifact_id = (stat.st_dev, stat.st_ino)
                owner = (field_id, key)
                if artifact_id in self._artifact_owners and self._artifact_owners[artifact_id] != owner:
                    raise _cell_record_error()
                self._artifact_owners[artifact_id] = owner
                if os.fstat(stream.fileno()).st_size > 128 * 1024 ** 2 + 16384:
                    raise _cell_record_error()
                digest, original = _cell_record_stream_hash(stream, self._cancelled)
                if digest != artifact['sha256']:
                    raise _cell_record_error()
                streams[-1] = (stream, key, original)
                maps.append(_cell_record_map(stream, shape))
            signed, canvas = maps
            labels = canvas[..., -1]
            for start in range(0, height, 64):
                _cell_record_checkpoint(self._cancelled)
                block = canvas[start:start + 64]
                values = labels[start:start + 64]
                if (not np.isfinite(block).all() or np.any(block[..., :-1] < 0)
                        or np.any(values < 0) or np.any(values > 65535)
                        or not np.equal(values, np.floor(values)).all()
                        or np.isin(values, excluded).any()):
                    raise _cell_record_error()
                global_rows = np.arange(start, min(start + 64, height))
                outside = (global_rows < top) | (global_rows >= bottom)
                if (np.any(block[outside]) or np.any(block[:, :left]) or np.any(block[:, right:])):
                    raise _cell_record_error()
            for start in range(0, bottom - top, 64):
                _cell_record_checkpoint(self._cancelled)
                pixels = signed[start:start + 64]
                if (not np.isfinite(pixels).all() or not np.array_equal(
                        canvas[top + start:min(top + start + 64, bottom), left:right, :-1], np.maximum(pixels, 0))):
                    raise _cell_record_error()
            region_labels = labels[top:bottom, left:right]
            if any(np.any(edge) for edge in (region_labels[0], region_labels[-1], region_labels[:, 0], region_labels[:, -1])):
                raise _cell_record_error()
            fractions = row['negative_fraction_per_channel']
            if not isinstance(fractions, dict) or fractions.keys() != {str(channel) for channel in self.physical_channel_order}:
                raise _cell_record_error()
            for index, channel in enumerate(self.physical_channel_order):
                value = fractions[str(channel)]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise _cell_record_error()
                expected = float(value)
                actual = sum(np.count_nonzero(signed[start:start + 64, :, index] < 0)
                             for start in range(0, bottom - top, 64)) / ((bottom - top) * (right - left))
                if not np.isfinite(expected) or expected != actual:
                    raise _cell_record_error()
            cell_labels = np.array(region_labels, dtype=np.uint16, copy=True)
            cell_labels.flags.writeable = False
            try:
                yield dict(field_id=field_id, signed=signed, labels=cell_labels,
                           physical_channel_order=self.physical_channel_order,
                           channel_positions={channel: index for index, channel in enumerate(self.physical_channel_order)},
                           source_origin_px=tuple(row['source_origin_px']), source_shape=tuple(source_shape),
                           pixel_size_um=self.pixel_size_um, source_identity=dict(identity),
                           provenance=json.loads(json.dumps(row)))
            finally:
                for stream, key, original in streams:
                    digest, current = _cell_record_stream_hash(stream, self._cancelled)
                    with _cell_record_open(self._root_fd, row[key]['path']) as replacement:
                        stat = os.fstat(replacement.fileno())
                        identity_now = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                    if current != original or identity_now != original or digest != row[key]['sha256']:
                        raise _cell_record_error()
        finally:
            for array in maps:
                array._mmap.close()
            for stream, _, _ in streams:
                stream.close()
            self._active = None

    @property
    def configuration(self):
        """Return captured configuration without exposing mutable validation state."""
        return json.loads(json.dumps(self._configuration))

    def fields(self):
        """Borrow one iterator whose maps remain owned by the dataset context."""
        if self._closed or self._iterator is not None:
            raise _cell_record_error()
        self._iterator = self._iter_fields()
        return self._iterator

    def _iter_fields(self):
        """Recheck each borrowed field and metadata before advancing or early closure."""
        try:
            for field_id in self._record['fields']:
                self._metadata_current()
                with self._field(field_id) as field:
                    yield field
                self._metadata_current()
        finally:
            self._metadata_current(full=True)

    def close(self):
        """Drain the borrowed iterator and close maps even after failure or cancellation."""
        if self._closed:
            return
        try:
            if self._iterator is not None:
                self._iterator.close()
        finally:
            self._closed = True
            if self._active is not None:
                streams, maps = self._active
                for array in maps:
                    array._mmap.close()
                for stream, _, _ in streams:
                    stream.close()
                self._active = None
            os.close(self._root_fd)


@contextmanager
def _open_calibrated_cell_dataset(source, *, channels=None, cancelled=None):
    """Open completed cell-role records on POSIX with borrowed read-only field maps."""
    try:
        dataset = _CalibratedCellDataset(source, channels, cancelled)
    except (KeyError, TypeError, OSError, json.JSONDecodeError, OverflowError) as error:
        raise _cell_record_error() from error
    try:
        yield dataset
    except BaseException as caller_error:
        try:
            dataset.close()
        except BaseException as cleanup_error:
            raise caller_error from cleanup_error
        raise
    else:
        dataset.close()
