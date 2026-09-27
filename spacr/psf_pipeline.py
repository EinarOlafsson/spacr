"""Opt-in PSF and enhancement-chain preparation and provenance for segmentation inputs.

Kernels are captured once per run. Only private intensity arrays are processed;
raw stacks and mask labels are never rewritten by this module. V1 reuse checks
both the requested kernel/settings and the exact completed archive bytes.

THE ENHANCEMENT CHAIN TRAVELS THE SAME ROUTE. Make Masks' chain
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

SPECTRAL UNMIXING COMES FIRST. With ``unmix`` on, a bleed-through matrix is
estimated from the single-stain control wells ``unmix_controls`` names and
every raw field is unmixed across all its channels before illumination
correction, the PSF and the chain; the matrix joins the same record. Measure
estimates its own over the measured channels and unmixes each field before
its preprocessing hooks, recording the matrix in the saved run settings and
``measurements/bleed_through.json``.

SELF-SUPERVISED DENOISING COMES BEFORE THE CHAIN. With ``n2v_denoise`` on,
a Noise2Void (N2V2) model per segmentation channel is trained by CAREamics,
in its own environment, on a sample of the run's own raw fields -- no clean
images are needed -- or read from ``n2v_model``. Each field's selected
channels are denoised after illumination correction and before the PSF and
the chain, while the noise is still independent from pixel to pixel, which
is what Noise2Void assumes. The checkpoints' hashes and the training record
join the same provenance record.
"""
from __future__ import annotations

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
_UNMIX_MAX_CONDITION = 1e6


@dataclass(frozen=True)
class _UnmixPlan:
    """A bleed-through matrix estimated from single-stain control wells.

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
    """

    channels: tuple
    matrix: tuple
    background_percentile: float
    controls: dict
    fields: dict

    def provenance(self):
        """The plan as a JSON-safe record for the run's provenance."""
        return {
            'method': 'single-stain linear unmixing',
            'channels': [int(channel) for channel in self.channels],
            'matrix': [[float(value) for value in row] for row in self.matrix],
            'background_percentile': float(self.background_percentile),
            'controls': {str(dye): list(wells)
                         for dye, wells in sorted(self.controls.items())},
            'fields': {str(dye): list(stems)
                       for dye, stems in sorted(self.fields.items())},
        }

    def apply(self, image):
        """Unmix the plan's channels of one field, the other planes untouched.

        :param image: YXC or ZYXC intensities holding every plan channel.
        :returns: float32 copy with the plan's channels unmixed.
        """
        array = np.asarray(image)
        out = np.array(array, dtype=np.float32, copy=True)
        channels = list(self.channels)
        out[..., channels] = _unmix(array[..., channels], self.matrix,
                                    self.background_percentile)
        return out


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


def _unmix(image, matrix, background_percentile):
    """Solve every pixel of ``image`` for its dye amounts with ``matrix``.

    :param image: array whose last axis holds the matrix's channels in order.
    :param matrix: square bleed-through matrix.
    :param background_percentile: per-plane percentile taken as background;
        it is removed before solving and added back after, and the result is
        clipped at zero.
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
    return np.clip(solved, 0.0, None).reshape(array.shape).astype(np.float32)


def _prepare_unmixing(settings, source_dir, channels=None, load=None):
    """Estimate the run's unmixing plan from its single-stain control wells.

    :param settings: ``unmix``, ``unmix_controls`` and
        ``unmix_background_percentile``.
    :param source_dir: folder of ``plate_well_field[_time].npy`` fields.
    :param channels: channel indices to unmix; None means every channel of
        the fields, as Make Masks unmixes whole stacks.
    :param load: reads one field into a channel-last array; ``np.load``.
    :returns: a :class:`_UnmixPlan`, or None when ``unmix`` is off.
    :raises ValueError: for missing or unreadable controls or settings.
    """
    from .schema import parse_field_stem, parse_well

    if not settings.get('unmix', False):
        return None
    controls = _parse_unmix_controls(settings.get('unmix_controls'))
    if not controls:
        raise ValueError(
            'unmix is on but unmix_controls names no single-stain control '
            'wells; give them as channel:well[,well], for example 0:A01; 1:B01')
    percentile = float(settings.get('unmix_background_percentile', 5.0))
    if not 0.0 <= percentile < 100.0:
        raise ValueError('unmix_background_percentile must be at least 0 '
                         'and below 100')
    load = load or np.load
    wanted = {dye: {parse_well(well, strict=True) for well in wells}
              for dye, wells in controls.items()}
    source = Path(source_dir)
    names = sorted(path.name for path in source.glob('*.npy')
                   if not path.name.startswith('.')) if source.is_dir() else []
    stems = {dye: [] for dye in controls}
    for name in names:
        try:
            identity = parse_field_stem(name)
        except ValueError:
            continue
        for dye, wells in wanted.items():
            if ((identity.rowID, identity.columnID) in wells
                    and len(stems[dye]) < _UNMIX_MAX_FIELDS):
                stems[dye].append(name)
    missing = sorted(dye for dye, found in stems.items() if not found)
    if missing:
        raise ValueError(
            f'no field in {source} comes from the single-stain control wells '
            f'of channel {missing[0]}: {", ".join(controls[missing[0]])}')
    fields_by_dye = {}
    for dye, found in stems.items():
        fields_by_dye[dye] = []
        for name in found:
            checkpoint()
            field = np.asarray(load(source / name))
            if channels is None:
                channels = tuple(range(field.shape[-1]))
            outside = [c for c in (*channels, dye) if not 0 <= c < field.shape[-1]]
            if outside:
                raise ValueError(
                    f'unmixing channel {outside[0]} is outside the '
                    f'{field.shape[-1]} channels of {name}')
            if dye not in channels:
                raise ValueError(
                    f'unmix_controls names channel {dye}, which is not among '
                    f'the unmixed channels {list(channels)}')
            fields_by_dye[dye].append(field[..., list(channels)])
    matrix = _estimate_bleed_through(fields_by_dye, channels)
    return _UnmixPlan(
        channels=tuple(int(channel) for channel in channels),
        matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        background_percentile=percentile, controls=controls,
        fields={dye: tuple(Path(name).stem for name in found)
                for dye, found in stems.items()})


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
    settings.pop(_UNMIX_RECORD_KEY, None)
    plan = _prepare_unmixing(settings, settings['src'],
                             channels=tuple(int(c) for c in settings['channels']))
    if plan is None:
        return None
    record = plan.provenance()
    settings[_UNMIX_RECORD_KEY] = json.dumps(record)
    folder = Path(settings['src']).parent / 'measurements'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'bleed_through.json').write_text(json.dumps(record, indent=2))
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
                record = json.loads(sidecar.read_text())
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
    if n2v is not None:
        record['n2v'] = n2v.provenance()
    return record


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
        record = json.loads(path.read_text())
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
        self.configuration = _configuration(plan, channels, pipeline_style,
                                            chain, unmix, n2v)
        self.fields = set()
        self._write(False)

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

    def mark_completed(self, field_id):
        """Record a field only after its archive or combined stack is durable."""
        self.fields.add(str(field_id))

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
                              stack_dir=None, load=None):
    """Capture a kernel and start provenance, including a prior run's off switch.

    :param settings: PSF configuration accepted by :func:`prepare_psf` and
        the ``enhance_*`` chain settings accepted by :func:`prepare_chain`.
    :param root: experiment root; original image files remain untouched.
    :param channels: selected indices in persisted intensity-channel order.
    :param pipeline_style: V1 normalized archives or V2 combined output stacks.
    :param stack_dir: folder of the raw field stacks the unmixing controls
        are read from; ``root/stack`` when None.
    :param load: reads one raw field channel-last; ``np.load`` when None.
    :returns: a new session, or None for an untracked run with the PSF, the
        chain, unmixing and Noise2Void all off.
    """
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
