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

    :param settings: the run's ``psf_*`` and ``enhance_*`` settings.
    :returns: True when either is switched on or cannot be read.
    """
    if settings.get('psf_operation', 'none') != 'none':
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


def _configuration(plan, channels, pipeline_style, chain=None):
    """Describe the processing stage and source-channel order for later reuse.

    The chain's record is written ONLY when a chain step is on, so a record
    made before the chain existed still matches a run that asks for none.
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
    return record


def validate_psf_resume(settings, root, channels, *, expected_fields):
    """Refuse V1 archive reuse after a PSF change or incomplete processing.

    :param settings: requested PSF and ``enhance_*`` settings; off also
        checks a prior PSF or chain run.
    :param root: experiment root containing masks/ and psf/.
    :param channels: source intensity indices in archive order.
    :param expected_fields: exact field stems carried by the input archives.
    :returns: None for a compatible completed set or untouched legacy inputs.
    :raises ValueError: if archives cannot be proven to match this request.
    """
    fill_psf_settings(settings, root)
    plan = prepare_psf(settings)
    chain = prepare_chain(settings, plan)
    path = _record_path(root)
    if plan is None and chain is None and not path.exists():
        return
    message = ('PSF or enhancement preprocessing does not match the saved '
               'Mask inputs. Enable preprocessing to rebuild the complete '
               'archive set from stack/ before generating masks.')
    try:
        record = json.loads(path.read_text())
        if (record['configuration'] != _configuration(plan, channels, 'v1', chain)
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

    Holds the captured :class:`PSFPlan` and the enhancement chain of
    :func:`prepare_chain`. With a chain, every selected channel goes through
    :func:`apply_chain`, whose PSF stage is the same kernel in the chain's
    order; without one, the PSF path is exactly what it was.
    """

    def __init__(self, plan, root, channels, pipeline_style, chain=None):
        """Start a captured application with an explicit incomplete record."""
        self.plan = plan
        self.chain = chain
        self.root = Path(root)
        self.pipeline_style = pipeline_style
        self.configuration = _configuration(plan, channels, pipeline_style, chain)
        self.fields = set()
        self._write(False)

    @property
    def processes(self):
        """Whether :meth:`correct` changes intensities, so callers make float copies."""
        return self.plan is not None or self.chain is not None

    def correct(self, image):
        """Process one private field, or return it unchanged when switching off."""
        checkpoint()
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


def _prepare_segmentation_psf(settings, root, channels, *, pipeline_style='v1'):
    """Capture a kernel and start provenance, including a prior run's off switch.

    :param settings: PSF configuration accepted by :func:`prepare_psf` and
        the ``enhance_*`` chain settings accepted by :func:`prepare_chain`.
    :param root: experiment root; original image files remain untouched.
    :param channels: selected indices in persisted intensity-channel order.
    :param pipeline_style: V1 normalized archives or V2 combined output stacks.
    :returns: a new session, or None for an untracked run with the PSF and
        the chain both off.
    """
    inferred = fill_psf_settings(settings, root)
    if inferred is not None:
        print('PSF calibration was not set; inferred values (source in brackets):')
        for line in describe_optics(inferred):
            print('  ' + line)
    plan = prepare_psf(settings)
    chain = prepare_chain(settings, plan)
    if plan is None and chain is None and not _record_path(root).exists():
        return None
    return _SegmentationPSFSession(plan, root, channels, pipeline_style, chain)
