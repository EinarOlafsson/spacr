"""Opt-in PSF preparation and provenance for segmentation inputs.

Kernels are captured once per run. Only private intensity arrays are processed;
raw stacks and mask labels are never rewritten by this module. V1 reuse checks
both the requested kernel/settings and the exact completed archive bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .point_spread import PSF, apply_psf, gaussian_psf, load_psf, _spacing
from .cancellation import checkpoint


__all__ = ['PSFPlan', 'prepare_psf', 'validate_psf_resume']


def _check_cancel():
    """Bridge the calling pipeline's cancellation token into the PSF engine."""
    checkpoint()
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

    def apply(self, image):
        """Process one spatial image with an explicit last channel axis.

        :param image: YXC or ZYXC intensity array, never a merged label stack.
        :returns: float32 processed intensities without source mutation.
        """
        return apply_psf(
            image, self.kernel, operation=self.operation,
            image_sampling_um=self.sampling_um, iterations=self.iterations,
            channel_axis=-1, cancel=_check_cancel).image

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


def _configuration(plan, channels, pipeline_style):
    """Describe the processing stage and source-channel order for later reuse."""
    return {
        'version': 1, 'pipeline_style': pipeline_style,
        'channels': [int(channel) for channel in channels],
        'processing': plan.provenance() if plan else {'operation': 'none'},
        'stage': 'after illumination, before background/percentile normalization',
        'measurement_intensity_source': 'original persisted intensities',
    }


def validate_psf_resume(settings, root, channels, *, expected_fields):
    """Refuse V1 archive reuse after a PSF change or incomplete processing.

    :param settings: requested PSF settings; off also checks a prior PSF run.
    :param root: experiment root containing masks/ and psf/.
    :param channels: source intensity indices in archive order.
    :param expected_fields: exact field stems carried by the input archives.
    :returns: None for a compatible completed set or untouched legacy inputs.
    :raises ValueError: if archives cannot be proven to match this request.
    """
    plan = prepare_psf(settings)
    path = _record_path(root)
    if plan is None and not path.exists():
        return
    message = ('PSF preprocessing does not match the saved Mask inputs. '
               'Enable preprocessing to rebuild the complete archive set '
               'from stack/ before generating masks.')
    try:
        record = json.loads(path.read_text())
        if (record['configuration'] != _configuration(plan, channels, 'v1')
                or not record['complete']
                or set(record['fields']) != set(expected_fields)):
            raise ValueError(message)
        archives = sorted((Path(root) / 'masks').glob('*.npz'))
        if {p.name: _digest(p) for p in archives} != record['archives']:
            raise ValueError(message)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(message) from exc


class _SegmentationPSFSession:
    """Track a complete application, including switching PSF processing off."""

    def __init__(self, plan, root, channels, pipeline_style):
        """Start a captured application with an explicit incomplete record."""
        self.plan = plan
        self.root = Path(root)
        self.pipeline_style = pipeline_style
        self.configuration = _configuration(plan, channels, pipeline_style)
        self.fields = set()
        self._write(False)

    def correct(self, image):
        """Process one private field, or return it unchanged when switching off."""
        checkpoint()
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

    :param settings: PSF configuration accepted by :func:`prepare_psf`.
    :param root: experiment root; original image files remain untouched.
    :param channels: selected indices in persisted intensity-channel order.
    :param pipeline_style: V1 normalized archives or V2 combined output stacks.
    :returns: a new session, or None for an untracked run with PSF disabled.
    """
    plan = prepare_psf(settings)
    if plan is None and not _record_path(root).exists():
        return None
    return _SegmentationPSFSession(plan, root, channels, pipeline_style)
