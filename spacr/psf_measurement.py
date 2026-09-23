"""Explicit PSF intensity selection and compatibility for quantitative Measure.

PSFs operate after Measure's normal rescaling and preprocessing hooks. Crops
retain their existing source intensities; quantitative PSF output stays float.
The captured plan, rather than a file path reopened per field, reaches workers.
"""
from __future__ import annotations

import json
from pathlib import Path

from .psf_pipeline import prepare_psf

SIGNATURE_KEY = '_psf_measurement_signature'


def prepare_measurement_psf(settings):
    """Capture a calibrated kernel only for explicitly processed measurements.

    :param settings: PSF settings plus ``psf_measurement_source``: ``original``
        (default; standard Measure intensities without PSF) or ``processed``.
        Processed requires convolution/deconvolution and explicit YX or ZYX
        sampling. Original retains stored PSF parameters without applying them.
    :returns: immutable PSFPlan or None. The caller's settings are unchanged.
    :raises ValueError: for an invalid source, disabled processing when requested,
        or invalid calibrated kernel settings.
    """
    source = settings.get('psf_measurement_source', 'original')
    if source == 'original':
        return None
    if source != 'processed':
        raise ValueError('psf_measurement_source must be original or processed')
    sampling = settings.get('psf_image_sampling_um')
    ndim = len(sampling) if isinstance(sampling, (tuple, list)) else 2
    plan = prepare_psf(settings, ndim=ndim)
    if plan is None:
        raise ValueError('Processed measurements require psf_operation '
                         'convolve or deconvolve')
    return plan


def measurement_psf_signature(plan):
    """Return stable JSON configuration, or None for standard intensities."""
    return (json.dumps(plan.provenance(), sort_keys=True, separators=(',', ':'))
            if plan is not None else None)


def measurement_psf_record(plan, image, *, hooks=(), channels=()):
    """Describe the exact intensity stream given to quantitative features.

    :param plan: captured processing plan or None for original choice.
    :param image: selected intensities after standard preprocessing, before PSF.
    :param hooks: names of registered standard preprocessing hooks, in order.
    :param channels: original intensity-channel indices in measurement order.
    :returns: JSON-safe field provenance, explicitly distinguishing crop pixels.
    """
    return {
        'source': 'processed' if plan is not None else 'original',
        'processing': plan.provenance() if plan is not None else None,
        'input_shape': list(image.shape), 'input_dtype': str(image.dtype),
        'output_dtype': 'float32' if plan is not None else str(image.dtype),
        'stage': 'after standard intensity rescaling and preprocessing hooks',
        'preprocessing_hooks': list(hooks),
        'channels': [int(channel) for channel in channels],
        'input_modified': False,
        'crop_intensity_source': 'source image after standard rescaling; '
                                 'PSF and preprocessing hooks not applied',
    }


def measurement_resume_settings(settings, *, recorded=False):
    """Compare active PSF behavior rather than dormant form parameters.

    :param settings: recorded or current Measure settings.
    :param recorded: keep a recorded kernel identity without reopening its file.
    :returns: fresh settings with inactive PSF knobs removed and one captured
        material signature. Legacy measurements have the original signature None.
    """
    result = {k: v for k, v in settings.items() if not k.startswith('psf_')}
    if SIGNATURE_KEY not in result:
        result[SIGNATURE_KEY] = (None if recorded else
                                measurement_psf_signature(prepare_measurement_psf(settings)))
    return result


def validate_measurement_psf_history(settings, db_path, plan):
    """Refuse to mix incompatible intensity sources even when resume is off.

    :param settings: current run settings, before its database snapshot is saved.
    :param db_path: measurement database; absence means a new project.
    :param plan: captured plan for this run.
    :raises ValueError: when existing measurement rows use another PSF identity
        or cannot be proved to use the requested processed intensities.
    """
    if not Path(db_path).is_file():
        return
    from .database_concurrency import connect
    from .resume import MEASURE_OWNED_TABLES, read_recorded_settings
    signature = measurement_psf_signature(plan)
    message = ('Existing measurements use a different PSF intensity source '
               'or kernel. Use a separate project/output database, or restore '
               'the recorded PSF configuration before measuring this project.')
    connection = connect(db_path, readonly=True)
    try:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        quantitative_tables = MEASURE_OWNED_TABLES - {'png_list', 'intensity_rescale'}
        occupied = any(connection.execute(
            f'SELECT 1 FROM "{table}" LIMIT 1').fetchone()
            for table in tables & quantitative_tables)
        if not occupied:
            return
        recorded = read_recorded_settings(str(db_path))
        previous = recorded.get(SIGNATURE_KEY)
        if previous in ('None', '', 'null'):
            previous = None
        if SIGNATURE_KEY in recorded and previous != signature:
            raise ValueError(message)
        found = False
        if 'intensity_rescale' in tables:
            columns = {row[1] for row in connection.execute(
                'PRAGMA table_info(intensity_rescale)')}
            if 'psf_signature' in columns:
                for (previous,) in connection.execute(
                        'SELECT DISTINCT psf_signature FROM intensity_rescale'):
                    found = True
                    if previous != signature:
                        raise ValueError(message)
        if signature is not None and not found and previous != signature:
            raise ValueError(message)
    finally:
        connection.close()
