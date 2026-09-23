"""Auditable image-quality screening before segmentation, using raw intensities."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULTS = {
    'image_qc_mode': 'off',
    'image_qc_channels': [],
    'image_qc_min_focus': {},
    'image_qc_max_saturation': {},
    'image_qc_saturation_level': {},
    'image_qc_max_nonfinite': 0.0,
}
REPORT = 'qc/image_quality.json'


def quality_policy(settings):
    """Validate a saved per-channel policy without inspecting any image.

    :param settings: Mask settings containing image_qc_* options.
    :returns: JSON-serializable policy; empty channel list means every channel.
    :raises ValueError: unsupported mode, invalid channel or invalid threshold.
    """
    policy = {key: settings.get(key, value) for key, value in DEFAULTS.items()}
    if policy['image_qc_mode'] not in ('off', 'report', 'exclude'):
        raise ValueError('image_qc_mode must be off, report or exclude')
    channels = policy['image_qc_channels']
    if not isinstance(channels, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in channels):
        raise ValueError('image_qc_channels must contain nonnegative channel indices')
    policy['image_qc_channels'] = list(dict.fromkeys(channels))
    for key in ('image_qc_min_focus', 'image_qc_max_saturation', 'image_qc_saturation_level'):
        values = policy[key]
        if not isinstance(values, dict):
            raise ValueError(f'{key} must map channel indices to thresholds')
        converted = {}
        for channel, value in values.items():
            if isinstance(channel, bool) or not str(channel).isdigit():
                raise ValueError(f'{key}: channel indices must be nonnegative integers')
            value = float(value)
            if not math.isfinite(value) or value < 0 or (
                    key == 'image_qc_max_saturation' and value > 1) or (
                    key == 'image_qc_saturation_level' and value == 0):
                raise ValueError(f'{key}: invalid threshold for channel {channel}')
            converted[str(int(channel))] = value
        policy[key] = converted
    limit = float(policy['image_qc_max_nonfinite'])
    if not math.isfinite(limit) or not 0 <= limit <= 1:
        raise ValueError('image_qc_max_nonfinite must be between 0 and 1')
    policy['image_qc_max_nonfinite'] = limit
    return policy


def assess_image(image, settings, channel_ids=None):
    """Measure focus, saturation and nonfinite pixels on unnormalized channels.

    :param image: YX, YXC or leading-dimensions plus YXC array.
    :param settings: image_qc_* policy, validated before use.
    :param channel_ids: optional acquisition-channel labels for stored C planes.
    :returns: one metric/reason record per selected channel. Focus is the best
        plane's Laplacian variance in raw intensity units squared, avoiding
        rejection solely because a z stack includes out-of-focus planes.
        Saturation uses an explicit acquisition level or integer dtype ceiling;
        it never uses the brightest observed pixel. Object counts are not read.
    :raises ValueError: channels, shape or saturation calibration are unavailable.
    """
    import numpy as np
    from scipy.ndimage import laplace

    policy = quality_policy(settings)
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim < 3 or not image.size:
        raise ValueError('Image quality requires a nonempty YX or (..., Y, X, C) image')
    channel_ids = list(range(image.shape[-1])) if channel_ids is None else list(channel_ids)
    if len(channel_ids) != image.shape[-1]:
        raise ValueError('Image quality channel mapping must match the raw channel planes')
    requested = policy['image_qc_channels'] or list(dict.fromkeys(channel_ids))
    if not set(requested) <= set(channel_ids):
        raise ValueError('Selected image-quality channels are absent from this field')
    for key in ('image_qc_min_focus', 'image_qc_max_saturation', 'image_qc_saturation_level'):
        if not set(map(int, policy[key])) <= set(requested):
            raise ValueError(f'{key} contains a channel that is not being screened')
    records = []
    for channel in requested:
        key = str(channel)
        plane = image[..., channel_ids.index(channel)]
        finite = np.isfinite(plane)
        invalid = float(1 - finite.mean())
        numeric = np.where(finite, plane, 0).astype(np.float64)
        planes = numeric.reshape((-1, *numeric.shape[-2:]))
        focus = max(float(np.var(laplace(member))) for member in planes)
        ceiling = policy['image_qc_saturation_level'].get(key)
        if ceiling is None and np.issubdtype(image.dtype, np.integer):
            ceiling = float(np.iinfo(image.dtype).max)
        fraction = float(np.mean(finite & (plane >= ceiling))) if ceiling is not None else None
        if key in policy['image_qc_max_saturation'] and ceiling is None:
            raise ValueError(f'Channel {channel}: floating images need image_qc_saturation_level')
        reasons = []
        if key in policy['image_qc_min_focus'] and focus < policy['image_qc_min_focus'][key]:
            reasons.append('focus_below_threshold')
        if key in policy['image_qc_max_saturation'] and fraction > policy['image_qc_max_saturation'][key]:
            reasons.append('saturation_above_threshold')
        if invalid > policy['image_qc_max_nonfinite']:
            reasons.append('nonfinite_pixels')
        records.append(dict(channel=channel, focus_variance=focus,
                            saturation_level=ceiling, saturation_fraction=fraction,
                            nonfinite_fraction=invalid, reasons=reasons))
    return records


def _atomic_text(path, text):
    """Publish one complete UTF-8 report without exposing partial writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.image_quality_')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_gallery(destination, fields, paths, channel_ids):
    """Write a read-only review gallery with at most 64 field thumbnails."""
    import base64
    from html import escape
    import numpy as np
    from PIL import Image

    by_name = {Path(path).name: Path(path) for path in paths}
    cards = []
    ordered = sorted(fields, key=lambda field: (not bool(field['reasons']), field['field']))
    for field in ordered[:64]:
        path = by_name[field['field']]
        image = np.load(path, mmap_mode='r', allow_pickle=False)
        metric = next((record for record in field['channels'] if record['reasons']), field['channels'][0])
        channel = metric['channel']
        if image.ndim > 2:
            position = list(channel_ids).index(channel) if channel_ids is not None else channel
            image = image[..., position]
        if image.ndim > 2:
            image = np.max(image.reshape((-1, *image.shape[-2:])), axis=0)
        step = max(1, int(math.ceil(max(image.shape) / 512)))
        image = np.asarray(image[::step, ::step], dtype=np.float32)
        values = image[np.isfinite(image)]
        low, high = np.percentile(values, (1, 99)) if values.size else (0., 1.)
        scaled = np.nan_to_num((image - low) / max(float(high - low), 1e-12), nan=0., posinf=1., neginf=0.)
        thumbnail = Image.fromarray((scaled.clip(0, 1) * 255).astype(np.uint8))
        thumbnail.thumbnail((256, 256))
        buffer = io.BytesIO()
        thumbnail.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        details = '; '.join(field['reasons']) or 'No configured criterion failed.'
        cards.append('<article><img src="data:image/png;base64,' + encoded + '"><div><b>' +
                     escape(field['field']) + '</b> — ' + escape(field['status']) + '. ' +
                     escape(details) + f' Preview channel {channel}. Focus variance {metric["focus_variance"]:.5g}; ' +
                     'saturated fraction ' + escape(str(metric['saturation_fraction'])) + '.</div></article>')
    html = ('<!doctype html><html><meta charset="utf-8"><title>Image Quality</title><style>'
            'body{background:#14171b;color:#edf0f4;font:16px system-ui;margin:24px}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}'
            'article{border:1px solid #43505e;border-radius:12px;padding:12px;background:#20252bcc}'
            'img{max-width:100%;display:block;margin:0 auto 8px;border-radius:8px}'
            'header{margin-bottom:18px;line-height:1.5}</style><header><b>Image Quality.</b> '
            'Flagged fields appear first. Up to 64 previews are shown; the CSV contains every field and channel. '
            'Preview contrast is stretched for viewing only; metrics use raw intensities. '
            'Volumes use a maximum projection for this preview. Excluded fields are not zero-object detections.'
            '</header><main>' + ''.join(cards) + '</main></html>')
    _atomic_text(destination.with_suffix('.html'), html)


def screen_fields(root, settings, paths=None, channel_ids=None):
    """Save a field/channel report and return fields explicitly excluded by policy.

    :param root: project folder; reports go to qc/image_quality.json and .csv.
    :param settings: Mask settings; mode off clears a previously active policy.
    :param paths: optional iterable of raw NPY paths; defaults to root/stack.
    :param channel_ids: optional acquisition-channel labels for supplied arrays.
    :returns: excluded basenames, empty in report-only or off mode. Inputs are
        never deleted or rewritten. Report and exact policy precede exclusion.
    :raises ValueError: enabled screening has no raw fields or invalid calibration.
    """
    import numpy as np
    from .cancellation import checkpoint

    root = Path(root)
    policy = quality_policy(settings)
    destination = root / REPORT
    if policy['image_qc_mode'] == 'off' and not destination.exists():
        return []
    fields, excluded, rows = [], [], []
    paths = [] if policy['image_qc_mode'] == 'off' else paths
    if policy['image_qc_mode'] != 'off':
        paths = sorted(root.joinpath('stack').glob('*.npy')) if paths is None else list(paths)
        if not paths:
            raise ValueError('Image quality needs the raw stack fields; restore stack/ or rerun preprocessing')
        for path in paths:
            checkpoint()
            path = Path(path)
            metrics = assess_image(np.load(path, mmap_mode='r', allow_pickle=False), policy, channel_ids)
            reasons = [f"channel {record['channel']}: {reason}"
                       for record in metrics for reason in record['reasons']]
            status = ('excluded_image_quality' if policy['image_qc_mode'] == 'exclude' else 'flagged') if reasons else 'accepted'
            fields.append(dict(field=path.name, status=status, reasons=reasons, channels=metrics))
            if status == 'excluded_image_quality':
                excluded.append(path.name)
            rows.extend(dict(field=path.name, status=status, **dict(record, reasons='; '.join(record['reasons'])))
                        for record in metrics)
    ensure_no_retained_measurements(root, excluded)
    report = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  policy=policy, excluded_fields=excluded, fields=fields)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=['field', 'status', 'channel', 'focus_variance',
                                              'saturation_level', 'saturation_fraction',
                                              'nonfinite_fraction', 'reasons'])
    writer.writeheader()
    writer.writerows(rows)
    _write_gallery(destination, fields, paths, channel_ids)
    _atomic_text(destination.with_suffix('.csv'), stream.getvalue())
    _atomic_text(destination, json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Image quality: {len(fields)} fields screened; {len(excluded)} excluded. Report: {destination}')
    return excluded


def ensure_no_retained_measurements(root, rejected):
    """Refuse exclusions that would leave previously measured fields in reports.

    Existing results are never deleted. Re-screening an analyzed project needs
    a fresh project if excluded fields already have measurement rows.
    """
    import sqlite3

    database = Path(root) / 'measurements' / 'measurements.db'
    if not rejected or not database.is_file():
        return
    identities = sorted({value for name in rejected for value in (name, Path(name).stem)})
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info({quoted})')}
            if 'file_name' not in columns:
                continue
            for offset in range(0, len(identities), 400):
                selected = identities[offset:offset + 400]
                placeholders = ','.join('?' for _ in selected)
                if connection.execute(f'SELECT 1 FROM {quoted} WHERE file_name IN ({placeholders}) LIMIT 1', selected).fetchone():
                    raise ValueError(
                        'Image-quality exclusions overlap existing measurements. Use a fresh project '
                        'for this exclusion policy, or select report mode to review without exclusion. '
                        'Existing images, measurements and the previous quality policy were retained.')


def excluded_fields(root):
    """Read the active saved exclusion policy for downstream field consumers.

    :param root: project folder whose qc/image_quality.json is authoritative.
    :returns: excluded NPY basenames, or an empty set when no policy is active.
    :raises ValueError: a saved report has an unsupported or invalid schema.
    """
    path = Path(root) / REPORT
    if not path.exists():
        return set()
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('version') != 1:
        raise ValueError(f'Unsupported image-quality report: {path}')
    if report.get('policy', {}).get('image_qc_mode') != 'exclude':
        return set()
    names = report.get('excluded_fields', [])
    if not isinstance(names, list) or any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise ValueError(f'Invalid image-quality field identities: {path}')
    return set(names)


def filter_batch(batch, filenames, settings):
    """Remove policy-excluded fields without substituting empty label images.

    :param batch: normalized image batch with its original field axis.
    :param filenames: matching basenames in batch order.
    :param settings: current run settings carrying image_qc_excluded_fields.
    :returns: filtered batch and matching filenames in their original order.
    """
    excluded = set(settings.get('image_qc_excluded_fields', ()))
    if not excluded:
        return batch, filenames
    keep = [index for index, name in enumerate(filenames) if str(name) not in excluded]
    return batch[keep], [filenames[index] for index in keep]
