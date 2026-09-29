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
    'image_qc_classifier': False,
    'image_qc_classifier_model': None,
    'image_qc_classifier_labels': None,
    'image_qc_classifier_threshold': 0.5,
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
    if not isinstance(policy['image_qc_classifier'], bool):
        raise ValueError('image_qc_classifier must be True or False')
    for key in ('image_qc_classifier_model', 'image_qc_classifier_labels'):
        value = policy[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            policy[key] = None
        elif isinstance(value, (str, os.PathLike)):
            policy[key] = os.fspath(value)
        else:
            raise ValueError(f'{key} must be a file path or blank')
    threshold = float(policy['image_qc_classifier_threshold'])
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('image_qc_classifier_threshold must be between 0 and 1')
    policy['image_qc_classifier_threshold'] = threshold
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
    classifier = None
    if policy['image_qc_mode'] != 'off':
        paths = sorted(root.joinpath('stack').glob('*.npy')) if paths is None else list(paths)
        if not paths:
            raise ValueError('Image quality needs the raw stack fields; restore stack/ or rerun preprocessing')
        if policy['image_qc_classifier']:
            classifier = _prepare_qc_classifier(root, policy, paths, channel_ids)
        for path in paths:
            checkpoint()
            path = Path(path)
            image = np.load(path, mmap_mode='r', allow_pickle=False)
            metrics = assess_image(image, policy, channel_ids)
            if classifier is not None:
                _classify_records(classifier, image, metrics, policy, channel_ids)
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
    columns = ['field', 'status', 'channel', 'focus_variance', 'saturation_level',
               'saturation_fraction', 'nonfinite_fraction', 'reasons']
    if classifier is not None:
        columns[-1:-1] = [f'p_{name}' for name in _QC_CLASSES]
    writer = csv.DictWriter(stream, fieldnames=columns)
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

    :param root: project folder containing measurements/measurements.db.
    :param rejected: rejected field filenames, also matched without extensions.
    :returns: None if no existing measurement row matches a rejected field.
    :raises ValueError: a table's file_name column contains a rejected identity.
    """
    from contextlib import closing
    from .database_concurrency import connect

    database = Path(root) / 'measurements' / 'measurements.db'
    if not rejected or not database.is_file():
        return
    identities = sorted({value for name in rejected for value in (name, Path(name).stem)})
    with closing(connect(database, readonly=True)) as connection:
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


_QC_CLASSES = ('out_of_focus', 'saturated', 'debris', 'bubble', 'empty')
_QC_TILE = 64
_QC_TILES = 16
_QC_BUILTIN_FIELDS = 2400
_QC_BUILTIN_EPOCHS = 12
_QC_FINE_TUNE_EPOCHS = 15
_QC_MODEL_VERSION = 1
_QC_LABEL_ALIASES = {'good': (), 'ok': (), 'pass': (), 'blur': ('out_of_focus',),
                     'blurry': ('out_of_focus',), 'defocus': ('out_of_focus',),
                     'saturation': ('saturated',), 'bubbles': ('bubble',),
                     'blank': ('empty',)}


def _builtin_qc_model_path():
    """Where the built-in classifier is cached after it is first trained."""
    return Path.home() / '.spacr' / 'models' / f'image_qc_classifier_v{_QC_MODEL_VERSION}.pt'


def _best_focus_plane(plane):
    """Return the 2-D plane of a volume with the largest Laplacian variance."""
    import numpy as np
    from scipy.ndimage import laplace

    plane = np.nan_to_num(np.asarray(plane, dtype=np.float32))
    if plane.ndim == 2:
        return plane
    planes = plane.reshape((-1, *plane.shape[-2:]))
    return planes[int(np.argmax([np.var(laplace(member)) for member in planes]))]


def _qc_inputs(plane, ceiling=None, tiles=_QC_TILES):
    """Turn one raw channel into the classifier's scale-free inputs.

    Three maps are derived from the raw plane: intensity above the median
    background divided by the brighter of the signal range and 20 noise
    standard deviations, so a field of noise alone stays near zero; the
    Laplacian magnitude relative to pixel noise, so blur shows as absent
    structure rather than a dim image; and the pixels at the saturation
    ceiling. ``tiles`` native-resolution 64-pixel tiles spread over the
    field show cell-scale detail and a 64 by 64 average of the whole field
    shows field-scale structure such as bubbles.

    :returns: ``(tiles, 3, 64, 64)`` and ``(3, 64, 64)`` float32 arrays.
    """
    import numpy as np
    import torch
    from scipy.ndimage import laplace

    plane = _best_focus_plane(plane)
    background = float(np.median(plane))
    step = np.diff(plane[:, ::2], axis=1).ravel()
    noise = 1.4826 * float(np.median(np.abs(step - np.median(step)))) / math.sqrt(2)
    scale = max(float(np.percentile(plane, 99.9)) - background, 20 * noise, 1e-6)
    maps = np.stack([np.clip((plane - background) / scale, -1, 3),
                     np.log1p(np.abs(laplace(plane)) / max(noise, 1e-3 * scale)),
                     (plane >= ceiling) if ceiling is not None else np.zeros_like(plane)]
                    ).astype(np.float32)
    size = _QC_TILE
    pad = [(0, 0)] + [(0, max(0, size - extent)) for extent in maps.shape[1:]]
    padded = np.pad(maps, pad, mode='reflect' if min(maps.shape[1:]) > 1 else 'edge')
    side = int(math.ceil(math.sqrt(tiles)))
    rows = np.linspace(0, padded.shape[1] - size, side).astype(int)
    columns = np.linspace(0, padded.shape[2] - size, side).astype(int)
    stack = np.stack([padded[:, y:y + size, x:x + size] for y in rows for x in columns][:tiles])
    tensor = torch.from_numpy(np.ascontiguousarray(padded))[None]
    thumbnail = torch.nn.functional.adaptive_avg_pool2d(tensor, (size, size))[0].numpy()
    return stack, thumbnail


def _qc_network():
    """Build the two-branch convolutional classifier with random weights.

    One branch reads each native-resolution tile and keeps the strongest
    response per tile, the other reads the whole-field average; tile
    responses are summarised by maximum and mean so a single bad region
    still counts. One logit per class in ``_QC_CLASSES``.
    """
    import torch
    from torch import nn

    def trunk():
        """Four 3x3 convolution stages shared in shape by both branches."""
        layers, width = [], 3
        for index, out in enumerate((16, 32, 48, 48)):
            layers += [nn.Conv2d(width, out, 3, padding=1), nn.ReLU()]
            if index < 3:
                layers.append(nn.MaxPool2d(2))
            width = out
        return nn.Sequential(*layers)

    class QCNet(nn.Module):
        """Tile and whole-field branches joined by one linear layer."""

        def __init__(self):
            """Build the two convolutional trunks and the shared head."""
            super().__init__()
            self.tile = trunk()
            self.field = trunk()
            self.head = nn.Linear(48 * 3, len(_QC_CLASSES))

        def forward(self, tiles, thumbnail):
            """Return class logits for a batch of fields."""
            batch, count = tiles.shape[:2]
            local = self.tile(tiles.flatten(0, 1)).amax((2, 3)).view(batch, count, -1)
            whole = self.field(thumbnail).amax((2, 3))
            return self.head(torch.cat([local.amax(1), local.mean(1), whole], 1))

    return QCNet()


def _train_qc_network(model, tiles, thumbnails, targets, epochs, seed=0, rate=2e-3, balance=10.0):
    """Fit ``model`` in place on the CPU with flips, rotations and tile dropout.

    At most eight CPU threads are used while fitting.

    :param tiles: ``(fields, tiles, 3, 64, 64)`` array.
    :param thumbnails: ``(fields, 3, 64, 64)`` array.
    :param targets: ``(fields, classes)`` array of 0/1 labels; a field with
        no defect has an all-zero row.
    :param balance: the most a rare class is weighted up relative to its
        absent cases; high when training from scratch, low when fine-tuning
        so a few labels do not inflate false flags.
    :returns: the model, left in evaluation mode.
    """
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(seed)
    tiles = torch.as_tensor(np.asarray(tiles, np.float32))
    thumbnails = torch.as_tensor(np.asarray(thumbnails, np.float32))
    targets = torch.as_tensor(np.asarray(targets, np.float32))
    positive = targets.mean(0).clamp(0.02, 0.98)
    loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=((1 - positive) / positive).clamp(max=float(balance)))
    optimiser = torch.optim.Adam(model.parameters(), lr=rate)
    model.train()
    keep = max(1, tiles.shape[1] // 2)
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 8))
    try:
        for _ in range(int(epochs)):
            order = torch.randperm(len(targets), generator=generator)
            for start in range(0, len(order), 32):
                index = order[start:start + 32]
                chosen = torch.randperm(tiles.shape[1], generator=generator)[:keep]
                local, whole = tiles[index][:, chosen], thumbnails[index]
                turns = int(torch.randint(4, (1,), generator=generator))
                local, whole = local.rot90(turns, (-2, -1)), whole.rot90(turns, (-2, -1))
                if torch.rand(1, generator=generator) < .5:
                    local, whole = local.flip(-1), whole.flip(-1)
                optimiser.zero_grad()
                loss(model(local, whole), targets[index]).backward()
                optimiser.step()
    finally:
        torch.set_num_threads(threads)
    return model.eval()


def _predict_qc(model, tiles, thumbnails):
    """Return class probabilities, one row per field."""
    import numpy as np
    import torch

    with torch.no_grad():
        outputs = [torch.sigmoid(model(torch.as_tensor(np.asarray(tiles[start:start + 64], np.float32)),
                                       torch.as_tensor(np.asarray(thumbnails[start:start + 64], np.float32))))
                   for start in range(0, len(tiles), 64)]
    return torch.cat(outputs).numpy() if outputs else np.zeros((0, len(_QC_CLASSES)), np.float32)


def _synthetic_qc_field(rng, defects, size=256, ceiling=65535):
    """Render one synthetic fluorescence field carrying the named defects.

    Cells are textured ellipses on an uneven background with shot and read
    noise. ``out_of_focus`` blurs the optics, ``saturated`` raises the gain
    until part of the field clips at ``ceiling``, ``debris`` adds bright
    fibres or aggregates, ``bubble`` adds a dimmed disc with a refractive
    rim and ``empty`` leaves background and noise only.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    y, x = np.mgrid[:size, :size].astype(np.float32)
    background = rng.uniform(80, 0.03 * ceiling)
    signal = np.zeros((size, size), np.float32)
    if 'empty' not in defects:
        radius = rng.uniform(3, 16)
        crowd = min(1200, int(rng.uniform(0, 1) * (size / radius) ** 2))
        region = np.ones((size, size), bool)
        if rng.uniform() < .4:
            smooth = gaussian_filter(rng.normal(size=(size, size)), size / rng.uniform(4, 10))
            region = smooth > np.percentile(smooth, rng.uniform(20, 70))
        spots = np.argwhere(region)
        for _ in range(int(rng.integers(2, 12)) + int(rng.integers(0, crowd + 1))):
            cy, cx = spots[int(rng.integers(len(spots)))] + rng.uniform(-.5, .5, 2)
            a, b = radius * rng.uniform(.6, 1.4, 2)
            reach = int(2 * max(a, b)) + 2
            top, left = max(0, int(cy) - reach), max(0, int(cx) - reach)
            bottom, right = min(size, int(cy) + reach), min(size, int(cx) + reach)
            if bottom <= top or right <= left:
                continue
            angle = rng.uniform(0, np.pi)
            dy, dx = y[top:bottom, left:right] - cy, x[top:bottom, left:right] - cx
            u = (dx * np.cos(angle) + dy * np.sin(angle)) / a
            v = (-dx * np.sin(angle) + dy * np.cos(angle)) / b
            signal[top:bottom, left:right] += rng.uniform(.3, 1) * np.clip(1.3 - u * u - v * v, 0, .3) / .3
        texture = 1 + rng.uniform(.3, 1.5) * gaussian_filter(rng.normal(size=signal.shape), rng.uniform(.8, 2))
        signal = gaussian_filter(signal * np.clip(texture, .3, None), rng.uniform(.5, 1.2))
    else:
        for _ in range(int(rng.integers(0, 3))):
            cy, cx = rng.uniform(0, size, 2)
            signal += .3 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 2)
    amplitude = rng.uniform(15, 60) * math.sqrt(background) + rng.uniform(0, .5) * (.6 * ceiling - background)
    field = signal * amplitude
    haze = gaussian_filter(rng.normal(size=(size, size)), size / rng.uniform(2, 8))
    haze = haze / max(float(np.abs(haze).max()), 1e-6)
    field = field + haze * (rng.uniform(0, 8) * math.sqrt(background) if 'empty' in defects
                            else rng.uniform(0, .4) * amplitude)
    if 'debris' in defects:
        for _ in range(int(rng.integers(1, 4))):
            debris = np.zeros_like(signal)
            if rng.uniform() < .5:
                cy, cx = rng.uniform(0, size, 2)
                heading = rng.uniform(0, 2 * np.pi)
                for _ in range(int(rng.integers(60, 200))):
                    heading += rng.normal(0, .15)
                    cy, cx = cy + np.sin(heading), cx + np.cos(heading)
                    if 0 <= cy < size and 0 <= cx < size:
                        debris[int(cy), int(cx)] = 1
                debris = gaussian_filter(debris, rng.uniform(1, 2.5))
                debris /= max(debris.max(), 1e-6)
            elif rng.uniform() < .5:
                cy, cx = rng.uniform(0, size, 2)
                corners = rng.integers(3, 8)
                angles = np.sort(rng.uniform(0, 2 * np.pi, corners))
                reach = rng.uniform(6, 30) * rng.uniform(.5, 1, corners)
                from matplotlib.path import Path as Outline
                outline = Outline(np.c_[cx + reach * np.cos(angles), cy + reach * np.sin(angles)])
                debris = outline.contains_points(np.c_[x.ravel(), y.ravel()]).reshape(x.shape).astype(np.float32)
            else:
                cy, cx = rng.uniform(0, size, 2)
                for _ in range(int(rng.integers(3, 9))):
                    oy, ox = rng.normal(0, rng.uniform(4, 12), 2)
                    spread = rng.uniform(3, 9)
                    debris += np.exp(-((y - cy - oy) ** 2 + (x - cx - ox) ** 2) / (2 * spread ** 2))
                debris = np.clip(debris, 0, 1)
            field += debris * max(amplitude, 30 * math.sqrt(background)) * rng.uniform(1.5, 6)
    optics = rng.uniform(2, 8) if 'out_of_focus' in defects else rng.uniform(0, 1)
    if optics:
        field = gaussian_filter(field, optics)
    field = field + background * (1 + rng.uniform(-.3, .3) * (x / size) + rng.uniform(-.3, .3) * (y / size))
    if 'bubble' in defects:
        cy, cx = rng.uniform(-.1, 1.1, 2) * size
        radius = rng.uniform(.18, .45) * size
        distance = np.hypot(y - cy, x - cx)
        inside = gaussian_filter((distance < radius).astype(np.float32), 2)
        rim = np.exp(-((distance - radius) ** 2) / (2 * rng.uniform(1, 3) ** 2))
        field = field * (1 - inside * rng.uniform(.4, .8)) + rim * rng.choice([-.5, 1.5]) * float(np.median(field))
    if 'saturated' in defects:
        fraction = rng.uniform(.01, .12)
        level = np.percentile(field, 100 * (1 - fraction))
        field = background + (field - background) * (ceiling * 1.05 - background) / max(level - background, 1e-6)
    else:
        peak = float(field.max())
        if peak > .85 * ceiling:
            field = background + (field - background) * (.85 * ceiling - background) / (peak - background)
    field = np.clip(field, 0, None)
    gain = rng.uniform(1, 4)
    field = rng.poisson(field / gain) * gain + rng.normal(0, rng.uniform(2, 10), field.shape)
    return np.clip(field, 0, ceiling).astype(np.uint16 if ceiling <= 65535 else np.float32)


def _synthetic_qc_labels(rng):
    """Draw the defects for one training field: most single, some paired."""
    draw = rng.uniform()
    if draw < .35:
        return ()
    first = _QC_CLASSES[int(rng.integers(len(_QC_CLASSES)))]
    if first != 'empty' and rng.uniform() < .2:
        second = _QC_CLASSES[int(rng.integers(len(_QC_CLASSES) - 1))]
        return tuple(sorted({first, second}))
    return (first,)


def _train_builtin_qc_model(fields=None, epochs=None, seed=0):
    """Train the built-in classifier on synthetic fields with planted defects.

    :returns: a model in evaluation mode.
    """
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tiles, thumbnails, targets = [], [], []
    for _ in range(int(fields or _QC_BUILTIN_FIELDS)):
        labels = _synthetic_qc_labels(rng)
        ceiling = int(rng.choice([4095, 65535]))
        image = _synthetic_qc_field(rng, labels, size=int(rng.choice([192, 256, 320])), ceiling=ceiling)
        local, whole = _qc_inputs(image, ceiling)
        tiles.append(local)
        thumbnails.append(whole)
        targets.append([name in labels for name in _QC_CLASSES])
    return _train_qc_network(_qc_network(), np.stack(tiles), np.stack(thumbnails),
                             np.asarray(targets, np.float32), epochs or _QC_BUILTIN_EPOCHS, seed)


def _save_qc_model(model, path, source):
    """Save weights with their class order so a later run reads tensors only."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.part')
    torch.save(dict(version=_QC_MODEL_VERSION, classes=list(_QC_CLASSES), source=str(source),
                    state=model.state_dict()), temporary)
    os.replace(temporary, path)
    return path


def _load_qc_model(path):
    """Load a saved classifier as tensors only, refusing another class order."""
    import torch

    saved = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or saved.get('version') != _QC_MODEL_VERSION or \
            tuple(saved.get('classes', ())) != _QC_CLASSES:
        raise ValueError(f'{path} is not an image-quality classifier saved by this version of spaCR')
    model = _qc_network()
    model.load_state_dict(saved['state'])
    return model.eval()


def _base_qc_model(policy):
    """The saved model named by the policy, or the cached built-in one."""
    if policy['image_qc_classifier_model']:
        return _load_qc_model(policy['image_qc_classifier_model'])
    path = _builtin_qc_model_path()
    if path.is_file():
        return _load_qc_model(path)
    print('Image quality: training the built-in classifier on synthetic fields '
          '(once, on the CPU; about a few minutes)...')
    model = _train_builtin_qc_model()
    _save_qc_model(model, path, 'built-in synthetic planted defects')
    return model


def _field_ceiling(image, policy, channel):
    """The saturation level assess_image would use for ``channel``."""
    import numpy as np

    level = policy['image_qc_saturation_level'].get(str(channel))
    if level is None and np.issubdtype(np.asarray(image).dtype, np.integer):
        level = float(np.iinfo(np.asarray(image).dtype).max)
    return level


def _classify_records(model, image, records, policy, channel_ids=None):
    """Add class probabilities and classifier reasons to assess_image records.

    Each screened channel is classified on its own; a class at or above
    ``image_qc_classifier_threshold`` adds ``classifier_<class>`` to that
    channel's reasons, which exclude mode then treats like any other flag.
    """
    import numpy as np

    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    channel_ids = list(range(image.shape[-1])) if channel_ids is None else list(channel_ids)
    for record in records:
        channel = record['channel']
        local, whole = _qc_inputs(image[..., channel_ids.index(channel)],
                                  _field_ceiling(image, policy, channel))
        probabilities = _predict_qc(model, local[None], whole[None])[0]
        for name, value in zip(_QC_CLASSES, probabilities):
            record[f'p_{name}'] = round(float(value), 4)
            if value >= policy['image_qc_classifier_threshold']:
                record['reasons'].append(f'classifier_{name}')
    return records


def _parse_qc_labels(value):
    """Split one annotation cell into classifier classes; good means none."""
    names = set()
    for token in str(value).replace(',', ';').split(';'):
        token = token.strip().lower().replace(' ', '_').replace('-', '_')
        if not token:
            continue
        if token in _QC_CLASSES:
            names.add(token)
        elif token in _QC_LABEL_ALIASES:
            names.update(_QC_LABEL_ALIASES[token])
        else:
            raise ValueError(f'Unknown image-quality label {value!r}; use good or '
                             + ', '.join(_QC_CLASSES))
    return names


def _labelled_qc_fields(policy, paths, channel_ids=None):
    """Read the annotation table and pair each labelled field with its inputs.

    The table needs a field column (the raw file name, with or without .npy)
    and a label column: good, or one or more of the classes separated by
    semicolons. An optional channel column picks the channel; otherwise the
    first screened channel is used.

    :returns: list of dicts with field, channel, labels, tiles, thumbnail and
        the rule-based focus and saturation metrics of that channel.
    """
    import numpy as np
    from .tabular import read_table

    table = read_table(policy['image_qc_classifier_labels'], report=None)
    table.columns = [str(column).strip().lower() for column in table.columns]
    table = table.rename(columns={'fieldid': 'field', 'chanid': 'channel'})
    if not {'field', 'label'} <= set(table.columns):
        raise ValueError('image_qc_classifier_labels needs field and label columns')
    by_name = {}
    for path in paths:
        by_name[Path(path).name] = Path(path)
        by_name[Path(path).stem] = Path(path)
    samples = []
    for row in table.to_dict('records'):
        name = str(row['field']).strip()
        path = by_name.get(name) or by_name.get(Path(name).name)
        if path is None:
            continue
        image = np.load(path, mmap_mode='r', allow_pickle=False)
        records = assess_image(image, policy, channel_ids)
        channel = row.get('channel')
        if channel is None or (isinstance(channel, float) and math.isnan(channel)):
            record = records[0]
        else:
            record = next((item for item in records if str(item['channel']) == str(int(channel))), None)
            if record is None:
                raise ValueError(f'Labelled channel {channel} of {name} is not being screened')
        stored = image[..., None] if image.ndim == 2 else image
        ids = list(range(stored.shape[-1])) if channel_ids is None else list(channel_ids)
        local, whole = _qc_inputs(stored[..., ids.index(record['channel'])],
                                  _field_ceiling(stored, policy, record['channel']))
        samples.append(dict(field=path.name, channel=record['channel'], labels=_parse_qc_labels(row['label']),
                            tiles=local, thumbnail=whole, focus=record['focus_variance'],
                            saturation=record['saturation_fraction'] or 0.0,
                            rule_flag=bool(record['reasons'])))
    return samples


def _precision_recall(truth, flagged):
    """Precision and recall of boolean calls; undefined values are NaN."""
    import numpy as np

    truth, flagged = np.asarray(truth, bool), np.asarray(flagged, bool)
    hits = float(np.sum(truth & flagged))
    precision = hits / flagged.sum() if flagged.sum() else float('nan')
    recall = hits / truth.sum() if truth.sum() else float('nan')
    return precision, recall


def _tuned_rule(focus, saturation, truth):
    """Best focus and saturation cut-offs for any-defect F1 on training fields.

    This is the most the current rule-based metrics can do when their
    thresholds are chosen from labelled fields: a field is flagged when its
    focus variance is below the focus cut-off or its saturated fraction is
    above the saturation cut-off.
    """
    import numpy as np

    focus, saturation, truth = map(np.asarray, (focus, saturation, truth))
    focus_cuts = np.concatenate([[-np.inf], np.unique(focus)])
    saturation_cuts = np.concatenate([[np.inf], np.unique(saturation)])
    best, choice = -1.0, (-np.inf, np.inf)
    for low in focus_cuts:
        for high in saturation_cuts:
            flagged = (focus < low) | (saturation > high)
            hits = np.sum(flagged & truth)
            score = 2 * hits / max(flagged.sum() + truth.sum(), 1)
            if score > best:
                best, choice = score, (low, high)
    return choice


def _benchmark_qc(samples, base, policy, folds=5, epochs=None, seed=0):
    """Cross-validated precision and recall: classifier against rule metrics.

    Each fold fine-tunes a copy of ``base`` on the other folds and scores the
    held-out fields; the rule baseline is scored twice, with the thresholds
    of the saved policy and with cut-offs tuned on the same training folds.

    :returns: rows with method, defect, precision, recall, positives, fields.
    """
    import copy
    import numpy as np

    truth = np.array([[name in sample['labels'] for name in _QC_CLASSES] for sample in samples])
    folds = max(2, min(int(folds), len(samples)))
    order = np.random.default_rng(seed).permutation(len(samples))
    probabilities = np.zeros(truth.shape, np.float32)
    tuned = np.zeros((len(samples), 2), bool)
    focus = np.array([sample['focus'] for sample in samples])
    saturation = np.array([sample['saturation'] for sample in samples])
    tiles = np.stack([sample['tiles'] for sample in samples])
    thumbnails = np.stack([sample['thumbnail'] for sample in samples])
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test)
        model = _train_qc_network(copy.deepcopy(base), tiles[train], thumbnails[train],
                                  truth[train], epochs or _QC_FINE_TUNE_EPOCHS, seed + fold,
                                  rate=5e-4, balance=3.0)
        probabilities[test] = _predict_qc(model, tiles[test], thumbnails[test])
        low, high = _tuned_rule(focus[train], saturation[train], truth[train].any(1))
        tuned[test] = np.c_[focus[test] < low, saturation[test] > high]
    called = probabilities >= policy['image_qc_classifier_threshold']
    configured = np.array([sample['rule_flag'] for sample in samples])
    rule = np.zeros(truth.shape, bool)
    rule[:, _QC_CLASSES.index('out_of_focus')] = tuned[:, 0]
    rule[:, _QC_CLASSES.index('saturated')] = tuned[:, 1]
    rows = []
    for method, calls in (('classifier', called), ('rule_saved_policy', None), ('rule_tuned', rule)):
        for index, name in enumerate(('any_defect',) + _QC_CLASSES):
            actual = truth.any(1) if index == 0 else truth[:, index - 1]
            if calls is None:
                flagged = configured if index == 0 else np.zeros(len(samples), bool)
            else:
                flagged = calls.any(1) if index == 0 else calls[:, index - 1]
            precision, recall = _precision_recall(actual, flagged)
            rows.append(dict(method=method, defect=name, precision=precision, recall=recall,
                             positives=int(actual.sum()), fields=len(samples)))
    return rows


def _prepare_qc_classifier(root, policy, paths, channel_ids=None):
    """Load or fine-tune the classifier this screening run will apply.

    With ``image_qc_classifier_labels`` the labelled fields are first used to
    score the classifier against the rule metrics by cross-validation
    (qc/image_qc_benchmark.csv), then to fine-tune on all of them; the
    result is saved as qc/image_qc_model.pt and applied to every field.
    """
    import numpy as np
    import pandas as pd
    from .tabular import write_table

    model = _base_qc_model(policy)
    if not policy['image_qc_classifier_labels']:
        return model
    samples = _labelled_qc_fields(policy, paths, channel_ids)
    if len(samples) < 4:
        raise ValueError('image_qc_classifier_labels matched fewer than 4 raw fields; '
                         'label at least 4 fields by their file names')
    qc = Path(root) / 'qc'
    if len(samples) >= 10:
        rows = _benchmark_qc(samples, model, policy)
        write_table(pd.DataFrame(rows), qc / 'image_qc_benchmark.csv')
        for row in rows:
            if row['defect'] == 'any_defect':
                print(f"Image quality {row['method']}: precision {row['precision']:.3f}, "
                      f"recall {row['recall']:.3f} over {row['fields']} labelled fields")
    else:
        print('Image quality: fewer than 10 labelled fields, so no held-out benchmark was run.')
    truth = np.array([[name in sample['labels'] for name in _QC_CLASSES] for sample in samples], np.float32)
    model = _train_qc_network(model, np.stack([sample['tiles'] for sample in samples]),
                              np.stack([sample['thumbnail'] for sample in samples]), truth, _QC_FINE_TUNE_EPOCHS,
                              rate=5e-4, balance=3.0)
    _save_qc_model(model, qc / 'image_qc_model.pt', policy['image_qc_classifier_labels'])
    return model
