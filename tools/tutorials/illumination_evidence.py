"""Independent input-preservation and pixel checks for the Illumination lesson.

This validates a bounded demonstration, not the biological suitability of a
flat-field estimate. No application calculation is imported here.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(stage):
    source = Path(stage) / 'example_data/plate1/merged'
    files = sorted(source.glob('*.npy'))
    if len(files) != 16:
        raise ValueError('Expected the preserved sixteen downloaded fields')
    parent = Path(stage) / 'illumination_runs'
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='real-downloaded-fields-', dir=parent))
    target = root / 'plate1/merged'
    target.mkdir(parents=True)
    records = []
    for path in files:
        image = np.load(path, mmap_mode='r', allow_pickle=False)
        if image.shape != (1994, 1994, 7) or image.dtype != np.uint16:
            raise ValueError('Downloaded field geometry or dtype changed')
        destination = target / path.name
        before = digest(path)
        shutil.copy2(path, destination)
        if digest(destination) != before or digest(path) != before:
            raise ValueError('Downloaded field changed while copying')
        records.append(dict(source=str(path), copy=str(destination), sha256=before))
    return dict(root=str(root), merged=str(target), files=records,
                synthetic_images=False, channel_indices=[0], shape=[1994, 1994, 7])


def require_preserved(records):
    for row in records:
        if digest(row['source']) != row['sha256'] or digest(row['copy']) != row['sha256']:
            raise ValueError('Original or private input field changed')
    return len(records) * 2


def verify_plane(plane, *, channels, shape):
    plane = np.asarray(plane)
    if plane.shape != (len(channels), *shape) or plane.dtype != np.float32:
        raise ValueError('Saved field shape or dtype differs')
    if not np.isfinite(plane).all() or np.any(plane <= 0):
        raise ValueError('Saved field must be finite and positive')
    means = plane.mean(axis=(1, 2), dtype=np.float64)
    if not np.allclose(means, 1, rtol=0, atol=1e-6):
        raise ValueError('Saved field is not mean-one normalised')
    return dict(pixels_checked=int(plane.size), mean=means.tolist(),
                minimum=float(plane.min()), maximum=float(plane.max()))


def inspect_model(path, merged, *, channels=(0,), shape=(1994, 1994), count=16):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['manifest']))
        if manifest.get('format') != 1 or set(manifest.get('index', {})) != {'field0'}:
            raise ValueError('Saved model identity differs')
        record = manifest['index']['field0']
        wanted = dict(plate='plate1', key='plate1', channels=list(channels),
                      dark=[0.0], n_fields=count, estimator='polynomial', degree=4)
        if any(record.get(key) != value for key, value in wanted.items()):
            raise ValueError('Saved field provenance differs from the actual requested run')
        meta = manifest.get('meta', {})
        wanted_meta = dict(src=[str(Path(merged).resolve())], channels=list(channels),
                           per_plate=True, estimator='polynomial', degree=4,
                           max_fields=count, dark=0.0, application_contract_version=1,
                           channel_index_space='persisted-intensity-axis',
                           estimated_from_intensity_state='raw')
        if any(meta.get(key) != value for key, value in wanted_meta.items()):
            raise ValueError('Saved model provenance differs from the actual requested run')
        plane = np.array(archive['field0'], copy=True)
    return plane, dict(manifest=manifest, **verify_plane(plane, channels=channels, shape=shape))


def corrected_reference(image, plane, dark=0):
    """The documented float32 formula with integer round-to-nearest and clipping."""
    image = np.asarray(image)
    field = np.moveaxis(np.asarray(plane, dtype=np.float32), 0, -1)
    if image.shape != field.shape:
        raise ValueError('Correction reference geometry differs')
    verify_plane(plane, channels=range(image.shape[-1]), shape=image.shape[:2])
    value = (image.astype(np.float32) - np.float32(dark)) * (np.float32(1) / field)
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        value = np.clip(np.rint(value), limits.min, limits.max)
    return value.astype(image.dtype)


def verify_corrected(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Corrected pixel shape or dtype differs')
    if not np.array_equal(actual, expected):
        raise ValueError('Corrected pixels differ from the independent formula')
    return dict(pixels_checked=int(actual.size), unequal_pixels=0)


def qc_reference(images, plane, bin_size):
    """Independently check the in-sample radial slopes, not optical correctness."""
    height, width = plane.shape[-2:]
    factor = int(bin_size)
    rows, cols = height // factor, width // factor

    def block_means(array):
        cropped = np.asarray(array[:rows * factor, :cols * factor], dtype=np.float64)
        return cropped.reshape(rows, factor, cols, factor).mean(axis=(1, 3)).astype(np.float32)

    fields = np.stack([block_means(image) for image in images])
    levels = np.median(fields.reshape(len(fields), -1), axis=1)
    if not np.all(np.isfinite(levels) & (levels > 0)):
        raise ValueError('QC reference needs positive finite field medians')
    before = np.median(fields / levels[:, None, None], axis=0)
    after = before * block_means(np.float32(1) / plane[0])
    y = ((np.arange(rows) * factor + (factor - 1) / 2) - (height - 1) / 2) / ((height - 1) / 2)
    x = ((np.arange(cols) * factor + (factor - 1) / 2) - (width - 1) / 2) / ((width - 1) / 2)
    radius = (np.sqrt(y[:, None] ** 2 + x[None, :] ** 2) / np.sqrt(2)).ravel()
    centered = radius - radius.mean()

    def slope(array):
        value = array.astype(np.float64).ravel()
        value = value / value.mean()
        return float(np.dot(centered, value - value.mean()) / np.dot(centered, centered))

    first, second = slope(before), slope(after)
    return dict(slope_before=first, slope_after=second,
                bias_removed_pct=100 * (1 - abs(second) / abs(first)) if abs(first) > 1e-12 else 0,
                n_fields=len(fields), binned_shape=[rows, cols],
                in_sample_not_independent_validation=True)
