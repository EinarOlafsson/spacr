"""Preserve and independently count the existing synthetic plaque masks."""
from collections import Counter
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage

from replication_demo import digest
from stage_lesson import read, write


def label_areas(mask):
    """Count one connected object per positive label, independently of regionprops."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError('Expected a two-dimensional nonnegative integer label mask')
    labels, counts = np.unique(mask, return_counts=True)
    result = []
    for identity, count in zip(labels, counts):
        if identity == 0:
            continue
        _, components = ndimage.label(mask == identity, structure=np.ones((3, 3), dtype=bool))
        if components != 1:
            raise ValueError('A teaching label is disconnected; unique labels are not object counts')
        result.append(int(count))
    return result


def area_summary(areas, px_per_mm=None):
    """Pixel counts stay pixels unless a real positive scale was supplied."""
    if not areas or any(not math.isfinite(n) or n <= 0 for n in areas):
        raise ValueError('Expected positive finite areas for the teaching objects')
    if px_per_mm is not None and (not math.isfinite(px_per_mm) or px_per_mm <= 0):
        raise ValueError('A supplied scale must be positive and finite')
    mean, std = statistics.mean(areas), statistics.pstdev(areas)
    return dict(plaque_count=len(areas), average_size=mean, std_dev_size=std,
        px_per_mm=px_per_mm, average_size_mm2=None if px_per_mm is None else mean/px_per_mm**2,
        std_dev_size_mm2=None if px_per_mm is None else std/px_per_mm**2)


def require_preserved(files):
    for record in files:
        if digest(record['source']) != record['sha256'] or digest(record['copy']) != record['sha256']:
            raise ValueError('An original or copied plaque input changed')


def prepare(stage):
    stage = Path(stage)
    source = stage.parent/'synthetic/plaque'
    old = read(source/'manifest.json')
    runs = stage/'plaque_runs'
    runs.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='SYNTHETIC-existing-masks-', dir=runs))
    (root/'masks').mkdir()
    files, references = [], {}
    for record in old['records']:
        name = record['file']
        if Path(name).name != name:
            raise ValueError('The teaching manifest must name a single image filename')
        for relative in (Path(name), Path('masks')/name):
            original, copied = source/relative, root/relative
            fingerprint = digest(original)
            shutil.copy2(original, copied)
            files.append(dict(source=str(original), copy=str(copied), sha256=fingerprint))
        with Image.open(root/'masks'/name) as image:
            mask = np.asarray(image)
        with Image.open(root/name) as image:
            if image.size != tuple(reversed(mask.shape)) or image.size != (768, 768):
                raise ValueError('Image and label dimensions differ from the preserved teaching example')
        areas = label_areas(mask)
        if areas != record['label_areas'] or len(areas) != record['expected_objects']:
            raise ValueError('The independently counted synthetic objects changed')
        references[name] = dict(areas=areas, summary=area_summary(areas))
    if len(references) != 4 or sum(len(r['areas']) for r in references.values()) != 10:
        raise ValueError('Expected exactly four teaching images and ten objects')
    require_preserved(files)
    manifest = dict(source=str(source), root=str(root), files=files, references=references,
        synthetic=True, acquired_images=False, masks_precomputed=True,
        old_database_copied=False, calibration_available=False, biological_effect_claim=False)
    write(root/'manifest.json', manifest)
    return manifest


def _equal(got, wanted):
    if wanted is None:
        return got is None
    return isinstance(got, (int, float)) and math.isclose(got, wanted, rel_tol=1e-12, abs_tol=1e-12)


def verify_tables(tables, reference):
    """Match exact identities, null calibration and independently counted pixel areas."""
    if set(tables) != {'summary', 'stats', 'details'}:
        raise ValueError('Plaque output tables differ')
    checked = 0
    for name in ('summary', 'stats'):
        rows = tables[name]
        names = [row['file'] for row in rows]
        if len(names) != len(set(names)) or set(names) != set(reference):
            raise ValueError('Plaque result image identities differ')
        for row in rows:
            ref = reference[row['file']]['summary']
            wanted = dict(ref, well_diameter_px=None)
            if name == 'summary':
                wanted['object_count'] = wanted.pop('plaque_count')
                wanted.pop('std_dev_size')
                wanted.pop('std_dev_size_mm2')
            for key, value in wanted.items():
                if not _equal(row.get(key), value) or key not in row:
                    raise ValueError('Plaque output value differs: '+key)
                checked += 1
    expected = Counter((name, float(area)) for name, value in reference.items() for area in value['areas'])
    got = Counter((row['file'], float(row['plaque_size'])) for row in tables['details'])
    if got != expected or any(row.get('plaque_size_mm2', 'missing') is not None for row in tables['details']):
        raise ValueError('Individual plaque areas or uncalibrated units differ')
    return dict(images=len(reference), plaques=sum(expected.values()), numeric_or_null_fields_checked=checked+2*sum(expected.values()))


def verify_database(path, reference):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        db.row_factory = sqlite3.Row
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if set(names) != {'summary', 'stats', 'details'}:
            raise ValueError('Unexpected actual plaque database tables')
        tables = {name: [dict(row) for row in db.execute('SELECT * FROM "'+name+'"')] for name in names}
    return verify_tables(tables, reference), tables
