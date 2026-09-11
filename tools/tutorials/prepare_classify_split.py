"""Export an explicit, tiny CV tutorial split from authoritative crop metadata.

No segmentation, labels, pixel values or source files are modified. The new
filenames are the database's PRCFO identities, not a guess from legacy image
names. This is an explicit dataset-preparation workaround, not an app repair.
Only eight existing examples per class/well are selected, before any fitting;
the balanced demonstration cannot estimate real-world class prevalence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

TRAIN_WELLS = {('plate1', 'r12', 'c1'), ('plate1', 'r5', 'c2')}
TEST_WELLS = {('plate1', 'r12', 'c2'), ('plate1', 'r5', 'c1')}
MARKER = '.spacr_crop_format.json'


def marker_profile(raw):
    """Compare storage semantics, not each original folder's timestamp."""
    marker = json.loads(raw)
    if (marker.get('spacr_crop_format') != 3 or marker.get('channel_order') != 'declared_rgb'
            or marker.get('narrowing') != 'high-byte'):
        raise ValueError('Expected the current declared-RGB high-byte example')
    marker.pop('updated_utc', None)
    return json.dumps(marker, sort_keys=True)


def plan(rows, per_class_well=8):
    """Validate canonical metadata and choose before looking at any model score."""
    if per_class_well < 1:
        raise ValueError('Selection must be nonempty')
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        identity = str(row['prcfo'])
        if identity in seen:
            raise ValueError('Duplicate metadata identity')
        seen.add(identity)
        match = re.fullmatch(r'(plate1)_(r\d+)_(c\d+)_(f\d+)_(o\d+)', identity)
        well = tuple(str(row[k]) for k in ('plateID', 'rowID', 'columnID'))
        if not match or tuple(match.groups()[:3]) != well:
            raise ValueError('Canonical identity disagrees with database well metadata')
        if well not in TRAIN_WELLS | TEST_WELLS or row['infected'] not in (1, 2):
            raise ValueError('Unexpected well or missing/unknown example annotation')
        groups[(well, int(row['infected']))].append(dict(row))
    expected = {(well, value) for well in TRAIN_WELLS | TEST_WELLS for value in (1, 2)}
    if set(groups) != expected or any(len(v) < per_class_well for v in groups.values()):
        raise ValueError('Every actual well must contain enough examples of both saved labels')
    selected = []
    for (well, label), candidates in sorted(groups.items()):
        candidates.sort(key=lambda r: hashlib.sha256(('42:' + r['prcfo']).encode()).hexdigest())
        for row in candidates[:per_class_well]:
            selected.append(dict(row, well=list(well), split='train' if well in TRAIN_WELLS else 'test',
                                 class_name=f'infected_{label}', filename=row['prcfo'] + '.png'))
    return selected


def prepare(source, destination):
    """Create a fresh, manifested split; preserve every input byte and marker."""
    from PIL import Image
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Use a new destination outside the source project')
    database = source / 'measurements/measurements.db'
    db_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            'SELECT png_path,prcfo,plateID,rowID,columnID,infected FROM png_list')]
    records = plan(rows)
    prefix = Path('/home/olafsson/.cache/spacr/example_data/plate1')
    fingerprints, marker_bytes = set(), set()
    for row in records:
        raw = (source / Path(row['png_path']).relative_to(prefix)).resolve()
        if not raw.is_relative_to(source) or raw.is_symlink():
            raise ValueError('Source crop leaves the project')
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        if digest in fingerprints:
            raise ValueError('Byte-identical examples would undermine this tiny demonstration')
        fingerprints.add(digest)
        with Image.open(raw) as im:
            if im.mode != 'RGB' or im.size != (224, 224):
                raise ValueError('Expected the unchanged RGB224x224 example crop')
            im.verify()
        marker = raw.parent / MARKER
        raw_marker = marker.read_bytes()
        marker_bytes.add(marker_profile(raw_marker))
        row.update(source=str(raw), source_sha256=digest,
                   source_marker=str(marker), source_marker_sha256=hashlib.sha256(raw_marker).hexdigest())
    if len(marker_bytes) != 1:
        raise ValueError('The copied example crops disagree on their storage format')
    # Validate the application's current parser against database truth before
    # training. This check is NOT used to invent either the names or the split.
    from spacr.classifier_evaluation import sample_identity
    for row in records:
        parsed = sample_identity(row['filename'])
        if parsed['well'] != '_'.join(row['well']) or parsed['object'] != row['prcfo']:
            raise ValueError('Current application cannot parse the exported canonical identity')
    destination.mkdir(parents=True)
    for row in records:
        target = destination / row['split'] / row['class_name'] / row['filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row['source'], target)
        marker = target.parent / MARKER
        if not marker.exists():
            shutil.copy2(row['source_marker'], marker)
        row['target'] = str(target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError('The exported image differs from its original')
    manifest = {'accepted': True, 'source_project': str(source), 'source_database': str(database),
                'source_database_sha256': db_hash, 'destination': str(destination),
                'selection': 'SHA256 of 42:PRCFO; eight per existing label per actual well',
                'class_values': {'infected_1': 1, 'infected_2': 2}, 'records': records,
                'train_wells': sorted(map(list, TRAIN_WELLS)), 'test_wells': sorted(map(list, TEST_WELLS)),
                'biological_labels_independently_validated': False, 'population_prevalence_preserved': False,
                'filename_parser_fixed': False, 'source_unchanged': True, 'model_started': False}
    if hashlib.sha256(database.read_bytes()).hexdigest() != db_hash:
        raise ValueError('The source database changed')
    for row in records:
        for key in ('source', 'source_marker'):
            if hashlib.sha256(Path(row[key]).read_bytes()).hexdigest() != row[key + '_sha256']:
                raise ValueError('An original input changed')
    (destination / 'tutorial_input_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'destination': str(destination), 'objects': len(records),
                      'train_objects': sum(r['split'] == 'train' for r in records),
                      'test_objects': sum(r['split'] == 'test' for r in records),
                      'source_unchanged': True, 'model_started': False}))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination)
