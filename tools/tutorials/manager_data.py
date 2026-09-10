"""Isolate an existing, genuinely registered tutorial project without rebasing it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import tempfile

from capture_report import copy_private_source, require_unchanged, snapshot_source

SOURCE_RELATIVE = 'external_mask_runs/example-pj_qwr_5/project'


def prepare(stage):
    stage = Path(stage).resolve()
    if os.environ.get('SPACR_ARTIFACTS_DB'):
        raise ValueError('Data Manager capture must not use an external registry')
    source = stage / SOURCE_RELATIVE
    wal = source / 'artifacts.db-wal'
    if wal.exists() and wal.stat().st_size:
        raise ValueError('The source registry has uncheckpointed writes')
    with sqlite3.connect((source/'artifacts.db').as_uri()+'?mode=ro&immutable=1', uri=True) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(row) for row in con.execute('SELECT * FROM artifacts ORDER BY artifact_id')]
    if (len(rows) != 3 or {row['kind'] for row in rows} !=
            {'measurements-db', 'crops', 'resource-log'} or
            any(row['module'] != 'measure' or row['status'] != 'complete' for row in rows)):
        raise ValueError('The real example artifact registry differs from the scoped source')
    parent = stage / 'data_manager_runs'
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='real-project-', dir=parent))
    clone = root / 'project'
    original_readonly = root / 'original_readonly'
    original_readonly.mkdir()
    archive = root / 'archive'
    archive.mkdir()
    before, copied = copy_private_source(source, clone)
    return {'source': str(source), 'root': str(root), 'clone': str(clone),
            'original_readonly': str(original_readonly), 'archive': str(archive),
            'source_files': before, 'clone_files': copied, 'artifact_rows': rows,
            'registry_rewritten': False, 'synthetic_artifacts': False}


def verify_bind(inputs):
    source, clone, original = (Path(inputs[key]) for key in ('source', 'clone', 'original_readonly'))
    identity = lambda path: (path.stat().st_dev, path.stat().st_ino)
    if identity(source) != identity(clone) or identity(source) == identity(original):
        raise ValueError('Data Manager source is not the isolated writable clone')
    require_unchanged(inputs['source_files'], snapshot_source(original))
    return source


def verify_original(inputs):
    require_unchanged(inputs['source_files'], snapshot_source(inputs['original_readonly']))
    return {'files': len(inputs['source_files']), 'byte_identical': True}


def verify_crop_plan(plan, inputs):
    """A future destructive demonstration may target only this copied data/ tree."""
    source = verify_bind(inputs)
    expected = {str(source/name): row['bytes'] for name, row in inputs['clone_files'].items()
                if name.startswith('data/')}
    paths = [candidate.path for candidate in plan.candidates]
    if paths != [str(source/'data')] or not expected:
        raise ValueError('The plan is not exactly the scoped private crops directory')
    listing, truncated = plan.file_list()
    if (plan.total_files != len(expected) or plan.total_bytes != sum(expected.values())
            or truncated or len(listing) != len(expected) or set(listing) != set(expected)):
        raise ValueError('The crop plan differs from the independently copied file set')
    copied_crops = {name: row for name, row in inputs['clone_files'].items() if name.startswith('data/')}
    actual_crops = {'data/'+name: row for name, row in snapshot_source(source/'data').items()}
    if actual_crops != copied_crops:
        raise ValueError('The copied crop bytes changed before confirmation')
    return {'path': paths[0], 'files': len(expected), 'bytes': sum(expected.values()),
            'file_list': sorted(expected), 'token': plan.token}


REGISTRY_FILES = {'artifacts.db', 'artifacts.db-wal', 'artifacts.db-shm'}


def verify_pruned_files(before, after):
    """Only crops and registry bookkeeping may differ after confirmed cleanup."""
    expected = {name: row for name, row in before.items()
                if not name.startswith('data/') and name not in REGISTRY_FILES}
    actual = {name: row for name, row in after.items() if name not in REGISTRY_FILES}
    if not expected or actual != expected:
        raise ValueError('Cleanup changed files other than the exact crops and registry')
    return {'unchanged_non_registry_files': len(expected),
            'deleted_files': sorted(name for name in before if name.startswith('data/'))}


def registry_rows(root):
    with sqlite3.connect((Path(root)/'artifacts.db').as_uri()+'?mode=ro&immutable=1', uri=True) as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute('SELECT * FROM artifacts ORDER BY artifact_id')]


def verify_pruned_registry(before, after, freed_bytes):
    if [r['artifact_id'] for r in before] != [r['artifact_id'] for r in after]:
        raise ValueError('Cleanup lost or invented registry records')
    marks = []
    for old, new in zip(before, after):
        if old['kind'] != 'crops':
            if old != new:
                raise ValueError('Cleanup changed another artifact record')
            continue
        old_extra, new_extra = json.loads(old['extra_json']), json.loads(new['extra_json'])
        expected_keys = {'pruned_utc', 'pruned_by_spacr', 'pruned_freed_bytes'}
        if (set(new_extra) != set(old_extra) | expected_keys or
                any(new_extra.get(k) != v for k, v in old_extra.items()) or
                not new_extra.get('pruned_utc') or not new_extra.get('pruned_by_spacr') or
                new_extra.get('pruned_freed_bytes') != freed_bytes or
                {k: v for k, v in old.items() if k != 'extra_json'} !=
                {k: v for k, v in new.items() if k != 'extra_json'}):
            raise ValueError('Cleanup did not preserve the crop recipe with exact prune metadata')
        marks.append({'artifact_id': old['artifact_id'],
                      **{k: new_extra[k] for k in sorted(expected_keys)}})
    if len(marks) != 1:
        raise ValueError('Expected exactly one retained crop recipe')
    return marks


def verify_archive_plan(plan, inputs):
    source = verify_bind(inputs)
    destination = Path(inputs['archive'])
    inventory = snapshot_source(source)
    expected = {str(child): str(destination/child.name) for child in source.iterdir()}
    actual = {item.source: item.destination for item in plan.items}
    if (str(source) != plan.root or str(destination) != plan.destination or
            not plan.whole_project or not expected or len(plan.items) != len(expected) or
            actual != expected or any(destination.iterdir()) or
            plan.total_files != len(inventory) or
            plan.total_bytes != sum(row['bytes'] for row in inventory.values())):
        raise ValueError('Archive plan is not the exact private project and empty private destination')
    return inventory


def verify_archived_files(before, destination_files, origin_files):
    expected = {k: v for k, v in before.items() if k not in REGISTRY_FILES}
    actual = {k: v for k, v in destination_files.items()
              if k not in REGISTRY_FILES | {'spacr_archive.json'}}
    if (not expected or actual != expected or 'spacr_archive.json' not in destination_files or
            set(origin_files) != {'spacr_archive_log.json'}):
        raise ValueError('Archive did not preserve every non-registry file and leave only its origin ledger')
    return {'byte_identical_non_registry_files': len(expected),
            'origin_files': sorted(origin_files), 'destination_files': len(destination_files)}
