"""Relocate two unchanged tutorial snapshots into a NEW comparison directory.

Run inside the extracted Run_Compare_real_snapshots folder with spaCR installed:
    python prepare_run_compare_download.py /absolute/path/to/new-comparison
This is explicit artifact transport, NOT a new Measure run or a GUI importer.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import os
import sys


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def prepare(destination, root=None):
    if os.environ.get('SPACR_ARTIFACTS_DB'):
        raise ValueError('Unset SPACR_ARTIFACTS_DB: never write an external registry')
    root=Path(root or Path(__file__).resolve().parent).resolve()
    destination=Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Choose a NEW folder; existing projects are never overwritten')
    manifest=json.loads((root/'source_records.json').read_text())
    entries=manifest['records']
    if len(entries)!=2 or len({e['original']['run_id'] for e in entries})!=2:
        raise ValueError('Require exactly two distinct historical runs')
    ready=[]
    for entry in entries:
        original=entry['original'];run=original['run_id']
        if not re.fullmatch('[a-f0-9]{12}',run):
            raise ValueError('Invalid snapshot directory identifier')
        source=root/'snapshots'/run/'measurements.db'
        if (original['kind']!='measurements-db' or original['module']!='measure'
                or original['status']!='complete' or original['fingerprint_method']!='sha256'
                or source.stat().st_size!=original['size_bytes']
                or sha(source)!=original['fingerprint']):
            raise ValueError('Snapshot does not match its complete historical fingerprint')
        row=deepcopy(original);extra=json.loads(row['extra_json'])
        extra['tutorial_relocation']={
            'original_project':row['project'],'original_path':row['path'],
            'original_registry_sha256':entry['original_registry_sha256'],
            'scope':'Only two selected real measurements artifacts, not full project archives'}
        copied=destination/'snapshots'/run/'measurements.db'
        row.update(project=str(destination),path=str(copied),extra_json=json.dumps(extra,sort_keys=True))
        ready.append((source,copied,row))
    from spacr.artifacts import Registry
    from spacr.run_compare import runs_in,compare_runs
    destination.mkdir(parents=True)
    registry=Registry(project=str(destination))
    with sqlite3.connect(destination/'artifacts.db') as con:
        columns=[r[1] for r in con.execute('PRAGMA table_info(artifacts)')]
        for source,copied,row in ready:
            if set(row)!=set(columns):raise ValueError('Installed artifact schema differs; do not guess a conversion')
            copied.parent.mkdir(parents=True);shutil.copy2(source,copied)
            if sha(copied)!=row['fingerprint']:raise ValueError('Copied snapshot changed')
            con.execute('INSERT INTO artifacts ('+','.join(columns)+') VALUES ('+
                        ','.join('?' for _ in columns)+')',[row[c] for c in columns])
    runs={r.run_id:r for r in runs_in(registry,str(destination))}
    a,b=(entry['original']['run_id'] for entry in entries)
    result=compare_runs(runs[a],runs[b])
    if not result.comparable or result.forced:
        raise ValueError('Installed comparison did not accept these snapshots')
    print('Prepared selected-artifact comparison workspace:',destination)
    print('Open Home > Data > Run Compare, Browse to this folder, select A and B, then Compare.')
    print('Historical versions, times, settings and output bytes are retained; no analysis was rerun.')
    return [row for _,_,row in ready]


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit(__doc__)
    prepare(sys.argv[1])
