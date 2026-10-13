"""Assemble a clearly labelled comparison index from two real run snapshots.

Only the copied index's project/path and its relocation note change. Original
run IDs, versions, timestamps, settings and full-content fingerprints survive.
No new scientific run, synthetic output or automatic GUI import is claimed.
"""
from copy import deepcopy
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3

from build_evaluation_example import sha
from stage_lesson import DEFAULT_STAGE, write

SOURCES=(('measure_fresh','d86c136e7261'),('measure_production','caa0bf7ca2ce'))
TABLES=('cell','nucleus','pathogen','cytoplasm','png_list')


def relocate_record(original, source_hash, source_size, project, destination, registry):
    if (original['kind']!='measurements-db' or original['module']!='measure'
            or original['status']!='complete' or original['fingerprint_method']!='sha256'
            or original['fingerprint']!=source_hash or original['size_bytes']!=source_size):
        raise ValueError('Require a complete actual Measure output matching its original full fingerprint')
    row=deepcopy(original)
    extra=json.loads(row['extra_json'])
    extra['tutorial_relocation']=dict(original_project=row['project'],original_path=row['path'],
        original_registry=str(registry),original_registry_sha256=sha(registry),
        scope='Only the real measurements artifact is included; this is not a full project archive')
    row.update(project=str(project),path=str(destination),extra_json=json.dumps(extra,sort_keys=True))
    if {key for key in row if row[key]!=original[key]}-{'project','path','extra_json'}:
        raise ValueError('Relocation changed historical run metadata')
    return row


def read_identity(database):
    with closing(sqlite3.connect(Path(database).as_uri()+'?mode=ro&immutable=1',uri=True)) as con:
        if con.execute('pragma quick_check').fetchone()!=('ok',):
            raise ValueError('An original measurements database fails integrity checks')
        counts={table:con.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in TABLES}
        keys={}
        for table in TABLES[:-1]:
            rows=con.execute('SELECT plateID,rowID,columnID,fieldID,object_label FROM '+table).fetchall()
            ordered=sorted(tuple(map(str,row)) for row in rows)
            if len(ordered)!=len(set(ordered)):raise ValueError('Object identities are duplicated')
            keys[table]=ordered
        fields=sorted(set(row[:4] for rows in keys.values() for row in rows))
    return dict(counts=counts,object_keys=keys,fields=fields)


def require_same_identity(a,b):
    if a['object_keys']!=b['object_keys'] or a['fields']!=b['fields']:
        raise ValueError('These snapshots do not describe the same object and field identities')
    if a['counts']!=b['counts']:
        raise ValueError('The narrated unchanged counts are absent')


def prepare(stage=DEFAULT_STAGE):
    if os.environ.get('SPACR_ARTIFACTS_DB'):
        raise ValueError('Do not write to an external artifact registry')
    stage=Path(stage);destination=stage/'run_compare_verified_snapshots_v1'
    if destination.exists():raise FileExistsError('Preserve the existing comparison workspace')
    original_hashes={};selected=[];identities=[]
    for prefix,run in SOURCES:
        root=stage/prefix/'example_data/plate1';registry=root/'artifacts.db'
        source=root/'measurements/measurements.db'
        for path in (registry,source):
            wal=Path(str(path)+'-wal')
            if wal.exists() and wal.stat().st_size:
                raise ValueError('An input has uncheckpointed writes')
            original_hashes[str(path)]=sha(path)
        with closing(sqlite3.connect(registry.as_uri()+'?mode=ro&immutable=1',uri=True)) as con:
            con.row_factory=sqlite3.Row
            rows=con.execute('SELECT * FROM artifacts WHERE run_id=? AND kind=?',(run,'measurements-db')).fetchall()
            if len(rows)!=1:raise ValueError('Expected exactly one real measurements artifact per run')
            row=dict(rows[0])
            if con.execute('SELECT COUNT(*) FROM artifact_inputs WHERE artifact_id=?',(row['artifact_id'],)).fetchone()[0]:
                raise ValueError('Do not silently drop recorded upstream edges')
        identity=read_identity(source);identities.append(identity)
        copied=destination/'snapshots'/run/'measurements.db'
        relocated=relocate_record(row,original_hashes[str(source)],source.stat().st_size,
                                  destination,copied,registry)
        selected.append((source,copied,row,relocated))
    require_same_identity(*identities)
    if selected[0][2]['run_id']==selected[1][2]['run_id']:
        raise ValueError('Do not compare a run with itself as two separate experiments')
    # No producer is called. This creates the schema only, then transports
    # actual historical rows without Registry.register's new version/time.
    from spacr.artifacts import Registry
    destination.mkdir()
    registry=Registry(project=str(destination))
    for source,copied,original,relocated in selected:
        copied.parent.mkdir(parents=True);shutil.copy2(source,copied)
        if sha(copied)!=original['fingerprint']:raise ValueError('A copied snapshot differs')
        columns=list(relocated)
        with sqlite3.connect(destination/'artifacts.db') as con:
            con.execute('INSERT INTO artifacts ('+','.join(columns)+') VALUES ('+
                ','.join('?' for _ in columns)+')',[relocated[key] for key in columns])
    from spacr.run_compare import runs_in,compare_runs
    runs={run.run_id:run for run in runs_in(registry,str(destination))}
    if set(runs)!={run for _,run in SOURCES}:raise ValueError('The real API did not list both transported runs')
    comparison=compare_runs(runs[SOURCES[0][1]],runs[SOURCES[1][1]])
    if not comparison.comparable or comparison.forced:
        raise ValueError('The real comparison refuses these snapshots')
    if any(sha(path)!=digest for path,digest in original_hashes.items()):
        raise ValueError('An original source changed during preparation')
    proof=dict(accepted=True,scope='Explicitly transported real Measure snapshots; not a new analysis or complete project archive',
        project=str(destination),source_hashes=original_hashes,
        records=[dict(original=old,transported=new) for _,_,old,new in selected],
        counts=identities[0]['counts'],fields=identities[0]['fields'],
        object_identity_rows_checked=sum(len(v) for identity in identities for v in identity['object_keys'].values()),
        historical_metadata_preserved=True,settings_unchanged_between_runs=selected[0][2]['settings_json']==selected[1][2]['settings_json'],
        selected_artifacts_only=True,registry_paths_explicitly_relocated=True,
        original_sources_unchanged=True,synthetic_runs=False,scientific_accuracy_validated=False,
        app_source_modified=False,published=False)
    write(destination/'preparation.json',proof)
    print(destination)
    return proof


if __name__=='__main__':prepare()
