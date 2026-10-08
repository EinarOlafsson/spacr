"""Verify the source-bound 5636 12-shard coverage phase and separate verdicts."""

import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ARCHIVE = Path(__file__).resolve().parent
SOURCE = '5636eac60938f5d6021eaf73546ff8a30ae17c32'
RUN = 37763414112


def _git(*args):
    return subprocess.check_output(('git', *args), cwd=ARCHIVE).decode().strip()


def _read(name, from_git):
    if not from_git:
        return (ARCHIVE / name).read_bytes()
    root = Path(_git('rev-parse', '--show-toplevel'))
    relative = ARCHIVE.relative_to(root).as_posix()
    return subprocess.check_output(('git', 'show', f'HEAD:{relative}/{name}'), cwd=ARCHIVE)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    from_git = sys.argv[1:] == ['--git']
    if sys.argv[1:] and not from_git:
        raise SystemExit('usage: python verify.py [--git]')
    manifest = json.loads(_read('MANIFEST.json', from_git))
    assert manifest['source_sha'] == SOURCE
    names = set()
    for row in manifest['payloads']:
        data = _read(row['path'], from_git)
        assert len(data) == row['bytes'] and _sha(data) == row['sha256']
        names.add(row['path'])
    assert len(names) == len(manifest['payloads'])
    if not from_git:
        assert {p.name for p in ARCHIVE.iterdir() if p.is_file()} == names | {'MANIFEST.json'}

    receipt = json.loads(_read('receipt.json', from_git))
    assert receipt['source_sha'] == SOURCE
    assert receipt['source_tree'] == _git('rev-parse', SOURCE + '^{tree}')
    assert receipt['run_id'] == RUN and receipt['run_attempt'] == 1
    for path, digest in {**receipt['workflow_blobs'], **receipt['guard_blobs']}.items():
        assert digest == _git('rev-parse', f'{SOURCE}:{path}')
    run = json.loads(_read('run-phase.json', from_git))
    jobs = json.loads(_read('jobs-phase.json', from_git))['jobs']
    assert run['id'] == RUN and run['head_sha'] == SOURCE
    assert run['status'] == receipt['run_snapshot_status']
    jobs_by_id = {job['id']: job for job in jobs}
    assert len(jobs_by_id) == len(jobs)

    shards = receipt['coverage_shards']
    assert {row['number'] for row in shards} == set(range(12))
    assert sum(row['conclusion'] == 'success' for row in shards) == receipt['coverage_success'] == 11
    assert {row['number'] for row in shards if row['conclusion'] == 'failure'} == {0}
    assert receipt['coverage_failure'] == 1
    for row in shards:
        number = row['number']
        job = json.loads(_read(f'coverage{number}-job.json', from_git))
        assert job['id'] == row['job_id'] and job['run_id'] == RUN
        assert job['head_sha'] == SOURCE
        assert job['conclusion'] == row['conclusion'] == jobs_by_id[job['id']]['conclusion']
        assert job['completed_at'] == row['completed_at']
        raw = gzip.decompress(_read(f'coverage{number}.log.gz', from_git))
        assert _sha(raw) == row['raw_log_sha256']
        assert f'spacr-coverage-data-{RUN}-1-{number}'.encode() in raw
    failed = gzip.decompress(_read('coverage0.log.gz', from_git))
    assert b'Fatal Python error: Segmentation fault' in failed
    assert b'worker gw1 was killed by SIGSEGV' in failed
    assert receipt['failure_node'].encode() in failed
    assert b'PySide6 6.12.0 -- Qt runtime 6.12.0 -- Qt compiled 6.12.0' in failed

    native_zip = _read('coverage0-native.zip', from_git)
    assert _sha(native_zip) == receipt['native_artifact']['zip_sha256']
    assert receipt['native_artifact']['id'] == 11552052325
    with zipfile.ZipFile(BytesIO(native_zip)) as native:
        assert native.testzip() is None
        native_text = native.read('native-core-backtrace.txt')
        journal = native.read(receipt['native_artifact']['crashed_process_journal'])
    assert b'ordinary extraction rejected: ELF PID or executable differs' in native_text
    assert b'No regular ELF core was available; no native backtrace can be recovered.' in native_text
    assert b'"pid":7463' in journal and b'"worker":"gw1"' in journal

    combine = json.loads(_read('combine-job.json', from_git))
    assert combine['id'] == receipt['combine_job_id']
    assert combine['run_id'] == RUN and combine['head_sha'] == SOURCE
    assert combine['conclusion'] == receipt['combine_conclusion'] == 'failure'
    assert jobs_by_id[combine['id']]['conclusion'] == 'failure'
    steps = {step['name']: step['conclusion'] for step in combine['steps']}
    assert {name: steps[name] for name in receipt['combine_step_results']} == receipt['combine_step_results']
    assert steps['Combine process data and write coverage.py JSON'] == 'success'
    assert steps['Gate on the per-module coverage ratchet baseline'] == 'success'
    assert steps['Require every coverage shard to have passed its tests'] == 'failure'
    combine_log = gzip.decompress(_read('combine.log.gz', from_git))
    assert b'spaCR shipped-module coverage ratchet: PASS' in combine_log
    assert b'coverage shards finished with failure' in combine_log

    report = _read('module-report.zip', from_git)
    assert _sha(report) == receipt['report_artifact']['zip_sha256']
    assert receipt['report_artifact']['id'] == 11551698279
    with zipfile.ZipFile(BytesIO(report)) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == {'coverage.json', 'module-coverage-ratchet.json',
                                     'module-coverage-ratchet.txt'}
        numerical_bytes = z.read('module-coverage-ratchet.json')
    assert _sha(numerical_bytes) == receipt['report_artifact']['ratchet_json_sha256']
    numerical = json.loads(numerical_bytes)
    assert numerical['status'] == receipt['numerical_status'] == 'pass'
    assert numerical['summary'] == receipt['numerical_summary']
    assert numerical['summary']['shipped_modules'] == numerical['summary']['modules_checked'] == 664
    assert numerical['summary']['failed_modules'] == numerical['summary']['unconfirmed_modules'] == 0
    integrity = numerical['measurement_integrity']
    assert integrity == receipt['measurement_integrity']
    assert integrity['status'] == 'complete'
    assert integrity['shard_count'] == integrity['shards_with_records'] == 12
    assert integrity['loss_kinds'] == {'segfault': 1} and len(integrity['recovered']) == 1
    assert integrity['issues'] == []
    print('5636 coverage: 664-module numerical PASS; 11 selected-test shards pass, one SIGSEGV failure')


if __name__ == '__main__':
    main()
