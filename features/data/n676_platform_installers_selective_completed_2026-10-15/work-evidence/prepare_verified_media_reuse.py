#!/usr/bin/env python3
"""Validate a new selective04 candidate before preparing old readback reuse.

No network or publication call. Root must separately invoke resume-receipt,
which validates the existing remote branch/tag/commit relationships read-only.
"""
from pathlib import Path, PurePosixPath
import sys

ROOT = Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'tools/tutorials')]

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess

WORK = Path(__file__).resolve().parent
STAGE = WORK.parents[1] / 'host/stage-04-selective-20261015'
OLD = STAGE / 'release-candidate-append-34v6631r'
SNAPSHOT = WORK / 'hosted-media-publication-receipt.snapshot.json'
OLD_MANIFEST_SHA = '9ac6c7ee63581e1ab77e59957f25fdcdd3854aded7e74fc4645ad7d021de2172'
SNAPSHOT_SHA = '0dbc95c9d8c9f308c694f089361b8de18fa99046d7a20976e2ad602eced2695f'
COMMIT = '80a4c7222e259d5c437c54705087e000f247085b'
FILES, BYTES = 5370, 8693151860


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def read(path):
    return json.loads(path.read_text())


def binding(path):
    value = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
            size += len(block)
    return {'path': str(path.resolve()), 'bytes': size, 'sha256': value.hexdigest()}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def records(manifest):
    rows = {}
    for row in manifest['files']:
        name = row['path']
        rel = PurePosixPath(name)
        require(rel.parts and not rel.is_absolute() and '..' not in rel.parts
                and rel.as_posix() == name, 'Unsafe manifest path')
        if not name.startswith('media_host/'):
            continue
        require(name not in rows and type(row['bytes']) is int and row['bytes'] >= 0
                and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Invalid/duplicate hosted record')
        rows[name] = {'path': name, 'bytes': row['bytes'], 'sha256': row['sha256']}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    output = args.output_dir.absolute()
    require(not args.candidate.is_symlink() and candidate.parent == STAGE
            and candidate != OLD and candidate.is_dir(), 'Expected a distinct actual new04 candidate')
    require(output.parent == WORK and output.name.startswith('verified-media-reuse-')
            and not output.exists() and not output.is_symlink(), 'Use a fresh direct WORK/verified-media-reuse-* output')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
    require(re.fullmatch('[0-9a-f]{40}', args.source_commit)
            and git('rev-parse', 'HEAD') == args.source_commit, 'Actual current committed source required')
    require(not git('status', '--porcelain', '--', 'spacr', 'tools/tutorials', 'tests',
                    ':(exclude)tools/tutorials/release_candidate'), 'Source must be clean outside generated checkpoint')

    old_manifest_path = OLD / 'release-manifest.json'
    old_manifest_binding = binding(old_manifest_path)
    snapshot_binding = binding(SNAPSHOT)
    require(old_manifest_binding['sha256'] == OLD_MANIFEST_SHA
            and snapshot_binding['sha256'] == SNAPSHOT_SHA, 'Original immutable evidence changed')
    old_manifest = read(old_manifest_path)
    receipt = read(SNAPSHOT)
    checked = receipt['readback']
    require(receipt['manifest_sha256'] == OLD_MANIFEST_SHA
            and receipt['commit'] == checked['commit'] == COMMIT
            and receipt['media_files'] == FILES and receipt['media_bytes'] == BYTES
            and receipt['branch'] == 'n676-selective04-20261015'
            and receipt['tag'] == 'n676-selective04-20261015-verified'
            and receipt['media_root'] == 'https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/' + COMMIT,
            'Original receipt identity mismatch')
    require(checked['passed'] is True
            and checked['files_expected'] == checked['metadata_matched'] == checked['downloaded_sha256_matched'] == FILES
            and checked['bytes_read_back'] == BYTES
            and checked['metadata_failures'] == checked['download_failures'] == [],
            'Original readback must cover every exact media file')

    # Owning validator hashes ALL actual candidate bytes and checks all84
    # browser rows against this exact held, not-uploaded candidate manifest.
    from publish_release_candidate import require_verified_candidate
    manifest = require_verified_candidate(candidate)
    browser = read(candidate / 'checks/candidate-browser-checks.json')
    cases = browser['ready_playback_cases']
    require(manifest['ready_lessons'] == manifest['routes'] == 84
            and len(cases) == len({row['lesson'] for row in cases}) == 84
            and all(row['passed'] is True for row in cases), 'Exact current84 unique passed routes required')
    new_manifest_binding = binding(candidate / 'release-manifest.json')
    require(new_manifest_binding['sha256'] != OLD_MANIFEST_SHA, 'Expected the new player candidate manifest')
    before, after = records(old_manifest), records(manifest)
    require(len(before) == len(after) == FILES and before == after
            and sum(row['bytes'] for row in after.values()) == BYTES
            and manifest['media_host_bytes'] == BYTES, 'Hosted path/bytes/SHA map changed; cannot reuse readback')

    placeholder_path = candidate / 'checks/placeholder-mutation-checks.json'
    placeholder = read(placeholder_path)
    require(placeholder['passed'] is True and placeholder['baseline_before_and_after_passed'] is True
            and placeholder['source_files_modified'] is False and len(placeholder['mutations']) == 2,
            'Actual current placeholder guards required')
    require(placeholder['player_sha256'] == binding(candidate / 'web/app_v2.js')['sha256'],
            'Placeholder evidence must describe the new player')
    checkpoint_root = ROOT / 'tools/tutorials/release_candidate'
    checkpoint_path = checkpoint_root / 'checkpoint.json'
    checkpoint = read(checkpoint_path)
    require(Path(checkpoint['private_candidate']).resolve() == candidate
            and checkpoint['manifest_sha256'] == new_manifest_binding['sha256']
            and checkpoint['browser_checks_complete'] is True
            and checkpoint['published'] is False and checkpoint['release_hold'] is True,
            'Coherent held current checkpoint required')
    rows = checkpoint['files']
    require(len(rows) == len({row['path'] for row in rows}) == 145, 'Expected exact current145-file checkpoint')
    candidate_records = {row['path']: row for row in manifest['files']}
    for row in rows:
        relative = PurePosixPath(row['path'])
        path = checkpoint_root / relative
        require(relative.parts and not relative.is_absolute() and '..' not in relative.parts
                and path.resolve().is_relative_to(checkpoint_root) and not path.is_symlink(), 'Unsafe checkpoint path')
        actual = binding(path)
        require(actual['sha256'] == row['sha256'] and actual['bytes'] == row['bytes'], 'Checkpoint file changed')
        if row['path'].startswith('web/'):
            current = candidate_records.get(row['path'])
            require(current is not None and current['sha256'] == row['sha256']
                    and current['bytes'] == row['bytes'], 'Checkpoint web file is not this actual candidate')
    for name, actual_path in [('release-manifest.json', candidate / 'release-manifest.json'),
                              ('candidate-browser-checks.json', candidate / 'checks/candidate-browser-checks.json'),
                              ('placeholder-mutation-checks.json', placeholder_path)]:
        require(binding(checkpoint_root / name)['sha256'] == binding(actual_path)['sha256'], 'Checkpoint/candidate evidence mismatch')
    require(binding(old_manifest_path) == old_manifest_binding and binding(SNAPSHOT) == snapshot_binding
            and binding(candidate / 'release-manifest.json') == new_manifest_binding
            and git('rev-parse', 'HEAD') == args.source_commit, 'Inputs changed during validation')

    # No output exists until every actual local prerequisite above passed.
    output.mkdir()
    readback_path = output / 'original-readback.json'
    with readback_path.open('x') as stream:
        json.dump(checked, stream, indent=2)
        stream.write('\n')
    require(read(readback_path) == checked, 'Readback object passthrough changed')
    proof = {'schema': 1, 'lesson': '04_platform_installers',
             'status': 'local_reuse_prerequisites_passed_remote_resume_not_executed',
             'source_commit': args.source_commit, 'candidate': str(candidate),
             'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'old_manifest': old_manifest_binding, 'new_manifest': new_manifest_binding,
             'original_receipt': snapshot_binding, 'original_readback_passthrough': binding(readback_path),
             'immutable_media_commit': COMMIT, 'branch': receipt['branch'], 'tag': receipt['tag'],
             'hosted_file_count': FILES, 'hosted_bytes': BYTES,
             'hosted_records_equal': True, 'hosted_records': [after[name] for name in sorted(after)],
             'checkpoint': binding(checkpoint_path), 'checkpoint_files_checked': 145,
             'candidate_browser_checks': binding(candidate / 'checks/candidate-browser-checks.json'),
             'placeholder_checks': binding(placeholder_path),
             'checks': {'owning_actual_candidate_bytes_validated': True, 'exact_hosted_maps_equal': True,
                        'original_readback_object_unchanged': True, 'new_candidate_current84_browser_verified': True},
             'limits': ['Local prerequisites only; no new readback, network, upload, tag, Pages or publication operation.',
                        'Root must call existing resume-receipt, retaining its remote branch/tag/main guards.',
                        'Original snapshot/candidate are untouched; this proof is not a new remote-byte verification.']}
    with (output / 'reuse-proof.json').open('x') as stream:
        json.dump(proof, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'proof': binding(output / 'reuse-proof.json'), 'readback': binding(readback_path)}))


if __name__ == '__main__':
    main()
