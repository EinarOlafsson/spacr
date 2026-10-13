"""Independent, bounded checks for a read-only real-journal tutorial capture."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def snapshot(root):
    root = Path(root)
    files = sorted(path for path in root.rglob('*') if path.is_file())
    if not files or sum(path.stat().st_size for path in files) > 64 * 1024**2:
        raise ValueError('Expected a nonempty bounded private journal')
    if any(path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
           for path in files):
        raise ValueError('Journal snapshot must not follow external files')
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def verify_preserved(root, before):
    after = snapshot(root)
    if not before or after != before:
        raise ValueError('Private journal files changed')
    return {'files': len(after), 'unchanged': True}


def read_record(root, run_id):
    if Path(run_id).name != run_id:
        raise ValueError('Expected a single run identity')
    folder = Path(root) / run_id
    manifest = json.loads((folder / 'manifest.json').read_text())
    raw = (folder / 'settings.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['settings_sha256']:
        raise ValueError('Recorded settings hash does not match the snapshot')
    return {'run_id': run_id, 'manifest': manifest, 'settings': json.loads(raw)}


def verify_panels(record, panels):
    """Compare actual detail strings with raw files, not the GUI's own records."""
    manifest, settings = record['manifest'], record['settings']
    summary = json.loads(panels['overview'])
    expected_summary = {'run_id': record['run_id'], 'module': manifest['app_key'],
                        'status': manifest['status'], 'started_utc': manifest['start_utc'],
                        'ended_utc': manifest['end_utc']}
    expected_summary.update({key: manifest['performance'][key] for key in
                             ('input_files', 'input_bytes', 'output_files', 'output_bytes')})
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f'History overview differs: {key}')
    if json.loads(panels['settings']) != settings:
        raise ValueError('History settings differ')
    if json.loads(panels['files']) != {
        'inputs': manifest['input_hashes'], 'outputs': manifest['output_hashes'],
        'models': manifest.get('model_files') or manifest.get('model_hashes') or {}}:
        raise ValueError('History file/model records differ')
    if json.loads(panels['environment']) != {
        'environment': manifest['env'], 'seeds': manifest['seeds'],
        'settings_sha256': manifest['settings_sha256'],
        'input_tree_sha256': manifest['input_tree_sha256'],
        'output_tree_sha256': manifest['output_tree_sha256'],
        'manifest_schema': manifest['schema_version']}:
        raise ValueError('History environment differs')
    warnings = list(dict.fromkeys(manifest['warnings'] + manifest['provenance_warnings']))
    problems = ('WARNINGS\n' + ('\n\n'.join(warnings) if warnings else 'None recorded.')
                + '\n\nFAILURE TRACEBACK\n' + (manifest['traceback'] or 'None.'))
    if panels['problems'] != problems:
        raise ValueError('History warnings or failure differ')
    return {'run_id': record['run_id'], 'all_five_panels_match_raw_journal': True,
            'settings_count': len(settings), 'warning_count': len(warnings),
            'failure_present': bool(manifest['traceback']),
            'recorded_input_files': len(manifest['input_hashes']),
            'recorded_output_files': len(manifest['output_hashes'])}


def verify_visible(actual, expected):
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('Visible run identities differ from expected journals')
    return list(actual)
