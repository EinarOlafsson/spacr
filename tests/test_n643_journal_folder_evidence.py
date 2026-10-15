"""Age pruning requires journal evidence, including real partial-run layouts."""
import json
import os
import time
from pathlib import Path

import pytest

from spacr import run_journal as journal


@pytest.fixture
def root(tmp_path, monkeypatch):
    home = tmp_path / 'private-home'
    monkeypatch.setattr(journal, '_spacr_home', lambda: home)
    folder = home / 'runs'
    folder.mkdir(parents=True)
    monkeypatch.setattr(journal, 'runs_root', lambda: folder)
    return folder


def old(folder):
    moment = time.time() - 90 * 86400
    for path in sorted(folder.rglob('*'), reverse=True):
        os.utime(path, (moment, moment), follow_symlinks=False)
    os.utime(folder, (moment, moment))
    return folder


def manifest(folder, **overrides):
    data = {'app_key': 'mask', 'start_utc': '2020-01-01T00:00:00+00:00', 'status': 'success'}
    data.update(overrides)
    (folder / 'manifest.json').write_text(json.dumps(data))


@pytest.mark.parametrize('name', ['unrelated-notebooks', '2020-01-01_000000_abcdef12__mask'])
def test_populated_unknown_folder_is_preserved_in_plan_and_revalidation(root, name):
    folder = root / name
    folder.mkdir()
    (folder / 'my-experiment.txt').write_bytes(b'private experiment')
    old(folder)
    plan = journal._prune_plan('run_folders', 30, 0)
    assert plan['delete'] == []
    plan['delete'].append((str(folder), 18))
    assert journal._prune(plan)[:2] == (0, 0)
    assert (folder / 'my-experiment.txt').read_bytes() == b'private experiment'


@pytest.mark.parametrize('renamed', [False, True])
def test_actual_serialized_run_remains_eligible_after_rename(root, renamed):
    folder = journal._new_run_dir('mask')
    run = journal.Run(app_key='mask', settings={'example': 1}, dir=folder)
    run.status = 'success'
    run._write_settings()
    run._write_manifest()
    if renamed:
        folder = folder.rename(root / 'my-labelled-run')
    old(folder)
    plan = journal._prune_plan('run_folders', 30, 0)
    assert [Path(path) for path, size in plan['delete']] == [folder]
    assert journal._prune(plan)[0] == 1
    assert not folder.exists()


def test_legacy_manifest_without_schema_version_remains_eligible(root):
    folder = root / 'legacy-renamed-run'
    folder.mkdir()
    manifest(folder)
    old(folder)
    assert journal._prune(journal._prune_plan('run_folders', 30, 0))[0] == 1


@pytest.mark.parametrize('settings', [False, True])
def test_native_empty_or_settings_bootstrap_remains_eligible(root, settings):
    folder = journal._new_run_dir('mask')
    if settings:
        journal.Run(app_key='mask', settings={}, dir=folder)._write_settings()
    old(folder)
    assert journal._prune(journal._prune_plan('run_folders', 30, 0))[0] == 1


@pytest.mark.parametrize('payload', [[], None, {}, {'status': []}, {'schema_version': 999}, {'schema_version': None}, {'status': 'unknown'}, {'start_utc': 'not a timestamp'}])
def test_invalid_or_unknown_manifest_is_preserved(root, payload):
    folder = journal._new_run_dir('mask')
    journal.Run(app_key='mask', settings={}, dir=folder)._write_settings()
    if isinstance(payload, dict):
        manifest(folder, **payload)
        if not payload:
            (folder / 'manifest.json').write_text('{}')
    else:
        (folder / 'manifest.json').write_text(json.dumps(payload))
    old(folder)
    assert journal._prune_plan('run_folders', 30, 0)['delete'] == []
    assert folder.exists()


@pytest.mark.parametrize('marker', ['manifest.json', 'settings.json'])
def test_oversized_journal_marker_is_preserved(root, marker):
    folder = journal._new_run_dir('mask')
    payload = {'app_key': 'mask', 'start_utc': '2020-01-01', 'status': 'success', 'padding': 'x' * (1024 * 1024)}
    (folder / marker).write_text(json.dumps(payload))
    old(folder)
    assert journal._prune_plan('run_folders', 30, 0)['delete'] == []


def test_native_settings_bootstrap_with_unknown_populated_file_is_preserved(root):
    folder = journal._new_run_dir('mask')
    journal.Run(app_key='mask', settings={}, dir=folder)._write_settings()
    (folder / 'my-notebook.txt').write_bytes(b'private notebook')
    old(folder)
    assert journal._prune_plan('run_folders', 30, 0)['delete'] == []


def test_native_settings_bootstrap_with_empty_outputs_and_real_environment_is_eligible(root):
    folder = journal._new_run_dir('mask')
    journal.Run(app_key='mask', settings={}, dir=folder)._write_settings()
    environment = folder / 'environment'
    environment.mkdir()
    (environment / 'requirements-lock.txt').write_text('spacr==1.5.1.4\n')
    old(folder)
    assert journal._prune(journal._prune_plan('run_folders', 30, 0))[0] == 1


def test_symlinked_manifest_does_not_turn_unknown_folder_into_a_journal(root):
    folder = root / 'private-data'
    folder.mkdir()
    marker = root.parent / 'external-manifest.json'
    marker.write_text(json.dumps({'app_key': 'mask', 'start_utc': '2020-01-01', 'status': 'success'}))
    (folder / 'manifest.json').symlink_to(marker)
    old(folder)
    assert journal._prune_plan('run_folders', 30, 0)['delete'] == []
    assert marker.exists()


def test_invalidated_manifest_after_planning_is_not_deleted(root):
    folder = root / 'renamed-run'
    folder.mkdir()
    manifest(folder)
    old(folder)
    plan = journal._prune_plan('run_folders', 30, 0)
    (folder / 'manifest.json').write_text('{}')
    old(folder)
    deleted, freed, refused = journal._prune(plan)
    assert (deleted, freed) == (0, 0)
    assert refused and folder.exists()


def test_active_run_protection_precedes_journal_evidence_reads(root, monkeypatch):
    monkeypatch.setattr(journal, '_env_snapshot', lambda: {'packages': {}})
    monkeypatch.setattr(journal, '_write_environment_lock', lambda *args: {})
    monkeypatch.setattr(journal, '_check_lock_for_run', lambda *args: None)
    monkeypatch.setattr(journal, 'begin_recording', lambda *args, **kwargs: None)
    monkeypatch.setattr(journal, 'finish_recording', lambda *args, **kwargs: None)
    monkeypatch.setattr(journal, '_notify_run_finished', lambda *args: None)
    for name in ('_capture_initial_provenance', '_capture_final_provenance', '_snapshot_log_tail'):
        monkeypatch.setattr(journal.Run, name, lambda self: None)
    with journal.open_run('mask', {}) as run:
        old(run.dir)
        monkeypatch.setattr(journal, '_small_journal_mapping', lambda *args: pytest.fail('active marker read'))
        plan = journal._prune_plan('run_folders', 30, 0)
        assert plan['delete'] == []
        plan['delete'].append((str(run.dir), 0))
        deleted, freed, refused = journal._prune(plan)
        assert (deleted, freed) == (0, 0)
        assert refused == [run.dir.name + ': in use']


@pytest.mark.parametrize('schema', [1, 2])
def test_known_legacy_manifest_schema_remains_eligible(root, schema):
    folder = root / f'legacy-schema-{schema}'
    folder.mkdir()
    manifest(folder, schema_version=schema)
    old(folder)
    assert journal._prune(journal._prune_plan('run_folders', 30, 0))[0] == 1


def test_unreadable_manifest_is_preserved(root, monkeypatch):
    folder = root / 'renamed-run'
    folder.mkdir()
    manifest(folder)
    old(folder)
    original = Path.open

    def unreadable(path, *args, **kwargs):
        if path == folder / 'manifest.json':
            raise PermissionError('controlled unreadable journal')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', unreadable)
    assert journal._prune_plan('run_folders', 30, 0)['delete'] == []
    assert folder.exists()


@pytest.mark.parametrize('child', ['outputs', 'environment'])
def test_native_bootstrap_with_populated_unknown_contents_is_preserved(root, child):
    folder = journal._new_run_dir('mask')
    journal.Run(app_key='mask', settings={}, dir=folder)._write_settings()
    nested = folder / child
    nested.mkdir(exist_ok=True)
    (nested / 'private-notebook.txt').write_bytes(b'private notes')
    old(folder)
    assert journal._prune_plan('run_folders', 30, 0)['delete'] == []
