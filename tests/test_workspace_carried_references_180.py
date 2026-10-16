"""Verified workspace copies become real provider inputs after relocation."""
import copy
import os
from pathlib import Path
import shutil

import pytest

from spacr import workspace


class Reader:
    def __init__(self, state):
        self.state = state
        self.received = None
        self.content = None
    def workspace_state(self):
        return self.state
    def apply_workspace_state(self, state):
        self.received = state
        self.content = Path(state['input']['files'][0]).read_bytes()
        state['input']['files'].append('provider-local edit')
        return True


def _saved(tmp_path, *, mode='copy', hash_files=True):
    original = tmp_path / 'original table.csv'
    original.write_bytes(b'exact original rows\n')
    state = {'input': {'files': [str(original)]}, 'literal': 'prefix ' + str(original)}
    if mode == 'reference':
        state[workspace.CARRY_KEY] = [str(original)]
    run = tmp_path / 'run'
    workspace.save(run, workspace.collect({'table': Reader(state)}, hash_files=hash_files), mode=mode)
    return original, run, workspace.load(run)


@pytest.mark.parametrize('mode', ['copy', 'reference'])
def test_relocated_carried_copy_is_read_without_mutating_document(tmp_path, mode):
    original, run, document = _saved(tmp_path, mode=mode)
    original.unlink()
    relocated = tmp_path / 'moved run'
    shutil.move(run, relocated)
    before = copy.deepcopy(document)
    reader = Reader({})
    report = workspace.restore({'table': reader}, document, run_dir=relocated)
    assert report['restored'] == ['table'] and report['skipped'] == []
    assert reader.content == b'exact original rows\n'
    assert Path(reader.received['input']['files'][0]).is_relative_to(relocated / workspace.FILES_DIR)
    assert reader.received['literal'] == 'prefix ' + str(original)
    assert workspace.CARRY_KEY not in reader.received
    assert document == before
    assert report['files'][0]['state'] == 'carried'


def test_existing_changed_original_keeps_existing_reference_behavior(tmp_path):
    original, run, document = _saved(tmp_path)
    original.write_bytes(b'current user-edited rows\n')
    reader = Reader({})
    report = workspace.restore({'table': reader}, document, run_dir=run)
    assert reader.content == b'current user-edited rows\n'
    assert reader.received['input']['files'][0] == str(original)
    assert report['files'][0]['state'] == 'changed'


def test_reference_only_missing_file_is_not_replaced_by_a_guess(tmp_path):
    original = tmp_path / 'original.csv'
    original.write_bytes(b'data')
    run = tmp_path / 'run'
    workspace.save(run, workspace.collect({'table': Reader({'input': {'files': [str(original)]}})}))
    original.unlink()
    reader = Reader({})
    report = workspace.restore({'table': reader}, workspace.load(run), run_dir=run)
    assert report['restored'] == [] and reader.content is None
    assert report['files'][0]['state'] == 'missing'
    assert reader.received['input']['files'][0] == str(original)


def test_corrupted_carried_copy_is_reported_changed_and_never_read_by_provider(tmp_path):
    original, run, document = _saved(tmp_path)
    original.unlink()
    carried = run / document['files'][0]['copied']
    carried.write_bytes(b'corrupted different rows')
    reader = Reader({})
    report = workspace.restore({'table': reader}, document, run_dir=run)
    assert report['restored'] == [] and reader.content is None
    assert reader.received['input']['files'][0] == str(original)
    assert report['files'][0]['state'] == 'changed'
    assert report['files'][0]['skipped'] == 'copied_digest_mismatch'
    assert 'copied_digest_mismatch' in workspace.report_text(report)


@pytest.mark.parametrize('escape', ['traversal', 'absolute', 'file_link', 'directory_link', 'link_loop'])
def test_escaping_carried_path_is_not_hashed_or_passed_to_provider(tmp_path, monkeypatch, escape):
    original, run, document = _saved(tmp_path)
    original.unlink()
    outside = tmp_path / 'outside'
    outside.mkdir()
    target = outside / 'same.csv'
    target.write_bytes(b'exact original rows\n')
    if escape == 'traversal':
        copied = os.path.relpath(target, run)
    elif escape == 'absolute':
        copied = str(target)
    else:
        link = run / workspace.FILES_DIR / 'link'
        try:
            link.symlink_to(link.name if escape == 'link_loop' else outside if escape == 'directory_link' else target,
                            target_is_directory=escape == 'directory_link')
        except OSError as error:
            pytest.skip(str(error))
        copied = str(link.relative_to(run) / 'same.csv') if escape == 'directory_link' else str(link.relative_to(run))
    document['files'][0]['copied'] = copied
    hashes = []
    original_hash = workspace.hash_file
    def hash_file(path, *args, **kwargs):
        hashes.append(Path(path).resolve())
        return original_hash(path, *args, **kwargs)
    monkeypatch.setattr(workspace, 'hash_file', hash_file)
    before = copy.deepcopy(document)
    reader = Reader({})
    report = workspace.restore({'table': reader}, document, run_dir=run)
    assert report['restored'] == [] and reader.content is None
    assert hashes == []
    assert reader.received['input']['files'][0] == str(original)
    assert document == before
    assert report['files'][0]['state'] == 'missing'
    assert report['files'][0]['skipped'] == 'copied_path_outside_bundle'
    assert 'copied_path_outside_bundle' in workspace.report_text(report)
    assert target.read_bytes() == b'exact original rows\n'


def test_unhashed_inventory_gets_real_copy_digest_before_restore(tmp_path):
    original, run, document = _saved(tmp_path, hash_files=False)
    assert document['files'][0]['sha256'] == workspace.hash_file(original)
    original.unlink()
    reader = Reader({})
    assert workspace.restore({'table': reader}, document, run_dir=run)['restored'] == ['table']
    assert reader.content == b'exact original rows\n'


def test_legacy_copy_without_digest_is_visibly_unverified(tmp_path):
    original, run, document = _saved(tmp_path)
    original.unlink()
    del document['files'][0]['sha256']
    reader = Reader({})
    report = workspace.restore({'table': reader}, document, run_dir=run)
    assert report['restored'] == [] and reader.content is None
    assert report['files'][0]['state'] == 'missing'
    assert report['files'][0]['skipped'] == 'copied_digest_unverified'


def test_no_bundle_root_does_not_rewrite_missing_original(tmp_path):
    original, run, document = _saved(tmp_path)
    original.unlink()
    reader = Reader({})
    report = workspace.restore({'table': reader}, document)
    assert report['restored'] == [] and reader.content is None
    assert report['files'][0]['state'] == 'missing'


def test_unresolvable_tilde_literal_remains_unchanged(tmp_path):
    original, run, document = _saved(tmp_path)
    original.unlink()
    document['sections']['table']['note'] = '~spacr_no_such_account_180/file'
    reader = Reader({})
    report = workspace.restore({'table': reader}, document, run_dir=run)
    assert report['restored'] == ['table']
    assert reader.received['note'] == '~spacr_no_such_account_180/file'


def test_copy_without_obtainable_digest_never_replaces_live_input(tmp_path, monkeypatch):
    original = tmp_path / 'original.csv'
    original.write_bytes(b'original rows')
    document = workspace.collect({'table': Reader({'input': {'files': [str(original)]}})},
                                 hash_files=False)
    before = copy.deepcopy(document)
    monkeypatch.setattr(workspace, 'hash_file', lambda path: None)
    run = tmp_path / 'run'
    workspace.save(run, document, mode='copy')
    assert document == before
    assert original.read_bytes() == b'original rows'
    original.unlink()
    saved = workspace.load(run)
    saved_before = copy.deepcopy(saved)
    reader = Reader({})
    reader.content = b'previous live rows'
    report = workspace.restore({'table': reader}, saved, run_dir=run)
    assert report['restored'] == []
    assert report['skipped'][0]['section'] == 'table'
    assert reader.content == b'previous live rows'
    assert reader.received['input']['files'] == [str(original)]
    assert report['files'][0]['state'] == 'missing'
    assert report['files'][0]['skipped'] == 'copied_digest_unverified'
    assert saved == saved_before


def test_relocated_legacy_tuple_paths_keep_shape_and_document(tmp_path):
    original, run, document = _saved(tmp_path)
    original.unlink()
    document['sections']['table']['selection'] = (str(original), 'gene-name')
    before = copy.deepcopy(document)
    reader = Reader({})
    report = workspace.restore({'table': reader}, document, run_dir=run)
    assert report['restored'] == ['table']
    assert reader.received['selection'] == (reader.received['input']['files'][0], 'gene-name')
    assert isinstance(reader.received['selection'], tuple)
    assert document == before


def test_deep_saved_state_remains_bounded_and_keeps_literal_references(tmp_path):
    original, run, document = _saved(tmp_path)
    original.unlink()
    nested = {'literal': str(original), 'stable_id': 'user-defined'}
    for _ in range(20):
        nested = {'child': nested}
    document['sections']['table']['deep_state'] = nested
    before = copy.deepcopy(document)
    reader = Reader({})
    report = workspace.restore({'table': reader}, document, run_dir=run)
    assert report['restored'] == ['table']
    restored = reader.received['deep_state']
    for _ in range(20):
        restored = restored['child']
    assert restored == {'literal': str(original), 'stable_id': 'user-defined'}
    assert reader.content == b'exact original rows\n'
    assert document == before
