"""The cleanup recording must refuse any target except its verified private copy."""
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manager_data import verify_bind, verify_crop_plan
from capture_report import snapshot_source
from manager_data import verify_pruned_files, verify_pruned_registry, verify_archive_plan, verify_archived_files
import copy
import json


@pytest.fixture
def example(tmp_path):
    source = tmp_path/'private'
    (source/'data').mkdir(parents=True)
    (source/'data/one.png').write_bytes(b'actual-one')
    (source/'data/two.png').write_bytes(b'actual-two')
    (source/'measurements.db').write_bytes(b'preserved-measurement')
    original = tmp_path/'original'
    shutil.copytree(source, original)
    inputs = {'source': str(source), 'clone': str(source), 'original_readonly': str(original),
              'source_files': snapshot_source(original), 'clone_files': snapshot_source(source)}
    listing = [str(source/'data/one.png'), str(source/'data/two.png')]
    plan = SimpleNamespace(candidates=[SimpleNamespace(path=str(source/'data'))],
                           total_files=2, total_bytes=20, file_list=lambda: (tuple(listing), False),
                           token='exact-plan-token')
    return source, inputs, plan


def test_positive_exact_private_files_and_bytes_are_verified(example):
    source, inputs, plan = example
    assert verify_bind(inputs) == source
    proof = verify_crop_plan(plan, inputs)
    assert proof['files'] == 2 and proof['bytes'] == 20
    assert proof['file_list'] == [str(source/'data/one.png'), str(source/'data/two.png')]


@pytest.mark.parametrize('kind', ['whole-project', 'database', 'empty', 'duplicate-candidate'])
def test_broader_or_different_target_is_refused(example, kind):
    source, inputs, plan = example
    assert verify_crop_plan(plan, inputs)['files'] == 2
    replacement = {'whole-project': [str(source)], 'database': [str(source/'measurements.db')],
                   'empty': [], 'duplicate-candidate': [str(source/'data'), str(source/'data')]}[kind]
    plan.candidates = [SimpleNamespace(path=p) for p in replacement]
    with pytest.raises(ValueError, match='exactly the scoped private crops directory'):
        verify_crop_plan(plan, inputs)


@pytest.mark.parametrize('kind', ['truncated', 'missing', 'extra', 'duplicate', 'count', 'bytes'])
def test_incomplete_or_different_file_list_is_refused(example, kind):
    source, inputs, plan = example
    assert verify_crop_plan(plan, inputs)['files'] == 2
    listing, _ = plan.file_list()
    if kind == 'truncated': plan.file_list = lambda: (listing, True)
    if kind == 'missing': plan.file_list = lambda: (listing[:1], False)
    if kind == 'extra': plan.file_list = lambda: ((*listing, str(source/'measurements.db')), False)
    if kind == 'duplicate': plan.file_list = lambda: ((*listing, listing[0]), False)
    if kind == 'count': plan.total_files += 1
    if kind == 'bytes': plan.total_bytes += 1
    with pytest.raises(ValueError, match='independently copied file set'):
        verify_crop_plan(plan, inputs)


def test_same_size_changed_crop_fails_the_content_comparison(example):
    source, inputs, plan = example
    assert verify_crop_plan(plan, inputs)['files'] == 2
    (source/'data/one.png').write_bytes(b'changed!!!')
    with pytest.raises(ValueError, match='crop bytes changed'):
        verify_crop_plan(plan, inputs)


def test_unbound_identical_copy_cannot_authorize_original_deletion(example, tmp_path):
    source, inputs, plan = example
    assert verify_bind(inputs) == source
    other = tmp_path/'unbound-copy'
    shutil.copytree(source, other)
    inputs['clone'] = str(other)
    with pytest.raises(ValueError, match='isolated writable clone'):
        verify_crop_plan(plan, inputs)


def test_original_and_writable_source_must_be_different_inodes(example):
    source, inputs, plan = example
    assert verify_bind(inputs) == source
    inputs['original_readonly'] = str(source)
    with pytest.raises(ValueError, match='isolated writable clone'):
        verify_crop_plan(plan, inputs)


@pytest.mark.parametrize('change', ['left-crop', 'lost-measurement', 'changed-measurement', 'new-file'])
def test_cleanup_preserves_every_other_byte(example, change):
    _, inputs, _ = example
    before = inputs['clone_files']
    after = {k: v for k, v in before.items() if not k.startswith('data/')}
    assert verify_pruned_files(before, after)['unchanged_non_registry_files'] == 1
    after = copy.deepcopy(after)
    if change == 'left-crop': after['data/one.png'] = before['data/one.png']
    if change == 'lost-measurement': after.pop('measurements.db')
    if change == 'changed-measurement': after['measurements.db']['sha256'] = 'wrong'
    if change == 'new-file': after['unrelated'] = before['data/one.png']
    with pytest.raises(ValueError, match='Cleanup changed files'):
        verify_pruned_files(before, after)


@pytest.mark.parametrize('change', ['lost-row', 'changed-recipe', 'other-row', 'wrong-bytes', 'no-mark'])
def test_cleanup_retains_recipe_and_exact_mark(change):
    before = [dict(artifact_id='a', kind='crops', extra_json='{}', settings='recorded'),
              dict(artifact_id='b', kind='measurements-db', extra_json='{}', settings='recorded')]
    after = copy.deepcopy(before)
    mark = dict(pruned_utc='real-time', pruned_by_spacr='real-version', pruned_freed_bytes=20)
    after[0]['extra_json'] = json.dumps(mark)
    assert verify_pruned_registry(before, after, 20)[0]['artifact_id'] == 'a'
    if change == 'lost-row': after.pop()
    if change == 'changed-recipe': after[0]['settings'] = 'changed'
    if change == 'other-row': after[1]['settings'] = 'changed'
    if change == 'wrong-bytes': after[0]['extra_json'] = json.dumps({**mark, 'pruned_freed_bytes': 21})
    if change == 'no-mark': after[0]['extra_json'] = '{}'
    with pytest.raises(ValueError):
        verify_pruned_registry(before, after, 20)


@pytest.mark.parametrize('change', ['wrong-root', 'wrong-destination', 'not-whole', 'missing-item',
                                     'duplicate-item', 'occupied-destination', 'count', 'bytes'])
def test_archive_target_is_exact_and_empty(example, tmp_path, change):
    source, inputs, _ = example
    archive = tmp_path/'archive'
    archive.mkdir()
    inputs['archive'] = str(archive)
    inv = snapshot_source(source)
    plan = SimpleNamespace(root=str(source), destination=str(archive), whole_project=True,
                           items=[SimpleNamespace(source=str(p), destination=str(archive/p.name))
                                  for p in source.iterdir()], total_files=len(inv),
                           total_bytes=sum(r['bytes'] for r in inv.values()))
    assert verify_archive_plan(plan, inputs) == inv
    if change == 'wrong-root': plan.root = str(tmp_path)
    if change == 'wrong-destination': plan.destination = str(tmp_path)
    if change == 'not-whole': plan.whole_project = False
    if change == 'missing-item': plan.items.pop()
    if change == 'duplicate-item': plan.items.append(plan.items[0])
    if change == 'occupied-destination': (archive/'keep.txt').write_text('existing data')
    if change == 'count': plan.total_files += 1
    if change == 'bytes': plan.total_bytes += 1
    with pytest.raises(ValueError, match='exact private project'):
        verify_archive_plan(plan, inputs)


@pytest.mark.parametrize('change', ['missing', 'changed', 'extra', 'no-manifest', 'no-ledger', 'left-behind'])
def test_archive_every_file_is_accounted_for(example, change):
    _, inputs, _ = example
    before = inputs['clone_files']
    dest = copy.deepcopy(before)
    dest['spacr_archive.json'] = {'bytes': 12, 'sha256': 'manifest'}
    origin = {'spacr_archive_log.json': {'bytes': 12, 'sha256': 'ledger'}}
    assert verify_archived_files(before, dest, origin)['byte_identical_non_registry_files'] == 3
    if change == 'missing': dest.pop('measurements.db')
    if change == 'changed': dest['measurements.db']['sha256'] = 'changed'
    if change == 'extra': dest['unexpected'] = before['data/one.png']
    if change == 'no-manifest': dest.pop('spacr_archive.json')
    if change == 'no-ledger': origin.clear()
    if change == 'left-behind': origin['measurements.db'] = before['measurements.db']
    with pytest.raises(ValueError, match='Archive did not preserve'):
        verify_archived_files(before, dest, origin)
