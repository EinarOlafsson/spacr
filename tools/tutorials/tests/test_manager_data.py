"""The cleanup recording must refuse any target except its verified private copy."""
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manager_data import verify_bind, verify_crop_plan
from capture_report import snapshot_source


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
