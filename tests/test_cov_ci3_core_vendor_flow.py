"""Vendor watch grouping, readiness, snapshots and workspace guards (F548)."""

import json
import os

import numpy as np
import pytest
import tifffile

from spacr import core
from spacr.cancellation import PipelineCancelled
from tests.test_watch_folder_and_analyse import MASK, Recorder
from tests.test_watch_native_t_series_f548 import _watch as _series_watch
from tests.test_watch_nested_f548 import _fast
from tests.test_watch_timelapse_series_f548 import _converted_series


def _context(src, seen, **extra):
    return {'seen': seen, 'src': str(src), 'settle': 0.5, 'warned': set(),
            'settings': {}, 'patterns': {}, **extra}


def _seen(*names, changed=0.0):
    return {name: {'changed': changed, 'first': changed, 'identity': [1, 2, 3]}
            for name in names}


def test_groups_let_stop_through_a_description(tmp_path, monkeypatch):
    def stop(path, name):
        raise PipelineCancelled('stop while describing')

    monkeypatch.setattr(core, '_watch_vendor_describe', stop)
    context = _context(tmp_path, _seen('a.nd2'))
    with pytest.raises(PipelineCancelled, match='stop while describing'):
        core._watch_vendor_groups(context, 100.0)
    assert 'a.nd2' not in context.get('vendor_units', {})


def _micromanager(folder):
    folder.mkdir(parents=True)
    (folder / 'metadata.txt').write_text(json.dumps({
        'Summary': {'Frames': 2, 'ChNames': ['DAPI']},
        'FrameKey-0-0-0': {'FileName': 't0.tif'},
        'FrameKey-1-0-0': {'FileName': 't1.tif'}}))


def test_two_companions_with_one_field_key_both_wait(tmp_path):
    for folder in ('x_1', 'x1'):
        _micromanager(tmp_path / folder)
    names = [f'{folder}/{member}' for folder in ('x_1', 'x1')
             for member in ('metadata.txt', 't0.tif', 't1.tif')]
    context = _context(tmp_path, _seen(*names))
    groups = core._watch_vendor_groups(context, 100.0)
    assert list(groups) == ['x1_A01_1']
    assert {member for member, _channel in groups['x1_A01_1']} == set(names)
    reason = context['vendor_fields']['x1_A01_1']['reason']
    assert 'two acquisitions give the field key x1_A01_1' in reason


def test_a_stack_sharing_a_waiting_container_key_waits(tmp_path):
    (tmp_path / 'foo.nd2').write_bytes(b'growing')
    tifffile.imwrite(tmp_path / 'foo.tif', np.ones((4, 4), np.uint16))
    seen = _seen('foo.tif')
    seen.update(_seen('foo.nd2', changed=99.9))
    context = _context(tmp_path, seen)
    groups = core._watch_vendor_groups(context, 100.0)
    assert groups == {'foo': [('foo.nd2', None), ('foo.tif', None)]}
    assert 'two acquisitions give the field key foo' in context['vendor_fields']['foo']['reason']


def _pages(folder, count):
    planes = {}
    for t in range(count):
        tifffile.imwrite(folder / f'p{t}.tif', np.full((4, 4), t + 1, np.uint16))
        planes[(t, 0, 0)] = (f'p{t}.tif', ('page', 0))
    return planes


def test_described_ready_decodes_once_per_signature(tmp_path, capsys):
    planes = _pages(tmp_path, 2)
    good = {'folder': '', 'reason': None, 'sizes': {'T': 2, 'Z': 1, 'C': 1},
            'planes': planes}
    bad = dict(good, reason='0 of 2 declared plane files are present')
    context = _context(tmp_path, {}, settle=60.0, vendor='timelapse',
                       vendor_fields={'good': good, 'bad': bad})
    signature = {'p0.tif': [10, 1], 'p1.tif': [10, 1]}
    assert core._watch_described_ready(context, 'bad', signature, 100.0) is False
    assert core._watch_described_ready(context, 'bad', signature, 101.0) is False
    printed = capsys.readouterr().out
    assert printed.count('bad is not complete yet (0 of 2 declared plane files') == 1
    assert core._watch_described_ready(context, 'good', signature, 100.0) is True
    for name in ('p0.tif', 'p1.tif'):
        (tmp_path / name).unlink()
    assert core._watch_described_ready(context, 'good', signature, 100.0) is True
    assert core._watch_described_ready(context, 'good', {'p0.tif': [11, 2]}, 100.0) is False


def test_collection_refuses_a_missing_or_changed_vendor_copy(tmp_path):
    with pytest.raises(ValueError, match='vendor stack copy is missing or unsafe'):
        core._watch_collection_artifacts(str(tmp_path), vendor={'acq.tif': 'x'})
    copies = tmp_path / '.watch_vendor'
    copies.mkdir()
    (copies / 'acq.tif').write_bytes(b'copy')
    with pytest.raises(ValueError, match='vendor stack copy changed after its snapshot'):
        core._watch_collection_artifacts(str(tmp_path), vendor={'acq.tif': '0' * 64})
    (copies / 'other.tif').write_bytes(b'extra')
    with pytest.raises(ValueError, match='missing or unsafe'):
        core._watch_collection_artifacts(str(tmp_path), vendor={'acq.tif': '0' * 64})


def test_linked_snapshot_reuses_only_an_unchanged_private_copy(tmp_path):
    source = tmp_path / 'copy.nd2'
    source.write_bytes(b'container bytes')
    expected = [1, 2, source.stat().st_size, 3, 4]
    shared = (expected, str(source), 'digest')
    assert core._watch_linked_snapshot(None, str(tmp_path / 'a'), expected) is None
    assert core._watch_linked_snapshot(shared, str(tmp_path / 'a'), [9] * 5) is None
    grown = (expected[:2] + [expected[2] + 1] + expected[3:])
    assert core._watch_linked_snapshot((grown, str(source), 'digest'),
                                       str(tmp_path / 'a'), grown) is None
    assert not (tmp_path / 'a').exists()
    target = tmp_path / 'b.nd2'
    assert core._watch_linked_snapshot(shared, str(target), expected) == 'digest'
    assert os.path.samefile(target, source)
    assert core._watch_linked_snapshot(shared, str(target), expected) is None
    assert core._watch_linked_snapshot((expected, str(tmp_path / 'gone'), 'digest'),
                                       str(tmp_path / 'c'), expected) is None


@pytest.mark.parametrize('vendor', ['volume', 'series', 'timelapse'])
def test_resumed_vendor_field_restores_its_plan_before_checking_inputs(tmp_path, vendor):
    key, plan = 'acq', {'names': ['plane.tif']}
    ledger = {'src': str(tmp_path), 'fields': {key: {
        'collection_checkpoint': {'artifacts': {}}, 'vendor_plan': plan,
        'files': {'acq.tif': [1, 1]}, 'stable_since': 5.0}}}
    context = _context(tmp_path, _seen('acq.tif'), ledger=ledger,
                       ledger_path=str(tmp_path / 'ledger.json'),
                       work=str(tmp_path / 'work'), tried=set(), vendor=vendor,
                       volume_plan={}, native_series_plan={})
    core._watch_run_field(key, [('acq.tif', None)], {'acq.tif': [2, 2]}, context)
    entry = ledger['fields'][key]
    assert entry['status'] == 'failed'
    assert 'Collection checkpoint inputs changed' in entry['error']
    assert context['volume_plan'] == ({key: plan} if vendor == 'volume' else {})
    assert context['native_series_plan'] == ({key: plan} if vendor == 'series' else {})
    saved = json.loads((tmp_path / 'ledger.json').read_text())
    assert saved['fields'][key]['status'] == 'failed'


SERIES_REGEX = (r'(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+)_T(?P<timeID>\d+)F(?P<fieldID>\d+)'
                r'L\d+A\d+Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif')


def test_native_series_map_needs_cellvoyager_target_names(tmp_path):
    source, _rows = _converted_series(tmp_path)
    with pytest.raises(ValueError, match='native T-by-Z requires a fixed Convert map'):
        core._watch_folder_and_analyse(_series_watch(
            source, metadata_type='custom', custom_regex=SERIES_REGEX))
    assert not (source / 'spacr_watch').exists()


def test_saved_record_of_another_stack_ingest_is_refused(tmp_path):
    tifffile.imwrite(tmp_path / 'acq.tif', np.ones((2, 1, 8, 8), np.uint16), imagej=True,
                     metadata={'axes': 'TCYX'})
    first = core._watch_folder_and_analyse(
        dict(MASK, **_fast(tmp_path), timelapse=True), Recorder())
    assert first['failed'] == ['acq'] and first['done'] == []
    ledger = json.loads((tmp_path / 'spacr_watch' / 'watch_ledger.json').read_text())
    assert ledger['vendor_stacks'] == 'timelapse'
    ledger['vendor_stacks'] = 'volume'
    (tmp_path / 'spacr_watch' / 'watch_ledger.json').write_text(json.dumps(ledger))
    recorder = Recorder()
    with pytest.raises(ValueError, match='made with a different stack ingest'):
        core._watch_folder_and_analyse(
            dict(MASK, **_fast(tmp_path), timelapse=True), recorder)
    assert recorder.calls == []
