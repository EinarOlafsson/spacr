"""History evidence must fail on a wrong row, stale panel or modified journal."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_evidence import read_record, snapshot, verify_panels, verify_preserved, verify_visible


@pytest.fixture
def example():
    settings = {'src': '/private/example', 'plot': True}
    manifest = {'app_key': 'recruitment', 'status': 'success', 'start_utc': '2026-09-10T01:00Z',
                'end_utc': '2026-09-10T01:01Z', 'performance': {'input_files': 2, 'input_bytes': 32,
                    'output_files': 1, 'output_bytes': 15},
                'input_hashes': {'in.tif': {'sha256': 'a'*64}},
                'output_hashes': {'out.csv': {'sha256': 'b'*64}}, 'model_files': {},
                'env': {'python': '3.12'}, 'seeds': {'seed': 42}, 'settings_sha256': 'c'*64,
                'input_tree_sha256': 'd'*64, 'output_tree_sha256': 'e'*64,
                'schema_version': 2, 'warnings': ['Actual warning'],
                'provenance_warnings': [], 'traceback': ''}
    record = {'run_id': 'run-one', 'manifest': manifest, 'settings': settings}
    panels = {'overview': json.dumps({'run_id': 'run-one', 'module': 'recruitment',
        'status': 'success', 'started_utc': '2026-09-10T01:00Z', 'ended_utc': '2026-09-10T01:01Z',
        'input_files': 2, 'input_bytes': 32, 'output_files': 1, 'output_bytes': 15}),
        'settings': '{"src":"/private/example","plot":true}',
        'files': json.dumps({'inputs': {'in.tif': {'sha256': 'a'*64}},
                             'outputs': {'out.csv': {'sha256': 'b'*64}}, 'models': {}}),
        'environment': json.dumps({'environment': {'python': '3.12'}, 'seeds': {'seed': 42},
           'settings_sha256': 'c'*64, 'input_tree_sha256': 'd'*64, 'output_tree_sha256': 'e'*64,
           'manifest_schema': 2}),
        'problems': 'WARNINGS\nActual warning\n\nFAILURE TRACEBACK\nNone.'}
    return record, panels


def test_positive_all_five_tabs_match(example):
    result = verify_panels(*example)
    assert result['all_five_panels_match_raw_journal']
    assert result['warning_count'] == 1
    assert result['recorded_output_files'] == 1


@pytest.mark.parametrize('key', ['run_id', 'module', 'status', 'started_utc', 'ended_utc',
                                 'input_files', 'input_bytes', 'output_files', 'output_bytes'])
def test_other_identity_status_or_counts_are_rejected(example, key):
    record, panels = example
    value = json.loads(panels['overview'])
    value[key] = 'wrong'
    panels['overview'] = json.dumps(value)
    with pytest.raises(ValueError, match='History overview differs'):
        verify_panels(record, panels)


@pytest.mark.parametrize('tab', ['settings', 'files', 'environment', 'problems'])
def test_empty_or_stale_tab_is_not_accepted(example, tab):
    record, panels = example
    panels[tab] = '{}' if tab != 'problems' else 'No problems.'
    with pytest.raises(ValueError, match='History'):
        verify_panels(record, panels)


def test_positive_then_missing_real_failure_is_rejected(example):
    record, panels = example
    record['manifest']['traceback'] = 'TypeError: genuine failure'
    panels['problems'] = 'WARNINGS\nActual warning\n\nFAILURE TRACEBACK\nTypeError: genuine failure'
    assert verify_panels(record, panels)['failure_present']
    panels['problems'] = 'WARNINGS\nActual warning\n\nFAILURE TRACEBACK\nNone.'
    with pytest.raises(ValueError, match='failure differ'):
        verify_panels(record, panels)


def test_snapshot_hash_and_preservation_have_positive_counterparts(tmp_path):
    folder = tmp_path/'run-one'
    folder.mkdir()
    raw = b'{"plot": true}'
    (folder/'settings.json').write_bytes(raw)
    (folder/'manifest.json').write_text(json.dumps({'settings_sha256': hashlib.sha256(raw).hexdigest()}))
    assert read_record(tmp_path, 'run-one')['settings'] == {'plot': True}
    before = snapshot(tmp_path)
    assert verify_preserved(tmp_path, before) == {'files': 2, 'unchanged': True}
    (folder/'settings.json').write_bytes(b'{"plot": false}')
    with pytest.raises(ValueError, match='settings hash'):
        read_record(tmp_path, 'run-one')
    with pytest.raises(ValueError, match='files changed'):
        verify_preserved(tmp_path, before)


def test_missing_journal_file_is_detected(tmp_path):
    (tmp_path/'first').write_text('one')
    (tmp_path/'second').write_text('two')
    before = snapshot(tmp_path)
    assert verify_preserved(tmp_path, before)['unchanged']
    (tmp_path/'second').rename(tmp_path.parent/(tmp_path.name+'-preserved-second'))
    with pytest.raises(ValueError, match='files changed'):
        verify_preserved(tmp_path, before)


@pytest.mark.parametrize('actual', [['one'], ['one', 'one'], ['one', 'other']])
def test_wrong_missing_duplicate_filtered_records_fail(actual):
    assert verify_visible(['two', 'one'], ['one', 'two']) == ['two', 'one']
    with pytest.raises(ValueError, match='Visible run identities differ'):
        verify_visible(actual, ['one', 'two'])


def test_empty_view_is_valid_only_for_an_empty_expectation():
    assert verify_visible([], []) == []
    with pytest.raises(ValueError, match='Visible run identities differ'):
        verify_visible([], ['real-run'])
