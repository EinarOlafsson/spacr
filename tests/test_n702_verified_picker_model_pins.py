"""Real tiny local bytes bind picker identities before pure export attachment."""
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import os

import pytest

from spacr import model_zoo as zoo


@pytest.fixture
def selection(tmp_path, monkeypatch):
    data = b'controlled trusted checkpoint bytes'
    path = tmp_path / 'renamed-cache-v7.pt'
    path.write_bytes(data)
    raw = {'key': 'controlled_cellpose', 'name': 'publisher_weights',
           'kind': 'cellpose', 'sha256': hashlib.sha256(data).hexdigest(),
           'repo_id': 'controlled/fixture', 'repo_type': 'model', 'version': '001'}
    other = {**raw, 'key': 'controlled_detector', 'name': 'detector.pt', 'kind': 'detector'}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', (raw, other))
    entry = replace(zoo._entry_from_mapping(raw), version='17')
    return path, raw, other, entry


def test_real_verified_selection_pins_renamed_path_without_inventing_version(selection):
    path, raw, _, entry = selection
    verified = zoo._verified_model_reference(entry, str(path))
    record = dict(verified.record)
    assert record == zoo._model_reference_from_record(str(path), raw)
    assert record['version'] == '001'
    assert record['version'] != entry.version
    assert verified.path == str(path)
    with pytest.raises(FrozenInstanceError):
        verified.path = 'changed'


def test_attach_preserves_independent_existing_pins_and_never_rehashes(selection, monkeypatch):
    path, raw, other, entry = selection
    verified = zoo._verified_model_reference(entry, str(path))
    previous = zoo._model_reference_from_record('/existing/detector.pt', other)
    settings = {'plaque_model': str(path), 'well_detection': '/existing/detector.pt',
                'model_metadata': {'schema_version': 1, 'settings': {'well_detection': previous}}}
    original = copy.deepcopy(settings)
    monkeypatch.setattr(zoo.hashlib, 'sha256', lambda *a: pytest.fail('GUI attachment hashed bytes'))
    result = zoo._attach_model_reference('analyze_plaques', settings, 'plaque_model', verified)
    assert settings == original
    assert result['model_metadata']['settings']['well_detection'] == previous
    assert result['model_metadata']['settings']['plaque_model'] == zoo._model_reference_from_record(str(path), raw)
    result['model_metadata']['settings']['well_detection']['key'] = 'changed'
    assert settings == original
    assert dict(verified.record)['key'] == raw['key']


@pytest.mark.parametrize('change', ['key', 'name', 'kind', 'sha256', 'stock'])
def test_custom_same_basename_or_untrusted_selection_is_never_verified(selection, change):
    path, _, _, entry = selection
    updates = ({'source': 'stock'} if change == 'stock' else
               {change: 'detector' if change == 'kind' else 'different'})
    assert zoo._verified_model_reference(replace(entry, **updates), str(path)) is None


def test_known_selected_entry_wrong_bytes_refuses_without_changing_input(selection):
    path, _, _, entry = selection
    path.write_bytes(b'custom unrelated bytes with a familiar filename')
    before = path.read_bytes()
    with pytest.raises(zoo.ModelZooError):
        zoo._verified_model_reference(entry, str(path))
    assert path.read_bytes() == before


@pytest.mark.parametrize('at', [1, 2, 3])
def test_cancellation_before_between_or_after_reads_never_issues_a_pin(selection, at):
    path, _, _, entry = selection
    calls = []

    def cancelled():
        calls.append(None)
        return len(calls) == at

    before = path.read_bytes()
    with pytest.raises(zoo.DownloadCancelled):
        zoo._verified_model_reference(entry, str(path), cancelled=cancelled)
    assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['rewrite', 'grow'])
def test_real_file_mutation_during_stream_refuses(selection, monkeypatch, change):
    path, _, _, entry = selection
    real_sha = hashlib.sha256

    class Digest:
        def __init__(self):
            self.actual = real_sha()
            self.changed = False

        def update(self, block):
            self.actual.update(block)
            if not self.changed:
                if change == 'grow':
                    with path.open('ab') as handle:
                        handle.write(b'x')
                else:
                    path.write_bytes(b'x' * path.stat().st_size)
                self.changed = True

        def hexdigest(self):
            return self.actual.hexdigest()

    monkeypatch.setattr(zoo.hashlib, 'sha256', Digest)
    with pytest.raises(zoo.ModelZooError):
        zoo._verified_model_reference(entry, str(path))


@pytest.mark.parametrize('change', ['replace', 'rewrite', 'remove'])
def test_file_changed_after_worker_completion_refuses_pure_attach(selection, change):
    path, _, _, entry = selection
    verified = zoo._verified_model_reference(entry, str(path))
    if change == 'replace':
        other = path.with_name('replacement')
        other.write_bytes(path.read_bytes())
        os.replace(other, path)
    elif change == 'rewrite':
        path.write_bytes(b'x' * path.stat().st_size)
    else:
        path.unlink()
    settings = {'plaque_model': str(path)}
    original = copy.deepcopy(settings)
    with pytest.raises(zoo.ModelZooError):
        zoo._attach_model_reference('analyze_plaques', settings, 'plaque_model', verified)
    assert settings == original


@pytest.mark.parametrize('change', ['raw_dict', 'unverified_capsule', 'request', 'kind', 'existing_pin'])
def test_forged_or_stale_consumer_context_cannot_attach_a_pin(selection, change):
    path, _, other, entry = selection
    verified = zoo._verified_model_reference(entry, str(path))
    settings = {'plaque_model': str(path)}
    if change == 'raw_dict':
        verified = dict(verified.record)
    elif change == 'unverified_capsule':
        verified = replace(verified, _verification=None)
    elif change == 'request':
        settings['plaque_model'] += '-edited'
    elif change == 'kind':
        settings = {'well_detection': str(path)}
    else:
        settings['well_detection'] = '/actual/request'
        bad = zoo._model_reference_from_record('/stale/request', other)
        settings['model_metadata'] = {'schema_version': 1, 'settings': {'well_detection': bad}}
    original = copy.deepcopy(settings)
    key = 'well_detection' if change == 'kind' else 'plaque_model'
    with pytest.raises(zoo.ModelZooError):
        zoo._attach_model_reference('analyze_plaques', settings, key, verified)
    assert settings == original


def test_arbitrary_requested_value_cannot_claim_a_different_checked_file(selection):
    path, _, _, entry = selection
    with pytest.raises(zoo.ModelZooError):
        zoo._verified_model_reference(entry, str(path), requested_value='/unrelated/model')


def test_existing_backend_prefix_preserves_exact_request_and_physical_path(selection, monkeypatch):
    path, raw, _, entry = selection
    raw = {**raw, 'kind': 'cellpose3'}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', (raw,))
    entry = replace(entry, kind='cellpose3')
    requested = 'cellpose3:' + str(path)
    verified = zoo._verified_model_reference(entry, str(path), requested_value=requested)
    assert dict(verified.record)['requested_value'] == requested
    assert verified.path == str(path)
    result = zoo._attach_model_reference('mask', {'cell_model_name': requested},
                                         'cell_model_name', verified)
    assert result['model_metadata']['settings']['cell_model_name']['kind'] == 'cellpose3'


@pytest.mark.skipif(os.name != 'posix', reason='POSIX named FIFO control')
def test_nonregular_fifo_refuses_before_any_blocking_read(selection):
    path, _, _, entry = selection
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(zoo.ModelZooError):
        zoo._verified_model_reference(entry, str(path))
